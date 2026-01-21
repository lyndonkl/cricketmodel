"""
HeteroConv Builder for Cricket GNN

Constructs HeteroConv layers with appropriate convolution operators
for each edge type in the heterogeneous graph.

Includes FiLM (Feature-wise Linear Modulation) for phase-conditional
message passing, allowing the model to modulate its behavior based
on the current game phase (powerplay/middle/death).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, TransformerConv
from typing import Dict, List, Tuple, Optional

# Edge type definition
EdgeType = Tuple[str, str, str]


class SAGEConvWrapper(nn.Module):
    """
    Wrapper around SAGEConv that ignores edge_attr parameter.

    SAGEConv doesn't support edge attributes, but HeteroConv forwards
    edge_attr_dict to all convolutions. This wrapper drops edge_attr
    to avoid TypeError.
    """
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean'):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels, aggr=aggr)

    def forward(self, x, edge_index, edge_attr=None):
        # Ignore edge_attr - SAGEConv doesn't use it
        return self.conv(x, edge_index)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    FiLM modulates neural network activations based on conditioning input:
        output = gamma * input + beta

    where gamma and beta are learned functions of the conditioning signal.

    In cricket context, this allows situation-specific behavior:
    - Phase (powerplay/middle/death): Tactical mode adaptation
    - Chase pressure (RRR): Risk tolerance modulation
    - Resource state (wickets/balls): Strategic urgency

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018
    """

    def __init__(self, condition_dim: int, hidden_dim: int):
        """
        Args:
            condition_dim: Dimension of conditioning signal (e.g., phase + chase + resource)
            hidden_dim: Dimension of features to modulate
        """
        super().__init__()

        # Generate gamma and beta from conditioning signal
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma and beta
        )

        # Initialize close to identity: gamma=1, beta=0
        with torch.no_grad():
            # Last layer: initialize to small values so gamma ≈ 1, beta ≈ 0
            self.film_generator[-1].weight.fill_(0.0)
            self.film_generator[-1].bias.zero_()
            # Set gamma bias to 1 (first half of output)
            self.film_generator[-1].bias[:hidden_dim].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: Features to modulate [num_nodes, hidden_dim]
            condition: Conditioning signal [batch_size, condition_dim]
            batch: Optional batch indices [num_nodes] mapping each node to its graph.
                   Used to expand graph-level conditioning to node-level.

        Returns:
            Modulated features [num_nodes, hidden_dim]
        """
        # Generate gamma and beta
        film_params = self.film_generator(condition)  # [batch_size, hidden_dim * 2]
        gamma, beta = film_params.chunk(2, dim=-1)    # Each: [batch_size, hidden_dim]

        # Expand graph-level conditioning to node-level using batch indices
        if batch is not None and gamma.shape[0] != x.shape[0]:
            # Use batch tensor to index: [batch_size, dim] -> [num_nodes, dim]
            gamma = gamma[batch]
            beta = beta[batch]
        elif gamma.shape[0] == 1 and x.shape[0] > 1:
            # Fallback: broadcast single condition to all nodes
            gamma = gamma.expand(x.shape[0], -1)
            beta = beta.expand(x.shape[0], -1)

        # Apply modulation
        return gamma * x + beta


def get_edge_types_for_conv() -> List[EdgeType]:
    """
    Get all edge types that should have convolution operators.

    Returns:
        List of (src_type, relation, tgt_type) tuples
    """
    from ..data.edge_builder import get_all_edge_types
    return get_all_edge_types()


def build_hetero_conv(
    hidden_dim: int,
    num_heads: int = 4,
    dropout: float = 0.1,
    edge_types: Optional[List[EdgeType]] = None,
) -> HeteroConv:
    """
    Build a HeteroConv layer with appropriate convolutions per edge type.

    Different edge types use different convolution operators:
    - Hierarchical conditioning: GATv2Conv (attention-based)
    - Intra-layer relations: GATv2Conv (attention-based)
    - Actor matchup: GATv2Conv (attention-based for matchup importance)
    - Temporal precedes: TransformerConv (position-aware)
    - Same bowler/batsman: GATv2Conv (attention over actor's balls)
    - Cross-domain (ball -> context): SAGEConv (simple aggregation)
    - Query aggregation: GATv2Conv (attention-weighted aggregation)

    Args:
        hidden_dim: Hidden dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Dropout rate
        edge_types: List of edge types (defaults to all edge types)

    Returns:
        HeteroConv module
    """
    if edge_types is None:
        edge_types = get_edge_types_for_conv()

    assert hidden_dim % num_heads == 0, \
        f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

    head_dim = hidden_dim // num_heads
    convs = {}

    for edge_type in edge_types:
        src_type, rel, tgt_type = edge_type

        if rel == 'conditions':
            # Hierarchical conditioning: use attention
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,  # Output: num_heads * head_dim = hidden_dim
                dropout=dropout,
            )

        elif rel == 'relates_to':
            # Intra-layer relations: use attention
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel == 'matchup':
            # Actor matchup: attention for importance weighting
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel in ['recent_precedes', 'medium_precedes']:
            # Multi-scale temporal edges: position-aware attention with distance features
            # - recent_precedes: Last 6 balls (within-over context, fast decay)
            # - medium_precedes: 7-18 balls (momentum window, medium decay)
            # TransformerConv with edge features allows the model to learn
            # scale-appropriate attention patterns
            convs[edge_type] = TransformerConv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=1,  # Temporal distance feature dimension
            )

        elif rel == 'distant_precedes':
            # Distant temporal context: simpler aggregation for sparse historical connections
            # These edges are sparser (every 6 balls) so mean aggregation is efficient
            convs[edge_type] = SAGEConvWrapper(
                hidden_dim,
                hidden_dim,
                aggr='mean',
            )

        elif rel in ['same_bowler', 'same_batsman']:
            # Actor grouping with temporal decay: recent balls in spell/form matter more
            # Uses TransformerConv with edge_dim to incorporate temporal distance
            # This allows the model to learn spell/form decay patterns
            convs[edge_type] = TransformerConv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=1,  # Temporal distance feature
            )

        elif rel == 'same_matchup':
            # Specific bowler-batsman matchup history (causal edges only)
            # No edge attributes needed - all matchup history is equally relevant
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel == 'same_over':
            # Within-over structural edges: tight local coherence
            # Same bowler rhythm, batsmen haven't swapped, consistent field
            # Use TransformerConv with ball-in-over position as edge attribute
            # Position in over is semantically meaningful (ball 1 vs ball 6 differ)
            convs[edge_type] = TransformerConv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=1,  # Ball-in-over position
            )

        elif rel in ['faced_by', 'bowled_by', 'partnered_by']:
            # Cross-domain edges: Use attention to weight recent/relevant balls more
            # This allows the model to learn which historical balls are most informative
            # for predicting the current situation (recency, similar game state, etc.)
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel == 'informs':
            # Dynamics aggregation: attention-weighted for importance
            # Boundaries and wickets contribute more to momentum/pressure than dots
            # GATv2Conv allows learning which recent balls matter most
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel == 'attends':
            # Query aggregation: attention for selective aggregation
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        elif rel == 'drives':
            # Dynamics -> Query: momentum/pressure directly drive predictions
            # Use attention-based aggregation so the model can weight different
            # dynamics signals (batting_momentum, bowling_momentum, pressure, dots)
            # This is critical for capturing feedback loops like:
            # - R1: Confidence spiral (momentum -> more runs)
            # - B1: Required rate pressure (pressure -> risk-taking)
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
            )

        else:
            # Default: simple message passing
            convs[edge_type] = SAGEConvWrapper(
                hidden_dim,
                hidden_dim,
                aggr='mean',
            )

    return HeteroConv(convs, aggr='sum')


class PhaseModulatedConvBlock(nn.Module):
    """
    Heterogeneous convolution block with enhanced FiLM modulation.

    Wraps a standard HeteroConvBlock and applies FiLM modulation
    to all node representations after message passing. The modulation
    is conditioned on the current game situation including:
    - Phase state (powerplay/middle/death)
    - Chase state (RRR, difficulty - 2nd innings only)
    - Resource state (wickets/balls remaining)

    This allows the model to learn situation-specific message passing patterns:
    - Different attention patterns during powerplay vs death overs
    - Chase-pressure-appropriate weighting of historical context
    - Adaptive aggregation based on game resources

    Structure:
    1. HeteroConv message passing
    2. Residual connection
    3. Layer normalization
    4. FiLM modulation (per node type)
    5. Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int = 14,  # phase(5) + chase(7) + wicket_buffer(2) = 14
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_types: Optional[List[EdgeType]] = None,
        node_types: Optional[List[str]] = None,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            condition_dim: Total dimension of conditioning signals (phase + chase + resource)
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_types: List of edge types for convolution
            node_types: List of node types for layer norms and FiLM
        """
        super().__init__()

        if node_types is None:
            from ..data.edge_builder import NODE_TYPES
            node_types = NODE_TYPES

        self.node_types = node_types

        # Core convolution
        self.conv = build_hetero_conv(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_types=edge_types,
        )

        # Layer norm per node type
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })

        # FiLM modulation per node type with enhanced conditioning
        self.film_layers = nn.ModuleDict({
            node_type: FiLMLayer(condition_dim, hidden_dim)
            for node_type in node_types
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, 'torch.Tensor'],
        edge_index_dict: Dict[EdgeType, 'torch.Tensor'],
        condition: 'torch.Tensor',
        edge_attr_dict: Optional[Dict[EdgeType, 'torch.Tensor']] = None,
        batch_dict: Optional[Dict[str, 'torch.Tensor']] = None,
    ) -> Dict[str, 'torch.Tensor']:
        """
        Forward pass with enhanced situation-conditioned FiLM modulation.

        Args:
            x_dict: Dict of node features {node_type: [num_nodes, hidden_dim]}
            edge_index_dict: Dict of edge indices {edge_type: [2, num_edges]}
            condition: Concatenated situation features [batch_size, condition_dim]
                       (phase_state + chase_state + wicket_buffer)
            edge_attr_dict: Optional dict of edge attributes
            batch_dict: Optional dict mapping node types to batch indices
                        {node_type: [num_nodes]} for expanding graph-level
                        conditioning to node-level

        Returns:
            Updated x_dict with same structure
        """
        # Message passing with edge attributes if provided
        if edge_attr_dict is not None:
            out_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            out_dict = self.conv(x_dict, edge_index_dict)

        # Residual + Norm + FiLM + Dropout
        result = {}
        for node_type in out_dict:
            h = out_dict[node_type]

            # Residual connection (if node type was in input)
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                h = h + x_dict[node_type]

            # Layer norm
            if node_type in self.norms:
                h = self.norms[node_type](h)

            # FiLM modulation (with batch indices for graph->node expansion)
            if node_type in self.film_layers:
                batch = batch_dict.get(node_type) if batch_dict else None
                h = self.film_layers[node_type](h, condition, batch=batch)

            # Dropout
            h = self.dropout(h)

            result[node_type] = h

        # Preserve node types that weren't updated by conv
        for node_type in x_dict:
            if node_type not in result:
                result[node_type] = x_dict[node_type]

        return result


class HeteroConvBlock(nn.Module):
    """
    A single heterogeneous convolution block with residual connection,
    layer normalization, and dropout.

    Structure:
    1. HeteroConv message passing
    2. Residual connection (skip connection from input)
    3. Layer normalization per node type
    4. Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_types: Optional[List[EdgeType]] = None,
        node_types: Optional[List[str]] = None,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_types: List of edge types for convolution
            node_types: List of node types for layer norms
        """
        super().__init__()

        if node_types is None:
            from ..data.edge_builder import NODE_TYPES
            node_types = NODE_TYPES

        self.conv = build_hetero_conv(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_types=edge_types,
        )

        # Layer norm per node type
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, 'torch.Tensor'],
        edge_index_dict: Dict[EdgeType, 'torch.Tensor'],
        edge_attr_dict: Optional[Dict[EdgeType, 'torch.Tensor']] = None,
    ) -> Dict[str, 'torch.Tensor']:
        """
        Forward pass with residual connection and normalization.

        Args:
            x_dict: Dict of node features {node_type: [num_nodes, hidden_dim]}
            edge_index_dict: Dict of edge indices {edge_type: [2, num_edges]}
            edge_attr_dict: Optional dict of edge attributes {edge_type: [num_edges, edge_dim]}

        Returns:
            Updated x_dict with same structure
        """
        # Message passing with edge attributes if provided
        if edge_attr_dict is not None:
            out_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        else:
            out_dict = self.conv(x_dict, edge_index_dict)

        # Residual + Norm + Dropout
        result = {}
        for node_type in out_dict:
            h = out_dict[node_type]

            # Residual connection (if node type was in input)
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                h = h + x_dict[node_type]

            # Layer norm
            if node_type in self.norms:
                h = self.norms[node_type](h)

            # Dropout
            h = self.dropout(h)

            result[node_type] = h

        # Preserve node types that weren't updated by conv
        for node_type in x_dict:
            if node_type not in result:
                result[node_type] = x_dict[node_type]

        return result


def build_conv_stack(
    num_layers: int,
    hidden_dim: int,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.ModuleList:
    """
    Build a stack of HeteroConvBlocks.

    Args:
        num_layers: Number of message passing layers
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        ModuleList of HeteroConvBlocks
    """
    return nn.ModuleList([
        HeteroConvBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])


def build_phase_modulated_conv_stack(
    num_layers: int,
    hidden_dim: int,
    condition_dim: int = 14,  # phase(5) + chase(7) + wicket_buffer(2) = 14
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.ModuleList:
    """
    Build a stack of PhaseModulatedConvBlocks with enhanced conditioning.

    Each layer applies FiLM modulation conditioned on game situation:
    - Phase state (5 features): powerplay/middle/death, over_progress, is_first_ball
    - Chase state (7 features): runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem
    - Wicket buffer (2 features): wickets_remaining, is_tail

    This allows situation-specific message passing patterns for
    different game states.

    Args:
        num_layers: Number of message passing layers
        hidden_dim: Hidden dimension
        condition_dim: Total dimension of conditioning signals (default 14)
        num_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        ModuleList of PhaseModulatedConvBlocks
    """
    return nn.ModuleList([
        PhaseModulatedConvBlock(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])
