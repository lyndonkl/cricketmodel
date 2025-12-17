"""
HeteroConv Builder for Cricket GNN

Constructs HeteroConv layers with appropriate convolution operators
for each edge type in the heterogeneous graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, TransformerConv
from typing import Dict, List, Tuple, Optional

# Edge type definition
EdgeType = Tuple[str, str, str]


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
            convs[edge_type] = SAGEConv(
                hidden_dim,
                hidden_dim,
                aggr='mean',
            )

        elif rel in ['same_bowler', 'same_batsman', 'same_matchup']:
            # Actor grouping: attention over grouped balls
            # same_matchup is the intersection of same_bowler and same_batsman
            convs[edge_type] = GATv2Conv(
                hidden_dim,
                head_dim,
                heads=num_heads,
                add_self_loops=False,
                concat=True,
                dropout=dropout,
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
            # Dynamics aggregation: simple mean is appropriate
            # as all recent balls contribute equally to momentum/pressure
            convs[edge_type] = SAGEConv(
                hidden_dim,
                hidden_dim,
                aggr='mean',
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
            convs[edge_type] = SAGEConv(
                hidden_dim,
                hidden_dim,
                aggr='mean',
            )

    return HeteroConv(convs, aggr='sum')


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
