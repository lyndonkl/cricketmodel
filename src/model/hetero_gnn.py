"""
Cricket Heterogeneous GNN Model

Main model that combines node encoders, heterogeneous message passing,
and prediction head for ball outcome prediction.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional

from .encoders import NodeEncoderDict
from .conv_builder import build_conv_stack, build_phase_modulated_conv_stack


@dataclass
class ModelConfig:
    """Configuration for CricketHeteroGNN."""

    # Entity counts (from entity mapper)
    num_venues: int
    num_teams: int
    num_players: int

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4

    # Embedding dimensions
    venue_embed_dim: int = 32
    team_embed_dim: int = 32
    player_embed_dim: int = 64
    role_embed_dim: int = 16  # For hierarchical player encoder

    # Number of player role categories (for hierarchical embeddings)
    num_roles: int = 8  # unknown + 7 roles

    # Regularization
    dropout: float = 0.1

    # Task
    num_classes: int = 7  # Dot, Single, Two, Three, Four, Six, Wicket

    # Model variants
    use_hybrid_readout: bool = True  # Use matchup + query hybrid readout
    use_innings_conditional: bool = True  # Use separate heads for 1st/2nd innings
    use_hierarchical_player: bool = True  # Use hierarchical player embeddings
    use_phase_modulation: bool = True  # Use FiLM phase-conditional message passing

    # Phase conditioning dimension (matches phase_state features)
    phase_dim: int = 5


class CricketHeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for Cricket Ball Prediction.

    Architecture:
    1. Node Encoders: Project each node type to common hidden_dim
    2. Message Passing: Stack of HeteroConvBlocks
    3. Readout: Extract query node representation
    4. Prediction: MLP classifier

    The model uses the query node as a learnable aggregation point.
    After message passing, information from all context nodes flows
    to the query node, which is then used for prediction.
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # === Node Encoders ===
        self.encoders = NodeEncoderDict(
            hidden_dim=config.hidden_dim,
            num_venues=config.num_venues,
            num_teams=config.num_teams,
            num_players=config.num_players,
            num_roles=config.num_roles,
            venue_embed_dim=config.venue_embed_dim,
            team_embed_dim=config.team_embed_dim,
            player_embed_dim=config.player_embed_dim,
            role_embed_dim=config.role_embed_dim,
            dropout=config.dropout,
            use_hierarchical_player=config.use_hierarchical_player,
        )

        # === Message Passing Layers ===
        self.conv_stack = build_conv_stack(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # === Prediction Head ===
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Message passing with edge attributes (e.g., temporal distance for precedes edges)
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict if edge_attr_dict else None)

        # 3. Readout from query nodes
        query_repr = x_dict['query']  # [batch_size, hidden_dim]

        # 4. Predict
        logits = self.predictor(query_repr)

        return logits

    def get_attention_weights(
        self,
        data,
        layer_idx: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from a specific layer (for interpretability).

        Args:
            data: HeteroData
            layer_idx: Which layer to extract from (-1 for last)

        Returns:
            Dict mapping edge type to attention weights
        """
        # This would require modifying the forward pass to return attention
        # For now, placeholder
        raise NotImplementedError(
            "Attention weight extraction requires custom implementation"
        )

    @classmethod
    def from_dataset_metadata(
        cls,
        metadata: dict,
        **kwargs
    ) -> 'CricketHeteroGNN':
        """
        Create model from dataset metadata.

        Args:
            metadata: Dict from CricketDataset.get_metadata()
            **kwargs: Override default config values

        Returns:
            Initialized model
        """
        config = ModelConfig(
            num_venues=metadata['num_venues'],
            num_teams=metadata['num_teams'],
            num_players=metadata['num_players'],
            **kwargs
        )
        return cls(config)


class CricketHeteroGNNWithPooling(CricketHeteroGNN):
    """
    Variant that also pools ball node representations.

    Instead of relying solely on the query node, this variant
    also performs explicit pooling over ball nodes and combines
    the result with the query representation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Additional pooling layers for ball nodes
        self.ball_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Combine query and pooled ball representations
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with ball pooling.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Message passing with edge attributes
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict if edge_attr_dict else None)

        # 3. Query representation
        query_repr = x_dict['query']  # [batch_size, hidden_dim]

        # 4. Pool ball nodes with attention
        ball_repr = x_dict['ball']  # [total_balls, hidden_dim]

        if ball_repr.shape[0] > 0:
            # Compute attention scores
            attn_scores = self.ball_attention(ball_repr)  # [total_balls, 1]
            attn_weights = torch.softmax(attn_scores, dim=0)

            # Weighted sum (simplified - would need batch handling)
            pooled_balls = (attn_weights * ball_repr).sum(dim=0, keepdim=True)

            # Expand to match batch size
            batch_size = query_repr.shape[0]
            pooled_balls = pooled_balls.expand(batch_size, -1)
        else:
            # No balls - use zeros
            pooled_balls = torch.zeros_like(query_repr)

        # 5. Combine
        combined = torch.cat([query_repr, pooled_balls], dim=-1)
        combined = self.combiner(combined)

        # 6. Predict
        logits = self.predictor(combined)

        return logits


class CricketHeteroGNNHybrid(CricketHeteroGNN):
    """
    Hybrid readout variant that combines matchup interaction with query aggregation.

    Cricket ball prediction is most naturally an edge-level task: the outcome depends
    on the specific striker-bowler interaction, modulated by context. This variant:

    1. Computes matchup interaction from striker + bowler representations
    2. Uses query node to aggregate global context (venue, phase, momentum)
    3. Combines matchup + context for final prediction

    This respects the GDL principle that the prediction task level should match
    the structural level of the phenomenon being predicted.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Matchup interaction layer
        # Projects concatenated striker + bowler representations
        self.matchup_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Query projection (for context aggregation)
        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Combine matchup and context
        # Input: matchup (hidden_dim) + context (hidden_dim) = 2 * hidden_dim
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with hybrid matchup + query readout.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Message passing with edge attributes
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict if edge_attr_dict else None)

        # 3. Matchup interaction (edge-level)
        # The core prediction depends on striker-bowler interaction
        striker = x_dict['striker_identity']  # [batch_size, hidden_dim]
        bowler = x_dict['bowler_identity']    # [batch_size, hidden_dim]
        matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))  # [batch_size, hidden_dim]

        # 4. Context aggregation (graph-level)
        # Query has aggregated global context: venue, phase, momentum, etc.
        query = self.query_proj(x_dict['query'])  # [batch_size, hidden_dim]

        # 5. Combine matchup + context
        # Matchup is modulated by context (pressure, phase, momentum)
        combined = torch.cat([matchup, query], dim=-1)  # [batch_size, 2*hidden_dim]
        combined = self.combiner(combined)  # [batch_size, hidden_dim]

        # 6. Predict
        logits = self.predictor(combined)

        return logits


class CricketHeteroGNNPhaseModulated(CricketHeteroGNNHybrid):
    """
    Phase-modulated variant using FiLM conditioning.

    Applies Feature-wise Linear Modulation (FiLM) to all node representations
    after each message passing layer, conditioned on game phase. This allows
    the model to learn phase-specific patterns:

    - Powerplay (overs 0-5): Aggressive batting, fielding restrictions
    - Middle overs (6-14): Rotation, building innings
    - Death overs (15-19): High risk/reward, pressure situations

    The same underlying graph structure is used, but the model can learn
    to weight information differently based on phase context.
    """

    def __init__(self, config: ModelConfig):
        # Call grandparent init to avoid double initialization
        CricketHeteroGNN.__init__(self, config)

        # Override conv_stack with phase-modulated version
        self.conv_stack = build_phase_modulated_conv_stack(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            phase_dim=config.phase_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Keep the hybrid readout components from parent
        self.matchup_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with phase-conditioned message passing.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Get phase condition for FiLM modulation
        phase_condition = data['phase_state'].x  # [batch_size, phase_dim]

        # 3. Message passing with phase modulation
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_block in self.conv_stack:
            x_dict = conv_block(
                x_dict, edge_index_dict, phase_condition,
                edge_attr_dict if edge_attr_dict else None
            )

        # 4. Matchup interaction (edge-level)
        striker = x_dict['striker_identity']
        bowler = x_dict['bowler_identity']
        matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))

        # 5. Context aggregation (graph-level)
        query = self.query_proj(x_dict['query'])

        # 6. Combine matchup + context
        combined = torch.cat([matchup, query], dim=-1)
        combined = self.combiner(combined)

        # 7. Predict
        logits = self.predictor(combined)

        return logits


class CricketHeteroGNNInningsConditional(CricketHeteroGNNHybrid):
    """
    Innings-conditional variant with separate prediction heads for 1st and 2nd innings.

    The prediction task is fundamentally different between innings:
    - 1st innings: No target, maximize score with wickets in hand
    - 2nd innings: Known target, balance risk vs required rate

    This variant uses:
    1. Shared encoder and message passing (identical context understanding)
    2. First innings head: Standard MLP from combined representation
    3. Second innings head: MLP that also receives chase state features

    The chase state injection allows the model to learn different decision boundaries
    for chasing scenarios (e.g., higher risk tolerance when behind required rate).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # First innings head (standard)
        self.first_innings_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

        # Second innings head (with chase state injection)
        # Chase state has 3 features: runs_required, balls_remaining, required_run_rate
        chase_state_dim = 3
        self.second_innings_head = nn.Sequential(
            nn.Linear(config.hidden_dim + chase_state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with innings-conditional prediction heads.

        Routes to first or second innings head based on is_chase flag.
        Supports batched data with mixed innings (applies correct head per sample).

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Message passing with edge attributes
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict if edge_attr_dict else None)

        # 3. Matchup interaction (edge-level)
        striker = x_dict['striker_identity']
        bowler = x_dict['bowler_identity']
        matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))

        # 4. Context aggregation (graph-level)
        query = self.query_proj(x_dict['query'])

        # 5. Combine matchup + context
        combined = torch.cat([matchup, query], dim=-1)
        combined = self.combiner(combined)

        # 6. Get chase state for second innings routing
        chase_state = data['chase_state'].x  # [batch_size, 3]
        is_chase = data.is_chase  # [batch_size] bool tensor

        # Ensure is_chase is 1D
        if is_chase.dim() > 1:
            is_chase = is_chase.squeeze(-1)

        # 7. Compute both predictions
        first_innings_logits = self.first_innings_head(combined)

        # Second innings: inject chase state
        combined_with_chase = torch.cat([combined, chase_state], dim=-1)
        second_innings_logits = self.second_innings_head(combined_with_chase)

        # 8. Select appropriate logits based on innings
        # Use is_chase as mask to select between heads
        is_chase_expanded = is_chase.unsqueeze(-1).expand_as(first_innings_logits)
        logits = torch.where(is_chase_expanded, second_innings_logits, first_innings_logits)

        return logits

    def forward_with_head_info(self, data) -> tuple:
        """
        Forward pass that also returns which head was used (for debugging/analysis).

        Returns:
            Tuple of (logits, is_chase_mask)
        """
        logits = self.forward(data)
        is_chase = data.is_chase
        if is_chase.dim() > 1:
            is_chase = is_chase.squeeze(-1)
        return logits, is_chase


class CricketHeteroGNNFull(nn.Module):
    """
    Full-featured Cricket GNN combining all architectural enhancements:

    1. Hierarchical player embeddings (cold-start handling)
    2. Hybrid matchup + query readout
    3. Phase-modulated message passing (FiLM)
    4. Innings-conditional prediction heads

    This is the recommended production model that incorporates all
    GDL-informed architectural decisions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # === Node Encoders (with hierarchical player embeddings) ===
        self.encoders = NodeEncoderDict(
            hidden_dim=config.hidden_dim,
            num_venues=config.num_venues,
            num_teams=config.num_teams,
            num_players=config.num_players,
            num_roles=config.num_roles,
            venue_embed_dim=config.venue_embed_dim,
            team_embed_dim=config.team_embed_dim,
            player_embed_dim=config.player_embed_dim,
            role_embed_dim=config.role_embed_dim,
            dropout=config.dropout,
            use_hierarchical_player=config.use_hierarchical_player,
        )

        # === Phase-Modulated Message Passing ===
        if config.use_phase_modulation:
            self.conv_stack = build_phase_modulated_conv_stack(
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                phase_dim=config.phase_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        else:
            self.conv_stack = build_conv_stack(
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )

        # === Hybrid Readout (matchup + query) ===
        self.matchup_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # === Innings-Conditional Prediction Heads ===
        if config.use_innings_conditional:
            # First innings head (standard)
            self.first_innings_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.LayerNorm(config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_classes),
            )

            # Second innings head (with chase state injection)
            chase_state_dim = 3
            self.second_innings_head = nn.Sequential(
                nn.Linear(config.hidden_dim + chase_state_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.LayerNorm(config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_classes),
            )
        else:
            # Single prediction head
            self.predictor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.LayerNorm(config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_classes),
            )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass with all enhancements.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Logits [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = self.encoders.encode_nodes(data)

        # 2. Message passing (with or without phase modulation)
        edge_index_dict = data.edge_index_dict

        # Extract edge attributes if available
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr

        if self.config.use_phase_modulation:
            phase_condition = data['phase_state'].x
            for conv_block in self.conv_stack:
                x_dict = conv_block(
                    x_dict, edge_index_dict, phase_condition,
                    edge_attr_dict if edge_attr_dict else None
                )
        else:
            for conv_block in self.conv_stack:
                x_dict = conv_block(
                    x_dict, edge_index_dict,
                    edge_attr_dict if edge_attr_dict else None
                )

        # 3. Hybrid readout (matchup + query)
        striker = x_dict['striker_identity']
        bowler = x_dict['bowler_identity']
        matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))
        query = self.query_proj(x_dict['query'])
        combined = torch.cat([matchup, query], dim=-1)
        combined = self.combiner(combined)

        # 4. Prediction (innings-conditional or single head)
        if self.config.use_innings_conditional:
            chase_state = data['chase_state'].x
            is_chase = data.is_chase
            if is_chase.dim() > 1:
                is_chase = is_chase.squeeze(-1)

            first_innings_logits = self.first_innings_head(combined)
            combined_with_chase = torch.cat([combined, chase_state], dim=-1)
            second_innings_logits = self.second_innings_head(combined_with_chase)

            is_chase_expanded = is_chase.unsqueeze(-1).expand_as(first_innings_logits)
            logits = torch.where(is_chase_expanded, second_innings_logits, first_innings_logits)
        else:
            logits = self.predictor(combined)

        return logits

    @classmethod
    def from_dataset_metadata(
        cls,
        metadata: dict,
        **kwargs
    ) -> 'CricketHeteroGNNFull':
        """
        Create model from dataset metadata.

        Args:
            metadata: Dict from CricketDataset.get_metadata()
            **kwargs: Override default config values

        Returns:
            Initialized model
        """
        config = ModelConfig(
            num_venues=metadata['num_venues'],
            num_teams=metadata['num_teams'],
            num_players=metadata['num_players'],
            **kwargs
        )
        return cls(config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: CricketHeteroGNN) -> str:
    """Get a summary of model architecture."""
    config = model.config
    num_params = count_parameters(model)

    summary = f"""
CricketHeteroGNN Summary
========================
Hidden dimension: {config.hidden_dim}
Number of layers: {config.num_layers}
Number of heads: {config.num_heads}
Dropout: {config.dropout}

Entity counts:
  Venues: {config.num_venues}
  Teams: {config.num_teams}
  Players: {config.num_players}

Embedding dimensions:
  Venue: {config.venue_embed_dim}
  Team: {config.team_embed_dim}
  Player: {config.player_embed_dim}

Output classes: {config.num_classes}
Total parameters: {num_params:,}
"""
    return summary
