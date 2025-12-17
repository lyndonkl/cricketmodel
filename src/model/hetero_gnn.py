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
from .conv_builder import build_conv_stack


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

    # Regularization
    dropout: float = 0.1

    # Task
    num_classes: int = 7  # Dot, Single, Two, Three, Four, Six, Wicket

    # Model variant
    use_hybrid_readout: bool = True  # Use matchup + query hybrid readout


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
            venue_embed_dim=config.venue_embed_dim,
            team_embed_dim=config.team_embed_dim,
            player_embed_dim=config.player_embed_dim,
            dropout=config.dropout,
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
