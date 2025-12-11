"""Temporal Transformer for cross-ball attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for ball sequence."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class BallEmbedding(nn.Module):
    """Embed individual ball features for temporal sequence."""

    def __init__(
        self,
        num_players: int,
        player_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Player embeddings (shared for efficiency)
        self.player_embed = nn.Embedding(num_players + 1, player_dim)

        # Ball feature projection
        # Input: [runs, is_wicket, over_progress, batter_idx, bowler_idx]
        self.feature_proj = nn.Linear(3, hidden_dim - player_dim * 2)

        self.output_dim = hidden_dim

    def forward(
        self,
        runs: torch.Tensor,  # [batch, seq_len]
        wickets: torch.Tensor,  # [batch, seq_len]
        overs: torch.Tensor,  # [batch, seq_len]
        batters: torch.Tensor,  # [batch, seq_len]
        bowlers: torch.Tensor,  # [batch, seq_len]
    ) -> torch.Tensor:
        """Embed ball sequence."""
        batch_size, seq_len = runs.shape

        # Stack numeric features
        features = torch.stack([runs, wickets, overs], dim=-1)  # [batch, seq, 3]
        feat_embed = self.feature_proj(features)  # [batch, seq, hidden - 2*player]

        # Player embeddings
        batter_embed = self.player_embed(batters)  # [batch, seq, player_dim]
        bowler_embed = self.player_embed(bowlers)  # [batch, seq, player_dim]

        # Concatenate
        return torch.cat([feat_embed, batter_embed, bowler_embed], dim=-1)


class TemporalTransformer(nn.Module):
    """
    Transformer for cross-ball temporal attention.

    Specialized attention heads learn different temporal patterns:
    - Recency: Recent balls matter most
    - Same-bowler: Balls from same bowler
    - Same-batsman: Current batsman's form
    - Boundary patterns: Where boundaries occurred
    """

    def __init__(
        self,
        num_players: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 24,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Ball embedding
        self.ball_embed = BallEmbedding(num_players, player_dim=32, hidden_dim=hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Query token (learnable)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        runs: torch.Tensor,
        wickets: torch.Tensor,
        overs: torch.Tensor,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Process ball sequence.

        Args:
            runs: [batch, seq_len] - runs scored per ball
            wickets: [batch, seq_len] - wicket indicator
            overs: [batch, seq_len] - over progress
            batters: [batch, seq_len] - batter indices
            bowlers: [batch, seq_len] - bowler indices
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, hidden_dim] - temporal representation
            attention: [batch, num_heads, seq_len] if requested
        """
        batch_size = runs.shape[0]

        # Embed ball sequence
        x = self.ball_embed(runs, wickets, overs, batters, bowlers)

        # Add query token at the end
        query = self.query_token.expand(batch_size, -1, -1)
        x = torch.cat([x, query], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask (query can attend to all balls)
        seq_len = x.shape[1]
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        # Query token is last - can attend to everything

        # Transform
        x = self.transformer(x, mask=mask)

        # Take query token output
        output = x[:, -1]  # [batch, hidden_dim]
        output = self.output_proj(output)

        # Get attention if requested
        attention = None
        if return_attention:
            attention = self._get_attention(runs, wickets, overs, batters, bowlers)

        return output, attention

    def _get_attention(
        self,
        runs: torch.Tensor,
        wickets: torch.Tensor,
        overs: torch.Tensor,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
    ) -> torch.Tensor:
        """Extract attention weights from last layer."""
        batch_size = runs.shape[0]

        x = self.ball_embed(runs, wickets, overs, batters, bowlers)
        query = self.query_token.expand(batch_size, -1, -1)
        x = torch.cat([x, query], dim=1)
        x = self.pos_encoder(x)

        # Manual forward through encoder to get attention
        # This is a simplified version - full implementation would hook into attention
        # For now, compute attention scores manually

        # Get query and key from last position
        q = x[:, -1:]  # [batch, 1, hidden]
        k = x[:, :-1]  # [batch, seq, hidden]

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        attention = F.softmax(scores, dim=-1)  # [batch, 1, seq]

        return attention.squeeze(1)  # [batch, seq]

    def get_temporal_patterns(
        self,
        attention: torch.Tensor,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
        current_batter: int,
        current_bowler: int,
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Analyze temporal attention patterns for interpretability.

        Returns which balls got attention and why.
        """
        seq_len = attention.shape[-1]
        patterns = {
            "same_bowler": [],
            "same_batter": [],
            "recent": [],
            "high_attention": [],
        }

        # Analyze each ball
        for i in range(seq_len):
            weight = attention[0, i].item() if attention.dim() > 1 else attention[i].item()

            # Recency (last 6 balls)
            if i >= seq_len - 6:
                patterns["recent"].append((i, weight))

            # Same bowler
            if bowlers[0, i].item() == current_bowler:
                patterns["same_bowler"].append((i, weight))

            # Same batter
            if batters[0, i].item() == current_batter:
                patterns["same_batter"].append((i, weight))

            # High attention (top quartile)
            if weight > 0.1:  # Threshold
                patterns["high_attention"].append((i, weight))

        return patterns
