"""Temporal Transformer for cross-ball attention.

Implements architecture from 04-temporal-attention.md:
- Specialized attention heads for different patterns
- Same-bowler and same-batsman attention biases
- Recency bias for recent balls
- Multi-scale temporal encoding
"""

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

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CricketPositionalEncoding(nn.Module):
    """Cricket-specific positional encoding with over structure."""

    def __init__(self, hidden_dim: int, max_overs: int = 20):
        super().__init__()
        self.over_embed = nn.Embedding(max_overs, hidden_dim // 2)
        self.ball_in_over_embed = nn.Embedding(6, hidden_dim // 2)

    def forward(
        self,
        x: torch.Tensor,
        overs: torch.Tensor,
        balls_in_over: torch.Tensor,
    ) -> torch.Tensor:
        over_enc = self.over_embed(overs)
        ball_enc = self.ball_in_over_embed(balls_in_over)
        pos_enc = torch.cat([over_enc, ball_enc], dim=-1)
        return x + pos_enc


class BallEmbedding(nn.Module):
    """Embed individual ball features for temporal sequence."""

    def __init__(
        self,
        num_players: int,
        player_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.player_embed = nn.Embedding(num_players + 1, player_dim)
        self.feature_proj = nn.Linear(3, hidden_dim - player_dim * 2)
        self.output_dim = hidden_dim

    def forward(
        self,
        runs: torch.Tensor,
        wickets: torch.Tensor,
        overs: torch.Tensor,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([runs, wickets, overs], dim=-1)
        feat_embed = self.feature_proj(features)
        batter_embed = self.player_embed(batters)
        bowler_embed = self.player_embed(bowlers)
        return torch.cat([feat_embed, batter_embed, bowler_embed], dim=-1)


class SpecializedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with specialized heads for cricket patterns.

    Head specializations:
    - Head 0: Recency bias (recent balls weighted higher)
    - Head 1: Same-bowler bias (balls from same bowler)
    - Head 2: Same-batsman bias (balls to same batsman)
    - Head 3+: Free to learn any pattern
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Learnable bias strengths for specialized heads
        self.recency_strength = nn.Parameter(torch.tensor(0.5))
        self.same_bowler_strength = nn.Parameter(torch.tensor(2.0))
        self.same_batsman_strength = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        x: torch.Tensor,
        same_bowler_mask: torch.Tensor | None = None,
        same_batsman_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            same_bowler_mask: [batch, seq_len, seq_len] - 1 where bowler matches
            same_batsman_mask: [batch, seq_len, seq_len] - 1 where batsman matches
            causal: Whether to use causal masking

        Returns:
            output: [batch, seq_len, hidden_dim]
            attention: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply specialized biases to specific heads
        if self.num_heads >= 3:
            # Head 0: Recency bias
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float)
            recency_bias = self.recency_strength * (positions / seq_len)
            recency_bias = recency_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq]
            scores[:, 0, :, :] = scores[:, 0, :, :] + recency_bias

            # Head 1: Same-bowler bias
            if same_bowler_mask is not None:
                bowler_bias = self.same_bowler_strength * same_bowler_mask
                scores[:, 1, :, :] = scores[:, 1, :, :] + bowler_bias

            # Head 2: Same-batsman bias
            if same_batsman_mask is not None:
                batsman_bias = self.same_batsman_strength * same_batsman_mask
                scores[:, 2, :, :] = scores[:, 2, :, :] + batsman_bias

        # Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, V)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out, attention


class TemporalTransformerLayer(nn.Module):
    """Single transformer layer with specialized attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = SpecializedMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        same_bowler_mask: torch.Tensor | None = None,
        same_batsman_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attention = self.attention(x, same_bowler_mask, same_batsman_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))

        return x, attention


class TemporalTransformer(nn.Module):
    """
    Transformer for cross-ball temporal attention.

    Features:
    - Specialized attention heads (recency, same-bowler, same-batsman)
    - Cricket-specific positional encoding
    - Attention extraction for interpretability
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

        # Positional encodings
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len + 1)

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Query token for output
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def _build_actor_masks(
        self,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build same-bowler and same-batsman attention masks."""
        batch_size, seq_len = batters.shape

        # Same bowler mask: [batch, seq, seq]
        bowler_expanded = bowlers.unsqueeze(2)  # [batch, seq, 1]
        same_bowler = (bowler_expanded == bowlers.unsqueeze(1)).float()

        # Same batsman mask
        batter_expanded = batters.unsqueeze(2)
        same_batsman = (batter_expanded == batters.unsqueeze(1)).float()

        return same_bowler, same_batsman

    def forward(
        self,
        runs: torch.Tensor,
        wickets: torch.Tensor,
        overs: torch.Tensor,
        batters: torch.Tensor,
        bowlers: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            runs: [batch, seq_len]
            wickets: [batch, seq_len]
            overs: [batch, seq_len]
            batters: [batch, seq_len]
            bowlers: [batch, seq_len]

        Returns:
            output: [batch, hidden_dim]
            attention_dict: Attention weights per layer/head if requested
        """
        batch_size = runs.shape[0]

        # Embed ball sequence
        x = self.ball_embed(runs, wickets, overs, batters, bowlers)

        # Append query token
        query = self.query_token.expand(batch_size, -1, -1)
        x = torch.cat([x, query], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Build actor masks (extend for query token)
        same_bowler, same_batsman = self._build_actor_masks(batters, bowlers)

        # Pad masks for query token (query can attend to all)
        pad = torch.ones(batch_size, 1, same_bowler.shape[2], device=x.device)
        same_bowler = torch.cat([same_bowler, pad], dim=1)
        same_batsman = torch.cat([same_batsman, pad], dim=1)
        pad_col = torch.ones(batch_size, same_bowler.shape[1], 1, device=x.device)
        same_bowler = torch.cat([same_bowler, pad_col], dim=2)
        same_batsman = torch.cat([same_batsman, pad_col], dim=2)

        # Process through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, same_bowler, same_batsman)
            attention_weights.append(attn)

        # Extract query token output
        output = x[:, -1]
        output = self.output_proj(output)

        if return_attention:
            attention_dict = self._format_attention(
                attention_weights, batters, bowlers
            )
            return output, attention_dict

        return output, None

    def _format_attention(
        self,
        attention_weights: list[torch.Tensor],
        batters: torch.Tensor,
        bowlers: torch.Tensor,
    ) -> dict:
        """Format attention weights for interpretability."""
        # Use last layer attention, query token row
        final_attn = attention_weights[-1][:, :, -1, :-1]  # [batch, heads, seq]

        # Average over batch
        attn_mean = final_attn.mean(dim=0)  # [heads, seq]

        return {
            "head_attention": {
                "recency": attn_mean[0].tolist() if attn_mean.shape[0] > 0 else [],
                "same_bowler": attn_mean[1].tolist() if attn_mean.shape[0] > 1 else [],
                "same_batsman": attn_mean[2].tolist() if attn_mean.shape[0] > 2 else [],
                "learned": attn_mean[3].tolist() if attn_mean.shape[0] > 3 else [],
            },
            "aggregate": final_attn.mean(dim=1).mean(dim=0).tolist(),
            "layer_attentions": [a.mean(dim=0).tolist() for a in attention_weights],
        }
