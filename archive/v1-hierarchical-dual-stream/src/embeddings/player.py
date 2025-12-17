"""Player embedding generation with cold-start handling."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PlayerStats:
    """Stats for generating player embedding."""

    # Batting stats
    batting_sr: float = 0.0
    batting_avg: float = 0.0
    batting_position: int = 6  # Default middle order

    # Bowling stats
    bowling_economy: float = 8.0
    bowling_sr: float = 20.0  # balls per wicket
    is_pace: bool = True

    # Role indicators
    is_batsman: bool = True
    is_bowler: bool = False
    matches_played: int = 0


# Role prototypes for cold-start
BATTING_ROLES = {
    "opener_aggressive": PlayerStats(batting_sr=145, batting_avg=28, batting_position=1),
    "opener_anchor": PlayerStats(batting_sr=125, batting_avg=35, batting_position=1),
    "top_order_accumulator": PlayerStats(batting_sr=130, batting_avg=38, batting_position=3),
    "middle_order_aggressor": PlayerStats(batting_sr=150, batting_avg=25, batting_position=5),
    "finisher": PlayerStats(batting_sr=155, batting_avg=22, batting_position=6),
    "wicketkeeper_bat": PlayerStats(batting_sr=135, batting_avg=28, batting_position=4),
    "lower_order": PlayerStats(batting_sr=115, batting_avg=15, batting_position=8),
}

BOWLING_ROLES = {
    "death_pace": PlayerStats(
        bowling_economy=8.5, bowling_sr=18, is_pace=True, is_bowler=True, is_batsman=False
    ),
    "powerplay_pace": PlayerStats(
        bowling_economy=7.5, bowling_sr=20, is_pace=True, is_bowler=True, is_batsman=False
    ),
    "pace_all_rounder": PlayerStats(
        batting_sr=135, batting_avg=20, bowling_economy=8.0, bowling_sr=22,
        is_pace=True, is_bowler=True, is_batsman=True
    ),
    "leg_spinner": PlayerStats(
        bowling_economy=7.0, bowling_sr=19, is_pace=False, is_bowler=True, is_batsman=False
    ),
    "off_spinner": PlayerStats(
        bowling_economy=7.2, bowling_sr=21, is_pace=False, is_bowler=True, is_batsman=False
    ),
    "spin_all_rounder": PlayerStats(
        batting_sr=125, batting_avg=22, bowling_economy=7.5, bowling_sr=24,
        is_pace=False, is_bowler=True, is_batsman=True
    ),
    "part_time_spinner": PlayerStats(
        bowling_economy=8.5, bowling_sr=30, is_pace=False, is_bowler=True, is_batsman=True
    ),
}


class PlayerEmbedding(nn.Module):
    """
    Generate player embeddings from stats.

    Key insight: Model never sees player IDs, only embedding vectors
    generated from features. This enables cold-start handling.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Batsman encoder: stats -> embedding
        self.batsman_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

        # Bowler encoder: stats -> embedding
        self.bowler_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

        # Experience modifier
        self.experience_scale = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.Sigmoid(),
        )

    def _stats_to_batsman_features(self, stats: PlayerStats) -> torch.Tensor:
        """Convert stats to batsman feature vector."""
        return torch.tensor([
            stats.batting_sr / 200.0,
            stats.batting_avg / 50.0,
            stats.batting_position / 11.0,
            float(stats.is_batsman),
            float(stats.is_bowler),  # All-rounder indicator
            min(stats.matches_played, 100) / 100.0,
        ], dtype=torch.float32)

    def _stats_to_bowler_features(self, stats: PlayerStats) -> torch.Tensor:
        """Convert stats to bowler feature vector."""
        return torch.tensor([
            stats.bowling_economy / 12.0,
            stats.bowling_sr / 40.0,
            float(stats.is_pace),
            float(stats.is_bowler),
            min(stats.matches_played, 100) / 100.0,
        ], dtype=torch.float32)

    def forward(
        self,
        stats: PlayerStats,
        as_batsman: bool = True,
    ) -> torch.Tensor:
        """Generate embedding for player from stats."""
        if as_batsman:
            features = self._stats_to_batsman_features(stats)
            base_embed = self.batsman_encoder(features)
        else:
            features = self._stats_to_bowler_features(stats)
            base_embed = self.bowler_encoder(features)

        # Scale by experience
        exp = torch.tensor([min(stats.matches_played, 100) / 100.0])
        exp_scale = self.experience_scale(exp)

        return base_embed * (0.5 + 0.5 * exp_scale)

    def from_role(self, role: str, as_batsman: bool = True) -> torch.Tensor:
        """Generate embedding from role prototype (cold-start)."""
        if as_batsman and role in BATTING_ROLES:
            stats = BATTING_ROLES[role]
        elif not as_batsman and role in BOWLING_ROLES:
            stats = BOWLING_ROLES[role]
        else:
            # Default fallback
            stats = PlayerStats()

        return self.forward(stats, as_batsman)

    def batch_forward(
        self,
        stats_list: list[PlayerStats],
        as_batsman: bool = True,
    ) -> torch.Tensor:
        """Generate embeddings for batch of players."""
        if as_batsman:
            features = torch.stack([
                self._stats_to_batsman_features(s) for s in stats_list
            ])
            return self.batsman_encoder(features)
        else:
            features = torch.stack([
                self._stats_to_bowler_features(s) for s in stats_list
            ])
            return self.bowler_encoder(features)


class PlayerEmbeddingTable(nn.Module):
    """
    Hybrid embedding: learned table + feature-based generation.

    For known players: Use learned embedding (refined during training)
    For unknown players: Generate from stats/role
    """

    def __init__(
        self,
        num_players: int,
        embed_dim: int = 64,
        use_learned: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_learned = use_learned

        # Learned embeddings for known players
        if use_learned:
            self.embedding = nn.Embedding(num_players + 1, embed_dim)  # +1 for unknown
            self.embedding.weight.data.normal_(0, 0.1)

        # Feature-based generator for cold-start
        self.generator = PlayerEmbedding(embed_dim)

        # Fusion layer
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        player_idx: torch.Tensor,
        stats: Optional[list[PlayerStats]] = None,
        as_batsman: bool = True,
    ) -> torch.Tensor:
        """
        Get player embeddings.

        Args:
            player_idx: Indices for known players (0 for unknown)
            stats: Optional stats for feature-based generation
            as_batsman: Whether player is batting
        """
        batch_size = player_idx.shape[0]

        if self.use_learned:
            # Get learned embeddings
            learned = self.embedding(player_idx)

            if stats is not None:
                # Also generate from features
                generated = self.generator.batch_forward(stats, as_batsman)
                # Fuse learned and generated
                combined = torch.cat([learned, generated], dim=-1)
                return self.fusion(combined)
            return learned

        # Pure feature-based (no learned embeddings)
        if stats is not None:
            return self.generator.batch_forward(stats, as_batsman)

        # Fallback to default stats
        default_stats = [PlayerStats() for _ in range(batch_size)]
        return self.generator.batch_forward(default_stats, as_batsman)
