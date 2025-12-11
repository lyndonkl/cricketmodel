"""Unified embedding manager for all entity types."""

import torch
import torch.nn as nn

from .player import PlayerEmbeddingTable, PlayerStats
from .venue import VenueEmbeddingTable, VenueStats


class TeamEmbedding(nn.Module):
    """Simple team embedding."""

    def __init__(self, num_teams: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_teams + 1, embed_dim)
        self.embedding.weight.data.normal_(0, 0.1)

    def forward(self, team_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(team_idx)


class EmbeddingManager(nn.Module):
    """
    Central manager for all embeddings.

    Provides consistent interface for:
    - Players (batters, bowlers)
    - Venues
    - Teams
    """

    def __init__(
        self,
        num_players: int,
        num_venues: int,
        num_teams: int,
        player_dim: int = 64,
        venue_dim: int = 32,
        team_dim: int = 32,
    ):
        super().__init__()

        self.player_dim = player_dim
        self.venue_dim = venue_dim
        self.team_dim = team_dim

        self.players = PlayerEmbeddingTable(num_players, player_dim)
        self.venues = VenueEmbeddingTable(num_venues, venue_dim)
        self.teams = TeamEmbedding(num_teams, team_dim)

    def get_batsman_embedding(
        self,
        player_idx: torch.Tensor,
        stats: list[PlayerStats] | None = None,
    ) -> torch.Tensor:
        """Get batsman embeddings."""
        return self.players(player_idx, stats, as_batsman=True)

    def get_bowler_embedding(
        self,
        player_idx: torch.Tensor,
        stats: list[PlayerStats] | None = None,
    ) -> torch.Tensor:
        """Get bowler embeddings."""
        return self.players(player_idx, stats, as_batsman=False)

    def get_venue_embedding(
        self,
        venue_idx: torch.Tensor,
        stats: list[VenueStats] | None = None,
    ) -> torch.Tensor:
        """Get venue embeddings."""
        return self.venues(venue_idx, stats)

    def get_team_embedding(self, team_idx: torch.Tensor) -> torch.Tensor:
        """Get team embeddings."""
        return self.teams(team_idx)

    def forward(
        self,
        batter_idx: torch.Tensor,
        bowler_idx: torch.Tensor,
        non_striker_idx: torch.Tensor,
        venue_idx: torch.Tensor,
        batting_team_idx: torch.Tensor,
        bowling_team_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get all embeddings for a batch."""
        return {
            "batter": self.get_batsman_embedding(batter_idx),
            "bowler": self.get_bowler_embedding(bowler_idx),
            "non_striker": self.get_batsman_embedding(non_striker_idx),
            "venue": self.get_venue_embedding(venue_idx),
            "batting_team": self.get_team_embedding(batting_team_idx),
            "bowling_team": self.get_team_embedding(bowling_team_idx),
        }
