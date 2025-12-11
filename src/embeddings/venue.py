"""Venue embedding generation with cold-start handling."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VenueStats:
    """Stats for generating venue embedding."""

    avg_first_innings_score: float = 165.0
    avg_second_innings_score: float = 155.0
    boundary_percentage: float = 0.15
    avg_wickets_per_innings: float = 6.0
    pace_wicket_share: float = 0.5  # Pace vs spin friendly


# Country-based venue prototypes
COUNTRY_PROTOTYPES = {
    "India": VenueStats(
        avg_first_innings_score=165, avg_second_innings_score=155,
        boundary_percentage=0.14, avg_wickets_per_innings=6.5, pace_wicket_share=0.35
    ),
    "Australia": VenueStats(
        avg_first_innings_score=170, avg_second_innings_score=160,
        boundary_percentage=0.16, avg_wickets_per_innings=6.0, pace_wicket_share=0.65
    ),
    "England": VenueStats(
        avg_first_innings_score=175, avg_second_innings_score=165,
        boundary_percentage=0.17, avg_wickets_per_innings=5.5, pace_wicket_share=0.60
    ),
    "South Africa": VenueStats(
        avg_first_innings_score=165, avg_second_innings_score=155,
        boundary_percentage=0.15, avg_wickets_per_innings=6.0, pace_wicket_share=0.55
    ),
    "New Zealand": VenueStats(
        avg_first_innings_score=160, avg_second_innings_score=150,
        boundary_percentage=0.14, avg_wickets_per_innings=6.5, pace_wicket_share=0.60
    ),
    "West Indies": VenueStats(
        avg_first_innings_score=165, avg_second_innings_score=155,
        boundary_percentage=0.16, avg_wickets_per_innings=6.0, pace_wicket_share=0.50
    ),
    "Pakistan": VenueStats(
        avg_first_innings_score=160, avg_second_innings_score=150,
        boundary_percentage=0.13, avg_wickets_per_innings=6.5, pace_wicket_share=0.40
    ),
    "Sri Lanka": VenueStats(
        avg_first_innings_score=160, avg_second_innings_score=150,
        boundary_percentage=0.14, avg_wickets_per_innings=6.5, pace_wicket_share=0.35
    ),
    "Bangladesh": VenueStats(
        avg_first_innings_score=155, avg_second_innings_score=145,
        boundary_percentage=0.13, avg_wickets_per_innings=7.0, pace_wicket_share=0.30
    ),
    "UAE": VenueStats(
        avg_first_innings_score=155, avg_second_innings_score=145,
        boundary_percentage=0.13, avg_wickets_per_innings=6.5, pace_wicket_share=0.40
    ),
}

# Venue type prototypes
VENUE_TYPES = {
    "high_scoring_flat": VenueStats(
        avg_first_innings_score=180, avg_second_innings_score=170,
        boundary_percentage=0.18, avg_wickets_per_innings=5.0, pace_wicket_share=0.50
    ),
    "spin_friendly": VenueStats(
        avg_first_innings_score=150, avg_second_innings_score=140,
        boundary_percentage=0.12, avg_wickets_per_innings=7.0, pace_wicket_share=0.30
    ),
    "pace_friendly": VenueStats(
        avg_first_innings_score=160, avg_second_innings_score=150,
        boundary_percentage=0.15, avg_wickets_per_innings=6.5, pace_wicket_share=0.70
    ),
    "balanced": VenueStats(),  # Default
}


class VenueEmbedding(nn.Module):
    """Generate venue embeddings from stats or prototypes."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(5, 24),
            nn.ReLU(),
            nn.Linear(24, embed_dim),
        )

    def _stats_to_features(self, stats: VenueStats) -> torch.Tensor:
        return torch.tensor([
            stats.avg_first_innings_score / 200.0,
            stats.avg_second_innings_score / 200.0,
            stats.boundary_percentage,
            stats.avg_wickets_per_innings / 10.0,
            stats.pace_wicket_share,
        ], dtype=torch.float32)

    def forward(self, stats: VenueStats) -> torch.Tensor:
        features = self._stats_to_features(stats)
        return self.encoder(features)

    def from_country(self, country: str) -> torch.Tensor:
        """Generate embedding from country prototype."""
        stats = COUNTRY_PROTOTYPES.get(country, VenueStats())
        return self.forward(stats)

    def from_type(self, venue_type: str) -> torch.Tensor:
        """Generate embedding from venue type."""
        stats = VENUE_TYPES.get(venue_type, VenueStats())
        return self.forward(stats)

    def batch_forward(self, stats_list: list[VenueStats]) -> torch.Tensor:
        features = torch.stack([self._stats_to_features(s) for s in stats_list])
        return self.encoder(features)


class VenueEmbeddingTable(nn.Module):
    """Hybrid venue embedding with learned + generated."""

    def __init__(self, num_venues: int, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(num_venues + 1, embed_dim)
        self.embedding.weight.data.normal_(0, 0.1)

        self.generator = VenueEmbedding(embed_dim)

    def forward(
        self,
        venue_idx: torch.Tensor,
        stats: list[VenueStats] | None = None,
    ) -> torch.Tensor:
        learned = self.embedding(venue_idx)

        if stats is not None:
            generated = self.generator.batch_forward(stats)
            return (learned + generated) / 2

        return learned
