"""
Node Encoders for Cricket Heterogeneous GNN

Each node type has its own encoder that projects raw features
to the common hidden dimension.

Includes HierarchicalPlayerEncoder for cold-start handling via team/role fallbacks.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..data.entity_mapper import NUM_ROLES


class EntityEncoder(nn.Module):
    """
    Encodes entity IDs (venues, teams, players) to hidden representations.

    Uses a learned embedding table followed by a projection layer.
    ID 0 is reserved for unknown entities.
    """

    def __init__(
        self,
        num_entities: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_entities: Number of unique entities (excluding unknown)
            embed_dim: Dimension of entity embeddings
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        # +1 for unknown entity (ID 0)
        self.embedding = nn.Embedding(num_entities + 1, embed_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Entity IDs [batch, 1] (long tensor)

        Returns:
            Hidden representations [batch, hidden_dim]
        """
        # x shape: [num_nodes, 1]
        emb = self.embedding(x.squeeze(-1))  # [num_nodes, embed_dim]
        return self.projection(emb)  # [num_nodes, hidden_dim]


class HierarchicalPlayerEncoder(nn.Module):
    """
    Encodes player IDs with hierarchical fallback for cold-start handling.

    When a player is known (has learned embedding), uses that embedding.
    When a player is unknown (ID=0), falls back to team + role embeddings.

    This handles the cold-start problem where:
    - Unknown international stars get team+role context (not zero vector)
    - Unknown debutants get role-based prior (not identical to stars)

    Hierarchy:
    - Level 1: Player embedding (specific individual)
    - Level 2: Team embedding (team-level characteristics)
    - Level 3: Role embedding (role-level prior: opener, finisher, bowler, etc.)
    """

    def __init__(
        self,
        num_players: int,
        num_teams: int,
        num_roles: int,
        player_embed_dim: int,
        team_embed_dim: int,
        role_embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_players: Number of unique players
            num_teams: Number of unique teams
            num_roles: Number of role categories (8: unknown + 7 roles)
            player_embed_dim: Dimension of player embeddings
            team_embed_dim: Dimension of team embeddings (for fallback)
            role_embed_dim: Dimension of role embeddings (for fallback)
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_players = num_players
        self.hidden_dim = hidden_dim

        # Primary: Player embedding (ID 0 = unknown, will use fallback)
        self.player_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)

        # Fallback embeddings for unknown players
        self.team_embed = nn.Embedding(num_teams + 1, team_embed_dim, padding_idx=0)
        self.role_embed = nn.Embedding(num_roles + 1, role_embed_dim, padding_idx=0)

        # Projection for known players (player embedding -> hidden)
        self.player_projection = nn.Sequential(
            nn.Linear(player_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Projection for unknown players (team + role -> hidden)
        fallback_input_dim = team_embed_dim + role_embed_dim
        self.fallback_projection = nn.Sequential(
            nn.Linear(fallback_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.player_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_embed.weight, mean=0.0, std=0.02)

        # Zero out padding embeddings
        with torch.no_grad():
            self.player_embed.weight[0].fill_(0)
            self.team_embed.weight[0].fill_(0)
            self.role_embed.weight[0].fill_(0)

    def forward(
        self,
        player_ids: torch.Tensor,
        team_ids: torch.Tensor,
        role_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode players with hierarchical fallback.

        For known players (player_id > 0): Use player embedding
        For unknown players (player_id = 0): Use team + role embeddings

        Args:
            player_ids: Player IDs [num_nodes, 1]
            team_ids: Team IDs for each player [num_nodes, 1]
            role_ids: Role IDs for each player [num_nodes, 1]

        Returns:
            Hidden representations [num_nodes, hidden_dim]
        """
        player_ids = player_ids.squeeze(-1)  # [num_nodes]
        team_ids = team_ids.squeeze(-1)
        role_ids = role_ids.squeeze(-1)

        # Get embeddings
        player_emb = self.player_embed(player_ids)  # [num_nodes, player_embed_dim]
        team_emb = self.team_embed(team_ids)        # [num_nodes, team_embed_dim]
        role_emb = self.role_embed(role_ids)        # [num_nodes, role_embed_dim]

        # Project known players
        known_output = self.player_projection(player_emb)  # [num_nodes, hidden_dim]

        # Project fallback (team + role)
        fallback_input = torch.cat([team_emb, role_emb], dim=-1)
        fallback_output = self.fallback_projection(fallback_input)  # [num_nodes, hidden_dim]

        # Select based on whether player is known
        # Unknown players have player_id = 0
        is_unknown = (player_ids == 0).unsqueeze(-1)  # [num_nodes, 1]

        # Use known embedding for known players, fallback for unknown
        output = torch.where(is_unknown, fallback_output, known_output)

        return output


class FeatureEncoder(nn.Module):
    """
    Encodes numeric features to hidden representations.

    Uses a small MLP with residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]

        Returns:
            Hidden representations [batch, hidden_dim]
        """
        return self.encoder(x)


class BallEncoder(nn.Module):
    """
    Encodes ball nodes with both numeric features and player embeddings.

    Ball nodes contain:
    - 17 numeric features:
        - Basic: runs, is_wicket, over, ball_in_over, is_boundary
        - Extras: is_wide, is_noball, is_bye, is_legbye
        - Wicket types: bowled, caught, lbw, run_out, stumped, other
        - Run-out attribution: striker_run_out, nonstriker_run_out
    - Bowler ID (to be embedded)
    - Batsman ID (to be embedded)
    """

    def __init__(
        self,
        num_players: int,
        player_embed_dim: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_players: Number of unique players
            player_embed_dim: Dimension of player embeddings
            feature_dim: Dimension of numeric features (9)
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Player embeddings
        self.bowler_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)
        self.batsman_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)

        # Projection
        concat_dim = 2 * player_embed_dim + feature_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.bowler_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.batsman_embed.weight, mean=0.0, std=0.02)
        # Zero out padding embeddings
        with torch.no_grad():
            self.bowler_embed.weight[0].fill_(0)
            self.batsman_embed.weight[0].fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        bowler_ids: torch.Tensor,
        batsman_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Numeric features [num_balls, feature_dim]
            bowler_ids: Bowler IDs [num_balls]
            batsman_ids: Batsman IDs [num_balls]

        Returns:
            Hidden representations [num_balls, hidden_dim]
        """
        bowler_emb = self.bowler_embed(bowler_ids)   # [num_balls, player_embed_dim]
        batsman_emb = self.batsman_embed(batsman_ids)  # [num_balls, player_embed_dim]

        combined = torch.cat([x, bowler_emb, batsman_emb], dim=-1)
        return self.projection(combined)


class QueryEncoder(nn.Module):
    """
    Provides learned query embedding.

    The query node uses a learned parameter as its initial representation,
    which is then refined through message passing.
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Learned query embedding
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: Number of samples in batch

        Returns:
            Query embeddings [batch_size, hidden_dim]
        """
        return self.embedding.expand(batch_size, -1)


class NodeEncoderDict(nn.Module):
    """
    Container for all node encoders organized by type.

    Provides a unified interface for encoding all node types
    in a heterogeneous graph.

    Uses HierarchicalPlayerEncoder for player nodes to handle cold-start
    via team/role fallback embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_venues: int,
        num_teams: int,
        num_players: int,
        num_roles: int = NUM_ROLES,
        venue_embed_dim: int = 32,
        team_embed_dim: int = 32,
        player_embed_dim: int = 64,
        role_embed_dim: int = 16,
        dropout: float = 0.1,
        use_hierarchical_player: bool = True,
    ):
        """
        Args:
            hidden_dim: Common hidden dimension for all encoders
            num_venues: Number of unique venues
            num_teams: Number of unique teams
            num_players: Number of unique players
            num_roles: Number of role categories (default 8)
            venue_embed_dim: Embedding dimension for venues
            team_embed_dim: Embedding dimension for teams
            player_embed_dim: Embedding dimension for players
            role_embed_dim: Embedding dimension for roles
            dropout: Dropout rate
            use_hierarchical_player: Use hierarchical player encoder with fallback
        """
        super().__init__()

        self.use_hierarchical_player = use_hierarchical_player

        # Entity encoders
        self.venue_encoder = EntityEncoder(num_venues, venue_embed_dim, hidden_dim, dropout)
        self.team_encoder = EntityEncoder(num_teams, team_embed_dim, hidden_dim, dropout)

        # Player encoder: hierarchical (with fallback) or simple
        if use_hierarchical_player:
            self.player_encoder = HierarchicalPlayerEncoder(
                num_players=num_players,
                num_teams=num_teams,
                num_roles=num_roles,
                player_embed_dim=player_embed_dim,
                team_embed_dim=team_embed_dim // 2,  # Smaller for fallback
                role_embed_dim=role_embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.player_encoder = EntityEncoder(num_players, player_embed_dim, hidden_dim, dropout)

        # Feature encoders (one per feature node type)
        # Feature dimensions:
        # - phase_state: 6 (is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over)
        # - striker_state: 8 (runs, balls, sr, dots_pct, is_set, boundaries, is_debut_ball, balls_since_on_strike)
        # - nonstriker_state: 8 (Z2 symmetric with striker: runs, balls, sr, dots_pct, is_set, boundaries, is_debut_ball, balls_since_as_nonstriker)
        self.feature_encoders = nn.ModuleDict({
            'score_state': FeatureEncoder(5, hidden_dim, dropout),  # +1 for is_womens_cricket
            'chase_state': FeatureEncoder(7, hidden_dim, dropout),  # Enhanced: runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem
            'phase_state': FeatureEncoder(6, hidden_dim, dropout),  # +1 for is_first_ball, +1 for is_super_over
            'time_pressure': FeatureEncoder(3, hidden_dim, dropout),
            'wicket_buffer': FeatureEncoder(2, hidden_dim, dropout),
            'striker_state': FeatureEncoder(8, hidden_dim, dropout),  # +1 is_debut_ball, +1 balls_since_on_strike
            'nonstriker_state': FeatureEncoder(8, hidden_dim, dropout),  # Z2 symmetric: +1 balls_since_as_nonstriker
            'bowler_state': FeatureEncoder(8, hidden_dim, dropout),  # +2 for bowling type (P1.2)
            'partnership': FeatureEncoder(4, hidden_dim, dropout),
            'batting_momentum': FeatureEncoder(1, hidden_dim, dropout),
            'bowling_momentum': FeatureEncoder(1, hidden_dim, dropout),
            'pressure_index': FeatureEncoder(1, hidden_dim, dropout),
            'dot_pressure': FeatureEncoder(5, hidden_dim, dropout),  # +2 for pressure_accumulated, pressure_trend
        })

        # Ball encoder (18 features: runs, is_wicket, over, ball_in_over, is_boundary,
        # is_wide, is_noball, is_bye, is_legbye,
        # wicket_bowled, wicket_caught, wicket_lbw, wicket_run_out, wicket_stumped, wicket_other,
        # striker_run_out, nonstriker_run_out, bowling_end)
        self.ball_encoder = BallEncoder(
            num_players, player_embed_dim, 18, hidden_dim, dropout
        )

        # Query encoder
        self.query_encoder = QueryEncoder(hidden_dim)

    def encode_nodes(self, data) -> dict:
        """
        Encode all node features in a HeteroData batch.

        Args:
            data: HeteroData or batched HeteroData

        Returns:
            Dict mapping node type to encoded features
        """
        x_dict = {}

        # Entity nodes
        x_dict['venue'] = self.venue_encoder(data['venue'].x)
        x_dict['batting_team'] = self.team_encoder(data['batting_team'].x)
        x_dict['bowling_team'] = self.team_encoder(data['bowling_team'].x)

        # Player nodes: use hierarchical encoder if available
        if self.use_hierarchical_player:
            # Hierarchical encoder needs player_id, team_id, role_id
            x_dict['striker_identity'] = self.player_encoder(
                data['striker_identity'].x,
                data['striker_identity'].team_id,
                data['striker_identity'].role_id,
            )
            x_dict['nonstriker_identity'] = self.player_encoder(
                data['nonstriker_identity'].x,
                data['nonstriker_identity'].team_id,
                data['nonstriker_identity'].role_id,
            )
            x_dict['bowler_identity'] = self.player_encoder(
                data['bowler_identity'].x,
                data['bowler_identity'].team_id,
                data['bowler_identity'].role_id,
            )
        else:
            # Simple encoder: just player_id
            x_dict['striker_identity'] = self.player_encoder(data['striker_identity'].x)
            x_dict['nonstriker_identity'] = self.player_encoder(data['nonstriker_identity'].x)
            x_dict['bowler_identity'] = self.player_encoder(data['bowler_identity'].x)

        # Feature nodes
        for node_type, encoder in self.feature_encoders.items():
            x_dict[node_type] = encoder(data[node_type].x)

        # Ball nodes (handle empty case)
        if 'ball' in data.node_types and data['ball'].num_nodes > 0:
            x_dict['ball'] = self.ball_encoder(
                data['ball'].x,
                data['ball'].bowler_ids,
                data['ball'].batsman_ids
            )
        else:
            # Create empty tensor with correct dimensions
            device = data['venue'].x.device
            hidden_dim = x_dict['venue'].shape[-1]
            x_dict['ball'] = torch.zeros((0, hidden_dim), device=device)

        # Query node
        # Count number of query nodes (1 per sample in batch)
        num_queries = data['query'].num_nodes
        x_dict['query'] = self.query_encoder(num_queries)

        return x_dict
