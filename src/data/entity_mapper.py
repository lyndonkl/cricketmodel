"""
Entity Mapper for Cricket Ball Prediction

Maps entity names (venues, teams, players) to integer IDs for embedding layers.
Supports hierarchical player embeddings with team and role fallbacks.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple


# Player role categories for hierarchical embeddings
# These provide fallback information when a player is unknown
PLAYER_ROLES = [
    'unknown',      # 0 - fallback for completely unknown
    'opener',       # 1 - opening batsman (aggressive, faces new ball)
    'top_order',    # 2 - positions 3-4 (anchors innings)
    'middle_order', # 3 - positions 5-6 (builds or accelerates)
    'finisher',     # 4 - positions 6-7 (explosive, death overs)
    'bowler',       # 5 - primarily a bowler (tailender batting)
    'allrounder',   # 6 - genuine all-rounder (contributes both)
    'keeper',       # 7 - wicket-keeper (often finisher role)
]

ROLE_TO_ID = {role: idx for idx, role in enumerate(PLAYER_ROLES)}
NUM_ROLES = len(PLAYER_ROLES)


class EntityMapper:
    """
    Bidirectional mapping between entity names and integer IDs.

    Maintains separate mappings for:
    - Venues (cricket grounds)
    - Teams (national/franchise teams)
    - Players (batters/bowlers)

    ID 0 is reserved for unknown/unseen entities.
    """

    def __init__(self):
        """Initialize empty mappings."""
        self.venue_to_id: Dict[str, int] = {}
        self.team_to_id: Dict[str, int] = {}
        self.player_to_id: Dict[str, int] = {}

        self.id_to_venue: Dict[int, str] = {}
        self.id_to_team: Dict[int, str] = {}
        self.id_to_player: Dict[int, str] = {}

        # Hierarchical player information for cold-start handling
        # Maps player name to their primary team (most frequent)
        self.player_to_team: Dict[str, str] = {}
        # Maps player name to their inferred role
        self.player_to_role: Dict[str, str] = {}
        # Track player statistics for role inference
        self._player_stats: Dict[str, Dict] = {}

        # Track next available ID (0 is reserved for unknown)
        self._next_venue_id = 1
        self._next_team_id = 1
        self._next_player_id = 1

    def _get_or_create_id(
        self,
        name: str,
        name_to_id: Dict[str, int],
        id_to_name: Dict[int, str],
        next_id_attr: str
    ) -> int:
        """Get existing ID or create new one for an entity."""
        if name in name_to_id:
            return name_to_id[name]

        # Create new ID
        new_id = getattr(self, next_id_attr)
        name_to_id[name] = new_id
        id_to_name[new_id] = name
        setattr(self, next_id_attr, new_id + 1)
        return new_id

    def get_venue_id(self, venue: str, create: bool = False) -> int:
        """
        Get ID for a venue.

        Args:
            venue: Venue name (e.g., "Melbourne Cricket Ground")
            create: If True, create new ID for unseen venues

        Returns:
            Integer ID (0 if unknown and create=False)
        """
        if create:
            return self._get_or_create_id(
                venue, self.venue_to_id, self.id_to_venue, "_next_venue_id"
            )
        return self.venue_to_id.get(venue, 0)

    def get_team_id(self, team: str, create: bool = False) -> int:
        """
        Get ID for a team.

        Args:
            team: Team name (e.g., "Australia", "Mumbai Indians")
            create: If True, create new ID for unseen teams

        Returns:
            Integer ID (0 if unknown and create=False)
        """
        if create:
            return self._get_or_create_id(
                team, self.team_to_id, self.id_to_team, "_next_team_id"
            )
        return self.team_to_id.get(team, 0)

    def get_player_id(self, player: str, create: bool = False) -> int:
        """
        Get ID for a player.

        Args:
            player: Player name (e.g., "V Kohli", "JJ Bumrah")
            create: If True, create new ID for unseen players

        Returns:
            Integer ID (0 if unknown and create=False)
        """
        if create:
            return self._get_or_create_id(
                player, self.player_to_id, self.id_to_player, "_next_player_id"
            )
        return self.player_to_id.get(player, 0)

    def get_player_team_id(self, player: str) -> int:
        """
        Get team ID for a player (for hierarchical embedding fallback).

        Args:
            player: Player name

        Returns:
            Team ID (0 if player or team unknown)
        """
        team = self.player_to_team.get(player)
        if team:
            return self.team_to_id.get(team, 0)
        return 0

    def get_player_role_id(self, player: str) -> int:
        """
        Get role ID for a player (for hierarchical embedding fallback).

        Args:
            player: Player name

        Returns:
            Role ID (0 if role unknown)
        """
        role = self.player_to_role.get(player, 'unknown')
        return ROLE_TO_ID.get(role, 0)

    def get_player_hierarchy(self, player: str) -> Tuple[int, int, int]:
        """
        Get full hierarchical IDs for a player.

        Returns (player_id, team_id, role_id) for use with HierarchicalPlayerEncoder.

        Args:
            player: Player name

        Returns:
            Tuple of (player_id, team_id, role_id)
        """
        return (
            self.get_player_id(player),
            self.get_player_team_id(player),
            self.get_player_role_id(player)
        )

    def get_venue_name(self, venue_id: int) -> Optional[str]:
        """Get venue name from ID."""
        return self.id_to_venue.get(venue_id)

    def get_team_name(self, team_id: int) -> Optional[str]:
        """Get team name from ID."""
        return self.id_to_team.get(team_id)

    def get_player_name(self, player_id: int) -> Optional[str]:
        """Get player name from ID."""
        return self.id_to_player.get(player_id)

    @property
    def num_venues(self) -> int:
        """Number of unique venues (excluding unknown)."""
        return len(self.venue_to_id)

    @property
    def num_teams(self) -> int:
        """Number of unique teams (excluding unknown)."""
        return len(self.team_to_id)

    @property
    def num_players(self) -> int:
        """Number of unique players (excluding unknown)."""
        return len(self.player_to_id)

    def build_from_matches(self, match_files: list) -> None:
        """
        Build entity mappings from a list of match JSON files.

        Should be called on ALL data before splitting to ensure consistent IDs.
        Also builds player-team relationships and infers player roles.

        Args:
            match_files: List of paths to match JSON files
        """
        # Track player-team associations (player can play for multiple teams)
        player_team_counts: Dict[str, Dict[str, int]] = {}

        for match_file in match_files:
            with open(match_file, 'r') as f:
                match_data = json.load(f)

            info = match_data.get('info', {})

            # Add venue
            venue = info.get('venue')
            if venue:
                self.get_venue_id(venue, create=True)

            # Add teams
            teams = info.get('teams', [])
            for team in teams:
                self.get_team_id(team, create=True)

            # Add players from all innings and track team associations
            for innings in match_data.get('innings', []):
                batting_team = innings.get('team', '')

                for over_data in innings.get('overs', []):
                    over_num = over_data.get('over', 0)

                    for ball_idx, delivery in enumerate(over_data.get('deliveries', [])):
                        batter = delivery.get('batter')
                        bowler = delivery.get('bowler')
                        non_striker = delivery.get('non_striker')

                        # Add player IDs
                        if batter:
                            self.get_player_id(batter, create=True)
                            self._track_player_team(batter, batting_team, player_team_counts)
                            self._track_player_stats(batter, delivery, over_num, ball_idx, is_batting=True)
                        if bowler:
                            self.get_player_id(bowler, create=True)
                            # Bowler is on the fielding team
                            bowling_team = [t for t in teams if t != batting_team]
                            if bowling_team:
                                self._track_player_team(bowler, bowling_team[0], player_team_counts)
                            self._track_player_stats(bowler, delivery, over_num, ball_idx, is_batting=False)
                        if non_striker:
                            self.get_player_id(non_striker, create=True)
                            self._track_player_team(non_striker, batting_team, player_team_counts)

        # Finalize player-team mappings (use most frequent team)
        for player, team_counts in player_team_counts.items():
            if team_counts:
                primary_team = max(team_counts.items(), key=lambda x: x[1])[0]
                self.player_to_team[player] = primary_team

        # Infer player roles from accumulated statistics
        self._infer_player_roles()

    def _track_player_team(
        self,
        player: str,
        team: str,
        player_team_counts: Dict[str, Dict[str, int]]
    ) -> None:
        """Track player-team association frequency."""
        if player not in player_team_counts:
            player_team_counts[player] = {}
        if team:
            player_team_counts[player][team] = player_team_counts[player].get(team, 0) + 1

    def _track_player_stats(
        self,
        player: str,
        delivery: Dict,
        over_num: int,
        ball_idx: int,
        is_batting: bool
    ) -> None:
        """
        Track player statistics for role inference.

        For batsmen: track batting position (over when first faced), runs, boundaries
        For bowlers: track overs bowled, wickets
        """
        if player not in self._player_stats:
            self._player_stats[player] = {
                'batting_overs': [],      # Overs when batting
                'total_runs': 0,
                'boundaries': 0,
                'balls_faced': 0,
                'balls_bowled': 0,
                'wickets': 0,
                'first_batting_over': None,
            }

        stats = self._player_stats[player]

        if is_batting:
            # Track batting statistics
            if stats['first_batting_over'] is None:
                stats['first_batting_over'] = over_num
            stats['batting_overs'].append(over_num)
            stats['balls_faced'] += 1
            runs = delivery.get('runs', {}).get('batter', 0)
            stats['total_runs'] += runs
            if runs in [4, 6]:
                stats['boundaries'] += 1
        else:
            # Track bowling statistics
            stats['balls_bowled'] += 1
            if 'wickets' in delivery:
                stats['wickets'] += len(delivery['wickets'])

    def _infer_player_roles(self) -> None:
        """
        Infer player roles from accumulated statistics.

        Role inference logic:
        - opener: First batting over is typically 0, faces new ball
        - top_order: First batting over 0-3, anchors innings
        - middle_order: First batting over 4-10
        - finisher: First batting over 11+, high boundary rate
        - bowler: Mostly bowls, bats late or not at all
        - allrounder: Significant contributions in both
        - keeper: Would need additional data (not inferable from Cricsheet)
        """
        for player, stats in self._player_stats.items():
            balls_faced = stats['balls_faced']
            balls_bowled = stats['balls_bowled']
            first_over = stats['first_batting_over']
            boundaries = stats['boundaries']
            runs = stats['total_runs']

            # Default to unknown
            role = 'unknown'

            # Determine if primarily batter or bowler
            if balls_bowled > balls_faced * 2:
                # Bowls much more than bats - primarily a bowler
                role = 'bowler'
            elif balls_faced > balls_bowled * 2:
                # Bats much more than bowls - primarily a batter
                if first_over is not None:
                    if first_over <= 1:
                        role = 'opener'
                    elif first_over <= 4:
                        role = 'top_order'
                    elif first_over <= 12:
                        role = 'middle_order'
                    else:
                        role = 'finisher'

                    # Check for finisher characteristics (high boundary rate)
                    if balls_faced > 20:
                        boundary_rate = boundaries / balls_faced
                        if boundary_rate > 0.25 and first_over >= 10:
                            role = 'finisher'
            else:
                # Roughly equal - all-rounder
                if balls_faced >= 50 and balls_bowled >= 50:
                    role = 'allrounder'
                elif balls_bowled > balls_faced:
                    role = 'bowler'
                elif first_over is not None:
                    if first_over <= 4:
                        role = 'top_order'
                    else:
                        role = 'middle_order'

            self.player_to_role[player] = role

    def save(self, path: str) -> None:
        """Save mappings to a pickle file."""
        data = {
            'venue_to_id': self.venue_to_id,
            'team_to_id': self.team_to_id,
            'player_to_id': self.player_to_id,
            'id_to_venue': self.id_to_venue,
            'id_to_team': self.id_to_team,
            'id_to_player': self.id_to_player,
            'player_to_team': self.player_to_team,
            'player_to_role': self.player_to_role,
            '_next_venue_id': self._next_venue_id,
            '_next_team_id': self._next_team_id,
            '_next_player_id': self._next_player_id,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'EntityMapper':
        """Load mappings from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        mapper = cls()
        mapper.venue_to_id = data['venue_to_id']
        mapper.team_to_id = data['team_to_id']
        mapper.player_to_id = data['player_to_id']
        mapper.id_to_venue = data['id_to_venue']
        mapper.id_to_team = data['id_to_team']
        mapper.id_to_player = data['id_to_player']
        mapper.player_to_team = data.get('player_to_team', {})
        mapper.player_to_role = data.get('player_to_role', {})
        mapper._next_venue_id = data['_next_venue_id']
        mapper._next_team_id = data['_next_team_id']
        mapper._next_player_id = data['_next_player_id']

        return mapper

    @property
    def num_roles(self) -> int:
        """Number of role categories."""
        return NUM_ROLES

    def __repr__(self) -> str:
        return (
            f"EntityMapper("
            f"venues={self.num_venues}, "
            f"teams={self.num_teams}, "
            f"players={self.num_players})"
        )
