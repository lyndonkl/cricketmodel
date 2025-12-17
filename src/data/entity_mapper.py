"""
Entity Mapper for Cricket Ball Prediction

Maps entity names (venues, teams, players) to integer IDs for embedding layers.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Set


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

        Args:
            match_files: List of paths to match JSON files
        """
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

            # Add players from all innings
            for innings in match_data.get('innings', []):
                for over_data in innings.get('overs', []):
                    for delivery in over_data.get('deliveries', []):
                        batter = delivery.get('batter')
                        bowler = delivery.get('bowler')
                        non_striker = delivery.get('non_striker')

                        if batter:
                            self.get_player_id(batter, create=True)
                        if bowler:
                            self.get_player_id(bowler, create=True)
                        if non_striker:
                            self.get_player_id(non_striker, create=True)

    def save(self, path: str) -> None:
        """Save mappings to a pickle file."""
        data = {
            'venue_to_id': self.venue_to_id,
            'team_to_id': self.team_to_id,
            'player_to_id': self.player_to_id,
            'id_to_venue': self.id_to_venue,
            'id_to_team': self.id_to_team,
            'id_to_player': self.id_to_player,
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
        mapper._next_venue_id = data['_next_venue_id']
        mapper._next_team_id = data['_next_team_id']
        mapper._next_player_id = data['_next_player_id']

        return mapper

    def __repr__(self) -> str:
        return (
            f"EntityMapper("
            f"venues={self.num_venues}, "
            f"teams={self.num_teams}, "
            f"players={self.num_players})"
        )
