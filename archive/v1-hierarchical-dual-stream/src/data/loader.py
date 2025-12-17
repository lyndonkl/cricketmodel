"""Load and parse Cricsheet JSON match files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Delivery:
    """Single ball delivery."""

    ball_id: int  # Sequential ball number in innings
    over: int
    ball_in_over: int  # 0-5
    batter: str
    bowler: str
    non_striker: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    is_wicket: bool
    wicket_kind: str | None
    extras_type: str | None  # wide, noball, bye, legbye


@dataclass
class Innings:
    """Single innings data."""

    batting_team: str
    bowling_team: str
    deliveries: list[Delivery]
    total_runs: int
    total_wickets: int


@dataclass
class Match:
    """Parsed match data."""

    match_id: str
    venue: str
    city: str | None
    date: str
    teams: tuple[str, str]
    toss_winner: str
    toss_decision: str  # "bat" or "field"
    innings: list[Innings]
    player_registry: dict[str, str]  # name -> uuid


def _parse_delivery(over_num: int, ball_idx: int, ball_id: int, d: dict) -> Delivery:
    """Parse single delivery from JSON."""
    runs = d["runs"]
    wicket = d.get("wickets", [{}])[0] if "wickets" in d else {}
    extras = d.get("extras", {})

    extras_type = None
    if extras:
        extras_type = next(iter(extras.keys()), None)

    return Delivery(
        ball_id=ball_id,
        over=over_num,
        ball_in_over=ball_idx,
        batter=d["batter"],
        bowler=d["bowler"],
        non_striker=d["non_striker"],
        runs_batter=runs["batter"],
        runs_extras=runs["extras"],
        runs_total=runs["total"],
        is_wicket="wickets" in d,
        wicket_kind=wicket.get("kind"),
        extras_type=extras_type,
    )


def _parse_innings(innings_data: dict, bowling_team: str) -> Innings:
    """Parse single innings from JSON."""
    deliveries = []
    ball_id = 0
    total_runs = 0
    total_wickets = 0

    for over_data in innings_data.get("overs", []):
        over_num = over_data["over"]
        for ball_idx, d in enumerate(over_data["deliveries"]):
            delivery = _parse_delivery(over_num, ball_idx, ball_id, d)
            deliveries.append(delivery)
            total_runs += delivery.runs_total
            if delivery.is_wicket:
                total_wickets += 1
            ball_id += 1

    return Innings(
        batting_team=innings_data["team"],
        bowling_team=bowling_team,
        deliveries=deliveries,
        total_runs=total_runs,
        total_wickets=total_wickets,
    )


def load_match(path: Path) -> Match:
    """Load single match from Cricsheet JSON file."""
    with open(path) as f:
        data = json.load(f)

    info = data["info"]
    teams = tuple(info["teams"])

    # Parse innings
    innings_list = []
    for i, inn_data in enumerate(data.get("innings", [])):
        batting_team = inn_data["team"]
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        innings_list.append(_parse_innings(inn_data, bowling_team))

    return Match(
        match_id=path.stem,
        venue=info.get("venue", "Unknown"),
        city=info.get("city"),
        date=info["dates"][0],
        teams=teams,
        toss_winner=info.get("toss", {}).get("winner", teams[0]),
        toss_decision=info.get("toss", {}).get("decision", "bat"),
        innings=innings_list,
        player_registry=info.get("registry", {}).get("people", {}),
    )


def load_matches_from_dir(
    data_dir: Path,
    min_balls: int = 60,
) -> Iterator[Match]:
    """Load all matches from directory, filtering incomplete ones."""
    json_files = sorted(data_dir.glob("*.json"))

    for path in json_files:
        try:
            match = load_match(path)
            # Filter incomplete matches
            total_balls = sum(len(inn.deliveries) for inn in match.innings)
            if total_balls >= min_balls:
                yield match
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path.name}: {e}")
            continue
