"""
Feature Computation Utilities for Cricket Ball Prediction

Computes normalized features for all node types in the heterogeneous graph.
All features are normalized to approximately [0, 1] or [-1, 1] range.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def _flatten_deliveries(overs_data: List[Dict]) -> List[Dict]:
    """Flatten nested over/delivery structure into a list of deliveries."""
    deliveries = []
    for over_data in overs_data:
        over_num = over_data['over']
        for ball_idx, delivery in enumerate(over_data['deliveries']):
            delivery_copy = delivery.copy()
            delivery_copy['_over'] = over_num
            delivery_copy['_ball_in_over'] = ball_idx
            deliveries.append(delivery_copy)
    return deliveries


def _get_ball_number(delivery: Dict) -> int:
    """Get the ball number (0-indexed) from delivery."""
    return delivery['_over'] * 6 + delivery['_ball_in_over']


def _count_runs_wickets(deliveries: List[Dict]) -> Tuple[int, int, int]:
    """Count total runs, wickets, and legal balls from deliveries."""
    total_runs = 0
    wickets = 0
    legal_balls = 0

    for d in deliveries:
        total_runs += d['runs']['total']
        if 'wickets' in d:
            wickets += len(d['wickets'])
        # Count legal balls (exclude wides and no-balls for ball count)
        extras = d.get('extras', {})
        if 'wides' not in extras and 'noballs' not in extras:
            legal_balls += 1

    return total_runs, wickets, legal_balls


def compute_score_state(
    deliveries: List[Dict],
    innings_num: int,
    is_womens: bool = False
) -> List[float]:
    """
    Compute score state features.

    Args:
        deliveries: List of deliveries so far in this innings
        innings_num: 1 or 2
        is_womens: Whether this is women's cricket (different scoring distributions)

    Returns:
        [runs/250, wickets/10, balls/120, innings_indicator, is_womens_cricket]
    """
    runs, wickets, balls = _count_runs_wickets(deliveries)

    # Women's cricket indicator: different scoring patterns and distributions
    is_womens_flag = 1.0 if is_womens else 0.0

    return [
        min(runs / 250.0, 1.0),           # runs normalized
        wickets / 10.0,                    # wickets normalized
        min(balls / 120.0, 1.0),          # balls normalized (T20 = 120 balls)
        1.0 if innings_num == 2 else 0.0, # innings indicator
        is_womens_flag                     # women's cricket indicator
    ]


def compute_chase_state(
    deliveries: List[Dict],
    target: Optional[int],
    innings_num: int
) -> List[float]:
    """
    Compute chase state features (2nd innings only).

    Enhanced to include detailed RRR and chase difficulty features
    for better 2nd innings modeling.

    Args:
        deliveries: List of deliveries so far
        target: Target score (None for 1st innings)
        innings_num: 1 or 2

    Returns:
        [runs_needed/250, required_rate/20, is_chase, rrr_normalized,
         chase_difficulty, balls_remaining_norm, wickets_remaining_norm]
    """
    if innings_num == 1 or target is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # Not a chase

    runs, wickets, balls = _count_runs_wickets(deliveries)
    runs_needed = max(target - runs, 0)
    balls_remaining = max(120 - balls, 1)  # Avoid division by zero
    overs_remaining = balls_remaining / 6.0
    wickets_remaining = 10 - wickets

    required_rate = runs_needed / overs_remaining if overs_remaining > 0 else 0.0

    # Chase difficulty: categorize based on RRR and resources
    # comfortable (<6), gettable (6-8), challenging (8-10), difficult (10-12), improbable (>12)
    if required_rate < 6:
        chase_difficulty = 0.0  # comfortable
    elif required_rate < 8:
        chase_difficulty = 0.25  # gettable
    elif required_rate < 10:
        chase_difficulty = 0.5  # challenging
    elif required_rate < 12:
        chase_difficulty = 0.75  # difficult
    else:
        chase_difficulty = 1.0  # improbable

    return [
        min(runs_needed / 250.0, 1.0),       # runs needed normalized
        min(required_rate / 20.0, 1.0),       # RRR normalized (20 is extreme)
        1.0,                                   # is_chase flag
        min(required_rate / 12.0, 1.0),       # RRR normalized to par (12 is very high)
        chase_difficulty,                      # categorical difficulty
        balls_remaining / 120.0,               # balls remaining normalized
        wickets_remaining / 10.0,              # wickets remaining normalized
    ]


def compute_phase_state(balls_bowled: int, is_super_over: bool = False) -> List[float]:
    """
    Compute match phase features.

    Args:
        balls_bowled: Number of balls bowled in innings
        is_super_over: Whether this is a super over (tie-breaker)

    Returns:
        [is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over]
    """
    over = balls_bowled // 6
    ball_in_over = balls_bowled % 6

    # Powerplay: overs 0-5 (first 6 overs)
    is_powerplay = 1.0 if over < 6 else 0.0

    # Middle: overs 6-15
    is_middle = 1.0 if 6 <= over < 16 else 0.0

    # Death: overs 16-19
    is_death = 1.0 if over >= 16 else 0.0

    # Progress through current over
    over_progress = ball_in_over / 6.0

    # Cold start indicator: first ball of innings has no history
    # This helps the model learn first-ball-specific behavior
    is_first_ball = 1.0 if balls_bowled == 0 else 0.0

    # Super over indicator: tie-breaker has unique dynamics
    # Only 6 balls, extreme pressure, different player selection
    is_super_over_flag = 1.0 if is_super_over else 0.0

    return [is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over_flag]


def compute_time_pressure(
    balls_bowled: int,
    target: Optional[int],
    runs_scored: int,
    innings_num: int
) -> List[float]:
    """
    Compute time pressure features.

    Args:
        balls_bowled: Balls bowled so far
        target: Chase target (None for 1st innings)
        runs_scored: Runs scored so far
        innings_num: 1 or 2

    Returns:
        [balls_remaining/120, urgency, is_final_over]
    """
    balls_remaining = max(120 - balls_bowled, 0)
    current_over = balls_bowled // 6

    # Urgency calculation
    urgency = 0.0
    if innings_num == 2 and target is not None:
        runs_needed = max(target - runs_scored, 0)
        if balls_remaining > 0:
            required_rate = (runs_needed / balls_remaining) * 6  # per over
            # Urgency increases as RRR exceeds comfortable rate (~8)
            urgency = min(max(required_rate - 8.0, 0.0) / 12.0, 1.0)

    is_final_over = 1.0 if current_over >= 19 else 0.0

    return [
        balls_remaining / 120.0,  # balls remaining normalized
        urgency,                   # urgency score
        is_final_over             # final over indicator
    ]


def compute_wicket_buffer(wickets_fallen: int) -> List[float]:
    """
    Compute wicket buffer features.

    Args:
        wickets_fallen: Number of wickets fallen

    Returns:
        [wickets_in_hand/10, is_tail]
    """
    wickets_in_hand = 10 - wickets_fallen
    is_tail = 1.0 if wickets_in_hand < 3 else 0.0

    return [
        wickets_in_hand / 10.0,
        is_tail
    ]


def compute_batsman_state(
    deliveries: List[Dict],
    batsman_name: str
) -> List[float]:
    """
    Compute batsman performance features for current innings.

    Used for both striker and non-striker state computation.

    Args:
        deliveries: All deliveries in this innings so far
        batsman_name: Name of the batsman

    Returns:
        [runs/100, balls/60, sr/200, dots_pct, is_set, boundaries/10, is_debut_ball]
    """
    runs = 0
    balls_faced = 0
    dots = 0
    boundaries = 0

    for d in deliveries:
        if d['batter'] == batsman_name:
            runs += d['runs']['batter']
            # Count legal balls faced
            extras = d.get('extras', {})
            if 'wides' not in extras:
                balls_faced += 1
                if d['runs']['batter'] == 0:
                    dots += 1
                if d['runs']['batter'] in [4, 6]:
                    boundaries += 1

    strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0.0
    dots_pct = (dots / balls_faced) if balls_faced > 0 else 0.0
    is_set = 1.0 if balls_faced > 10 else 0.0

    # Cold start indicator: debut ball means this batsman has not faced any balls yet
    # This is crucial for new batsmen after a wicket falls - the model must rely
    # on player embeddings rather than in-innings performance
    is_debut_ball = 1.0 if balls_faced == 0 else 0.0

    return [
        min(runs / 100.0, 1.0),           # runs normalized
        min(balls_faced / 60.0, 1.0),     # balls faced normalized
        min(strike_rate / 200.0, 1.0),    # strike rate normalized
        dots_pct,                          # dot ball percentage
        is_set,                            # settled indicator
        min(boundaries / 10.0, 1.0),      # boundaries normalized
        is_debut_ball                      # cold start indicator
    ]


def compute_balls_since_on_strike(
    deliveries: List[Dict],
    striker_name: str
) -> float:
    """
    Compute how many balls have passed since the striker last faced a delivery.

    This captures the "cold restart" effect where a batsman who just came on strike
    (after being non-striker for several balls) is less "in rhythm" than one who
    has been continuously facing.

    Args:
        deliveries: All deliveries in this innings so far
        striker_name: Name of the current striker

    Returns:
        Normalized balls since last facing (0 = faced last ball, 1 = 12+ balls ago)
    """
    if len(deliveries) == 0:
        return 1.0  # No history, maximum "cold" state

    # Find the most recent ball this striker faced
    last_faced_idx = -1
    for i, d in enumerate(deliveries):
        if d['batter'] == striker_name:
            last_faced_idx = i

    if last_faced_idx == -1:
        # Striker hasn't faced any balls yet (debut)
        return 1.0

    # Calculate balls since last faced
    balls_since = len(deliveries) - last_faced_idx - 1

    # Normalize: 0 = just faced, 1 = 12+ balls since facing
    # 12 balls = 2 overs is the maximum "cold" window
    return min(balls_since / 12.0, 1.0)


def compute_balls_since_as_nonstriker(
    deliveries: List[Dict],
    nonstriker_name: str
) -> float:
    """
    Compute how many balls have passed since the non-striker last faced a delivery.

    This is the Z2 symmetric counterpart to balls_since_on_strike for the striker.
    Captures the "cold restart" effect for the non-striker - if they haven't faced
    for a while, they may be "cold" when they next come on strike.

    Args:
        deliveries: All deliveries in this innings so far
        nonstriker_name: Name of the current non-striker

    Returns:
        Normalized balls since last facing (0 = faced last ball, 1 = 12+ balls ago)
    """
    if len(deliveries) == 0:
        return 1.0  # No history, maximum "cold" state

    # Find the most recent ball this non-striker faced (when they were striker)
    last_faced_idx = -1
    for i, d in enumerate(deliveries):
        if d['batter'] == nonstriker_name:
            last_faced_idx = i

    if last_faced_idx == -1:
        # Non-striker hasn't faced any balls yet (debut)
        return 1.0

    # Calculate balls since last faced
    balls_since = len(deliveries) - last_faced_idx - 1

    # Normalize: 0 = just faced, 1 = 12+ balls since facing
    # 12 balls = 2 overs is the maximum "cold" window
    return min(balls_since / 12.0, 1.0)


def compute_striker_state(
    deliveries: List[Dict],
    striker_name: str
) -> List[float]:
    """
    Compute striker performance features for current innings.

    Args:
        deliveries: All deliveries in this innings so far
        striker_name: Name of the current striker

    Returns:
        [runs/100, balls/60, sr/200, dots_pct, is_set, boundaries/10, is_debut_ball,
         balls_since_on_strike]
    """
    base_features = compute_batsman_state(deliveries, striker_name)
    balls_since = compute_balls_since_on_strike(deliveries, striker_name)
    return base_features + [balls_since]


def compute_nonstriker_state(
    deliveries: List[Dict],
    nonstriker_name: str
) -> List[float]:
    """
    Compute non-striker performance features for current innings.

    Now Z2 symmetric with striker_state (both have 8 features).

    Args:
        deliveries: All deliveries in this innings so far
        nonstriker_name: Name of the current non-striker

    Returns:
        [runs/100, balls/60, sr/200, dots_pct, is_set, boundaries/10, is_debut_ball,
         balls_since_as_nonstriker]
    """
    base_features = compute_batsman_state(deliveries, nonstriker_name)
    balls_since = compute_balls_since_as_nonstriker(deliveries, nonstriker_name)
    return base_features + [balls_since]


def compute_bowler_state(
    deliveries: List[Dict],
    bowler_name: str
) -> List[float]:
    """
    Compute bowler performance features for current innings.

    Args:
        deliveries: All deliveries in this innings so far
        bowler_name: Name of the current bowler

    Returns:
        [balls/24, runs/50, wickets/5, econ/15, dots_pct, threat]
    """
    runs_conceded = 0
    balls_bowled = 0
    wickets = 0
    dots = 0

    for d in deliveries:
        if d['bowler'] == bowler_name:
            runs_conceded += d['runs']['total']
            wickets += len(d.get('wickets', []))
            # Count legal balls
            extras = d.get('extras', {})
            if 'wides' not in extras and 'noballs' not in extras:
                balls_bowled += 1
                if d['runs']['total'] == 0:
                    dots += 1

    overs_bowled = balls_bowled / 6.0 if balls_bowled > 0 else 0.0
    economy = (runs_conceded / overs_bowled) if overs_bowled > 0 else 0.0
    dots_pct = (dots / balls_bowled) if balls_bowled > 0 else 0.0

    # Threat level: combination of wicket-taking and economy
    threat = 0.0
    if balls_bowled > 0:
        wicket_rate = wickets / (balls_bowled / 6.0)  # wickets per over
        economy_factor = max(1.0 - economy / 10.0, 0.0)  # lower economy = higher threat
        threat = min((wicket_rate * 0.5 + economy_factor * 0.5), 1.0)

    return [
        min(balls_bowled / 24.0, 1.0),   # balls (4 overs max)
        min(runs_conceded / 50.0, 1.0),   # runs conceded
        min(wickets / 5.0, 1.0),          # wickets
        min(economy / 15.0, 1.0),         # economy
        dots_pct,                          # dot ball percentage
        threat                             # threat level
    ]


def compute_partnership(
    deliveries: List[Dict],
    striker_name: str,
    non_striker_name: str
) -> List[float]:
    """
    Compute current partnership features.

    Args:
        deliveries: All deliveries in this innings so far
        striker_name: Current striker
        non_striker_name: Current non-striker

    Returns:
        [runs/100, balls/60, run_rate/10, stability]
    """
    partnership_runs = 0
    partnership_balls = 0
    partnership_started = False

    # Find when this partnership started (last wicket or start of innings)
    last_wicket_idx = -1
    for i, d in enumerate(deliveries):
        if 'wickets' in d:
            last_wicket_idx = i

    # Count partnership from after last wicket
    for d in deliveries[last_wicket_idx + 1:]:
        current_pair = {d['batter'], d.get('non_striker')}
        target_pair = {striker_name, non_striker_name}

        if current_pair == target_pair:
            partnership_started = True
            partnership_runs += d['runs']['total']
            # Count legal balls
            extras = d.get('extras', {})
            if 'wides' not in extras and 'noballs' not in extras:
                partnership_balls += 1

    overs = partnership_balls / 6.0 if partnership_balls > 0 else 0.0
    run_rate = partnership_runs / overs if overs > 0 else 0.0

    # Stability: longer partnerships are more stable
    stability = min(partnership_balls / 30.0, 1.0)

    return [
        min(partnership_runs / 100.0, 1.0),   # partnership runs
        min(partnership_balls / 60.0, 1.0),   # partnership balls
        min(run_rate / 10.0, 1.0),            # run rate
        stability                              # stability indicator
    ]


def compute_dynamics(
    deliveries: List[Dict],
    target: Optional[int],
    innings_num: int,
    lookback: int = 12
) -> Dict[str, List[float]]:
    """
    Compute momentum and dynamics features.

    Args:
        deliveries: All deliveries so far
        target: Chase target (None for 1st innings)
        innings_num: 1 or 2
        lookback: Number of recent balls to consider

    Returns:
        Dict with batting_momentum, bowling_momentum, pressure_index, dot_pressure (3 features)
    """
    # Get recent deliveries
    recent = deliveries[-lookback:] if len(deliveries) >= lookback else deliveries

    if len(recent) == 0:
        return {
            'batting_momentum': [0.0],
            'bowling_momentum': [0.0],
            'pressure_index': [0.0],
            'dot_pressure': [0.0, 0.0, 1.0]  # No wicket yet = max "balls since"
        }

    # Calculate recent run rate
    recent_runs = sum(d['runs']['total'] for d in recent)
    recent_balls = len(recent)
    recent_overs = recent_balls / 6.0
    recent_rr = recent_runs / recent_overs if recent_overs > 0 else 0.0

    # Expected run rate based on phase
    total_balls = len(deliveries)
    current_over = total_balls // 6
    if current_over < 6:
        expected_rr = 7.5  # Powerplay expectation
    elif current_over < 16:
        expected_rr = 8.0  # Middle overs
    else:
        expected_rr = 10.0  # Death overs

    # Batting momentum: positive if scoring above expectation
    batting_momentum = (recent_rr / expected_rr - 1.0) if expected_rr > 0 else 0.0
    batting_momentum = max(min(batting_momentum, 1.0), -1.0)

    # Bowling momentum: inverse of batting
    bowling_momentum = -batting_momentum

    # Pressure index
    pressure = 0.0
    _, wickets, _ = _count_runs_wickets(deliveries)

    if innings_num == 2 and target is not None:
        runs_scored = sum(d['runs']['total'] for d in deliveries)
        runs_needed = max(target - runs_scored, 0)
        balls_remaining = max(120 - total_balls, 1)
        rrr = (runs_needed / (balls_remaining / 6.0))

        # Pressure increases with: high RRR, few wickets in hand, late overs
        rrr_pressure = min(rrr / 15.0, 1.0)
        wicket_pressure = wickets / 10.0
        time_pressure = total_balls / 120.0

        pressure = (rrr_pressure * 0.4 + wicket_pressure * 0.3 + time_pressure * 0.3)
    else:
        # First innings: pressure based on wickets and phase
        wicket_pressure = wickets / 10.0
        pressure = wicket_pressure * 0.5

    # Dot ball pressure
    consecutive_dots = 0
    balls_since_boundary = 0
    found_boundary = False

    for d in reversed(recent):
        if d['runs']['total'] == 0:
            consecutive_dots += 1
        else:
            break

    for d in reversed(deliveries):
        if d['runs']['batter'] in [4, 6]:
            found_boundary = True
            break
        balls_since_boundary += 1

    # Balls since last wicket (bowler momentum indicator - R4 feedback loop)
    # After taking a wicket, bowlers often have increased confidence
    balls_since_wicket = len(deliveries)  # Default: no wicket yet
    for i, d in enumerate(deliveries):
        if 'wickets' in d:
            balls_since_wicket = len(deliveries) - i - 1

    # Normalize: 0 = just fell, 1 = 30+ balls since wicket (partnership established)
    balls_since_wicket_norm = min(balls_since_wicket / 30.0, 1.0)

    return {
        'batting_momentum': [batting_momentum],
        'bowling_momentum': [bowling_momentum],
        'pressure_index': [min(pressure, 1.0)],
        'dot_pressure': [
            min(consecutive_dots / 6.0, 1.0),
            min(balls_since_boundary / 12.0, 1.0),
            balls_since_wicket_norm  # NEW: bowler momentum indicator
        ]
    }


def compute_ball_features(delivery: Dict) -> List[float]:
    """
    Compute features for a single ball node.

    Args:
        delivery: Single delivery dict with _over and _ball_in_over added

    Returns:
        List of 18 features:
        [runs/6, is_wicket, over/20, ball_in_over/6, is_boundary,
         is_wide, is_noball, is_bye, is_legbye,
         wicket_bowled, wicket_caught, wicket_lbw, wicket_run_out, wicket_stumped, wicket_other,
         striker_run_out, nonstriker_run_out, bowling_end]
    """
    runs = delivery['runs']['total']
    is_wicket = 1.0 if 'wickets' in delivery else 0.0
    over = delivery.get('_over', 0)
    ball_in_over = delivery.get('_ball_in_over', 0)
    is_boundary = 1.0 if delivery['runs']['batter'] in [4, 6] else 0.0

    # Bowling end: which end of the pitch (alternates by over)
    # This captures end-specific patterns (footmarks, pitch wear)
    bowling_end = float(over % 2)

    # Extract extras information
    extras = delivery.get('extras', {})
    is_wide = 1.0 if 'wides' in extras else 0.0
    is_noball = 1.0 if 'noballs' in extras else 0.0
    is_bye = 1.0 if 'byes' in extras else 0.0
    is_legbye = 1.0 if 'legbyes' in extras else 0.0

    # Wicket type one-hot encoding (6 categories)
    # This captures important information about HOW the wicket fell:
    # - bowled/lbw: bowler skill, good line and length
    # - caught: aggression/risk-taking by batsman
    # - run_out: partnership running decisions
    # - stumped: batsman error against spin
    wicket_bowled = 0.0
    wicket_caught = 0.0
    wicket_lbw = 0.0
    wicket_run_out = 0.0
    wicket_stumped = 0.0
    wicket_other = 0.0

    # Run-out attribution: WHO was run out (striker vs non-striker)
    # This matters because:
    # - Striker run out: misjudged single, hesitation
    # - Non-striker run out: backing up too far, miscommunication
    striker_run_out = 0.0
    nonstriker_run_out = 0.0

    # Get striker and non-striker names for run-out attribution
    striker_name = delivery.get('batter', '')
    nonstriker_name = delivery.get('non_striker', '')

    if 'wickets' in delivery and len(delivery['wickets']) > 0:
        wicket_kind = delivery['wickets'][0].get('kind', 'other').lower()
        if wicket_kind == 'bowled':
            wicket_bowled = 1.0
        elif wicket_kind in ['caught', 'caught and bowled']:
            wicket_caught = 1.0
        elif wicket_kind == 'lbw':
            wicket_lbw = 1.0
        elif wicket_kind == 'run out':
            wicket_run_out = 1.0
            # Determine WHO was run out for attribution
            player_out = delivery['wickets'][0].get('player_out', '')
            if player_out == striker_name:
                striker_run_out = 1.0
            elif player_out == nonstriker_name:
                nonstriker_run_out = 1.0
            # If player_out doesn't match either, leave both as 0
            # (data quality issue - rare)
        elif wicket_kind == 'stumped':
            wicket_stumped = 1.0
        else:
            # hit wicket, obstructing the field, timed out, retired, etc.
            wicket_other = 1.0

    return [
        runs / 6.0,           # runs normalized (max single ball = 6 + extras)
        is_wicket,            # wicket indicator
        over / 20.0,          # over normalized
        ball_in_over / 6.0,   # ball in over normalized
        is_boundary,          # boundary indicator
        is_wide,              # wide ball indicator
        is_noball,            # no-ball indicator
        is_bye,               # bye indicator
        is_legbye,            # leg bye indicator
        wicket_bowled,        # bowled dismissal
        wicket_caught,        # caught dismissal
        wicket_lbw,           # LBW dismissal
        wicket_run_out,       # run out dismissal (any)
        wicket_stumped,       # stumped dismissal
        wicket_other,         # other dismissal types
        striker_run_out,      # striker was run out
        nonstriker_run_out,   # non-striker was run out
        bowling_end           # which end of pitch (0 or 1)
    ]


# Number of features per ball node (for validation)
# 9 original + 6 wicket type one-hot + 2 run-out attribution + 1 bowling_end = 18 features
BALL_FEATURE_DIM = 18


def outcome_to_class(delivery: Dict) -> int:
    """
    Convert delivery outcome to class label.

    Classes:
        0: Dot (0 runs, no wicket)
        1: Single (1 run)
        2: Two (2 runs)
        3: Three (3 runs)
        4: Four (boundary)
        5: Six (boundary)
        6: Wicket

    Args:
        delivery: Single delivery dict

    Returns:
        Class label 0-6
    """
    # Wicket takes precedence
    if 'wickets' in delivery:
        return 6

    batter_runs = delivery['runs']['batter']

    if batter_runs == 0:
        return 0  # Dot
    elif batter_runs == 1:
        return 1  # Single
    elif batter_runs == 2:
        return 2  # Two
    elif batter_runs == 3:
        return 3  # Three
    elif batter_runs == 4:
        return 4  # Four
    elif batter_runs >= 6:
        return 5  # Six
    else:
        return 0  # Default to dot for edge cases


def class_to_outcome_name(class_idx: int) -> str:
    """Convert class index to human-readable name."""
    names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
    return names[class_idx] if 0 <= class_idx < len(names) else 'Unknown'


# Utility function to prepare deliveries from match data
def prepare_deliveries(innings_data: Dict) -> List[Dict]:
    """
    Prepare deliveries list from innings data with over/ball metadata.

    Args:
        innings_data: Single innings dict from match JSON

    Returns:
        List of delivery dicts with _over and _ball_in_over fields
    """
    return _flatten_deliveries(innings_data.get('overs', []))
