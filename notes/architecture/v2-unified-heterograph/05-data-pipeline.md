# Data Pipeline

## Overview

The data pipeline converts raw cricket match data into PyTorch Geometric `HeteroData` objects suitable for the unified graph model.

```
Raw Match JSON → CricketDataset → HeteroData → DataLoader → Model
```

---

## 1. Raw Data Format

### Input: Ball-by-Ball Match Data

Each match is stored as JSON with ball-by-ball information:

```json
{
  "match_id": "12345",
  "venue": "MCG",
  "team1": "India",
  "team2": "Australia",
  "innings": [
    {
      "batting_team": "India",
      "bowling_team": "Australia",
      "deliveries": [
        {
          "ball": 0.1,
          "batsman": "Rohit Sharma",
          "bowler": "Mitchell Starc",
          "runs_batsman": 0,
          "runs_extras": 0,
          "runs_total": 0,
          "wicket": null
        },
        {
          "ball": 0.2,
          "batsman": "Rohit Sharma",
          "bowler": "Mitchell Starc",
          "runs_batsman": 4,
          "runs_extras": 0,
          "runs_total": 4,
          "wicket": null
        },
        ...
      ]
    }
  ]
}
```

---

## 2. Entity Mappings

### ID Lookups

We maintain mappings from entity names to integer IDs:

```python
class EntityMapper:
    def __init__(self):
        self.venue_to_id = {}
        self.team_to_id = {}
        self.player_to_id = {}

    def get_venue_id(self, venue_name):
        if venue_name not in self.venue_to_id:
            self.venue_to_id[venue_name] = len(self.venue_to_id)
        return self.venue_to_id[venue_name]

    def get_team_id(self, team_name):
        if team_name not in self.team_to_id:
            self.team_to_id[team_name] = len(self.team_to_id)
        return self.team_to_id[team_name]

    def get_player_id(self, player_name):
        if player_name not in self.player_to_id:
            self.player_to_id[player_name] = len(self.player_to_id)
        return self.player_to_id[player_name]
```

---

## 3. HeteroData Construction

### Per-Ball Prediction Sample

Each training sample is a `HeteroData` object representing the state BEFORE a specific ball, with the label being the outcome of that ball.

```python
from torch_geometric.data import HeteroData

def create_hetero_data(match_data, ball_idx, entity_mapper):
    """
    Create a HeteroData object for predicting ball at ball_idx.

    Args:
        match_data: Parsed match dictionary
        ball_idx: Index of ball to predict (0-indexed)
        entity_mapper: EntityMapper for ID lookups

    Returns:
        HeteroData object with all nodes and edges
    """
    data = HeteroData()

    innings = match_data['innings'][current_innings_idx]
    deliveries = innings['deliveries'][:ball_idx]  # History BEFORE this ball

    # ============ NODE FEATURES ============

    # --- Global nodes ---
    data['global'].x = torch.tensor([
        [entity_mapper.get_venue_id(match_data['venue'])],
        [entity_mapper.get_team_id(innings['batting_team'])],
        [entity_mapper.get_team_id(innings['bowling_team'])],
    ], dtype=torch.long)
    data['global'].node_names = ['venue', 'batting_team', 'bowling_team']

    # --- State nodes ---
    state_features = compute_state_features(deliveries, innings)
    data['state'].x = torch.tensor([
        state_features['score_state'],      # [runs/250, wickets/10, balls/120, innings]
        state_features['chase_state'],       # [runs_needed/250, RRR/20, is_chase]
        state_features['phase_state'],       # [is_pp, is_middle, is_death, over_progress]
        state_features['time_pressure'],     # [balls_left/120, urgency, is_final]
        state_features['wicket_buffer'],     # [wickets_in_hand/10, is_tail]
    ], dtype=torch.float)

    # --- Actor nodes ---
    current_ball = innings['deliveries'][ball_idx]
    actor_features = compute_actor_features(deliveries, current_ball, entity_mapper)

    data['actor'].x = torch.tensor([
        [actor_features['striker_id']],                # Entity ID
        actor_features['striker_state'],               # [6 features]
        [actor_features['bowler_id']],                 # Entity ID
        actor_features['bowler_state'],                # [6 features]
        actor_features['partnership'],                 # [4 features]
    ])  # Note: Mixed types handled by separate encoders

    # --- Dynamics nodes ---
    dynamics_features = compute_dynamics_features(deliveries)
    data['dynamics'].x = torch.tensor([
        [dynamics_features['batting_momentum']],
        [dynamics_features['bowling_momentum']],
        [dynamics_features['pressure_index']],
        dynamics_features['dot_pressure'],             # [2 features]
    ], dtype=torch.float)

    # --- Ball nodes (history) ---
    ball_features = []
    ball_bowler_ids = []
    ball_batsman_ids = []

    for d in deliveries:
        ball_features.append([
            d['runs_total'] / 6,
            1.0 if d['wicket'] else 0.0,
            parse_over(d['ball']) / 20,
            (parse_ball_in_over(d['ball'])) / 6,
            1.0 if d['runs_batsman'] >= 4 else 0.0,
        ])
        ball_bowler_ids.append(entity_mapper.get_player_id(d['bowler']))
        ball_batsman_ids.append(entity_mapper.get_player_id(d['batsman']))

    if len(deliveries) > 0:
        data['ball'].x = torch.tensor(ball_features, dtype=torch.float)
        data['ball'].bowler_ids = torch.tensor(ball_bowler_ids, dtype=torch.long)
        data['ball'].batsman_ids = torch.tensor(ball_batsman_ids, dtype=torch.long)

    # --- Query node ---
    data['query'].x = torch.zeros(1, 1)  # Placeholder, encoder provides learned embedding

    # ============ EDGE INDICES ============

    num_balls = len(deliveries)

    # --- Hierarchical edges ---
    data['global', 'conditions', 'state'].edge_index = create_all_to_all_edges(3, 5)
    data['state', 'conditions', 'actor'].edge_index = create_all_to_all_edges(5, 5)
    data['actor', 'conditions', 'dynamics'].edge_index = create_all_to_all_edges(5, 4)

    # --- Intra-layer edges ---
    data['global', 'relates_to', 'global'].edge_index = create_self_loop_and_full(3)
    data['state', 'relates_to', 'state'].edge_index = create_self_loop_and_full(5)
    data['dynamics', 'relates_to', 'dynamics'].edge_index = create_self_loop_and_full(4)

    # Actor matchup edges (semantic structure)
    data['actor', 'matchup', 'actor'].edge_index = torch.tensor([
        [0, 1, 1, 0,  2, 3, 3, 2,  0, 2, 2, 0,  1, 4, 4, 1,  3, 4, 4, 3],
        [1, 0, 4, 2,  3, 2, 4, 0,  2, 0, 0, 2,  4, 1, 1, 4,  4, 3, 3, 4],
    ], dtype=torch.long)

    # --- Temporal edges ---
    if num_balls > 1:
        # Precedes: ball i -> ball i+1
        src = list(range(num_balls - 1))
        dst = list(range(1, num_balls))
        data['ball', 'precedes', 'ball'].edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Same bowler edges
        same_bowler_src, same_bowler_dst = [], []
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                if ball_bowler_ids[i] == ball_bowler_ids[j]:
                    same_bowler_src.extend([i, j])
                    same_bowler_dst.extend([j, i])

        if same_bowler_src:
            data['ball', 'same_bowler', 'ball'].edge_index = torch.tensor(
                [same_bowler_src, same_bowler_dst], dtype=torch.long
            )

        # Same batsman edges
        same_batsman_src, same_batsman_dst = [], []
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                if ball_batsman_ids[i] == ball_batsman_ids[j]:
                    same_batsman_src.extend([i, j])
                    same_batsman_dst.extend([j, i])

        if same_batsman_src:
            data['ball', 'same_batsman', 'ball'].edge_index = torch.tensor(
                [same_batsman_src, same_batsman_dst], dtype=torch.long
            )

    # --- Cross-domain edges ---
    if num_balls > 0:
        # Ball -> bowler identity (actor node 2)
        data['ball', 'bowled_by', 'actor'].edge_index = torch.tensor([
            list(range(num_balls)),
            [2] * num_balls,  # All point to bowler_identity node
        ], dtype=torch.long)

        # Ball -> striker identity (actor node 0)
        data['ball', 'faced_by', 'actor'].edge_index = torch.tensor([
            list(range(num_balls)),
            [0] * num_balls,  # All point to striker_identity node
        ], dtype=torch.long)

        # Dynamics <- recent balls (last 12)
        recent_balls = list(range(max(0, num_balls - 12), num_balls))
        data['ball', 'informs', 'dynamics'].edge_index = torch.tensor([
            recent_balls * 4,
            [0, 1, 2, 3] * len(recent_balls),  # Each ball informs all dynamics nodes
        ], dtype=torch.long)

    # --- Query edges ---
    # Query attends to global
    data['query', 'attends_to', 'global'].edge_index = torch.tensor([
        [0, 0, 0],
        [0, 1, 2],
    ], dtype=torch.long)

    # Query attends to state
    data['query', 'attends_to', 'state'].edge_index = torch.tensor([
        [0] * 5,
        list(range(5)),
    ], dtype=torch.long)

    # Query attends to actor
    data['query', 'attends_to', 'actor'].edge_index = torch.tensor([
        [0] * 5,
        list(range(5)),
    ], dtype=torch.long)

    # Query attends to dynamics
    data['query', 'attends_to', 'dynamics'].edge_index = torch.tensor([
        [0] * 4,
        list(range(4)),
    ], dtype=torch.long)

    # Query attends to all balls
    if num_balls > 0:
        data['query', 'attends_to', 'ball'].edge_index = torch.tensor([
            [0] * num_balls,
            list(range(num_balls)),
        ], dtype=torch.long)

    # ============ LABEL ============
    outcome = innings['deliveries'][ball_idx]
    data.y = torch.tensor([outcome_to_class(outcome)], dtype=torch.long)

    return data
```

---

## 4. Dataset Class

### InMemoryDataset Implementation

```python
from torch_geometric.data import InMemoryDataset, HeteroData
import os
import json

class CricketDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['matches.json']

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def download(self):
        pass  # Assume data already exists

    def process(self):
        # Load raw match data
        with open(os.path.join(self.raw_dir, 'matches.json')) as f:
            matches = json.load(f)

        # Split matches
        train_matches, val_matches, test_matches = split_matches(matches)

        if self.split == 'train':
            match_list = train_matches
        elif self.split == 'val':
            match_list = val_matches
        else:
            match_list = test_matches

        # Create entity mapper
        entity_mapper = EntityMapper()

        # Create HeteroData for each ball
        data_list = []
        for match in match_list:
            for innings_idx in range(len(match['innings'])):
                innings = match['innings'][innings_idx]
                for ball_idx in range(1, len(innings['deliveries'])):
                    # Need at least 1 ball of history
                    data = create_hetero_data(match, innings_idx, ball_idx, entity_mapper)
                    data_list.append(data)

        # Apply pre_transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save
        self.save(data_list, self.processed_paths[0])
```

---

## 5. DataLoader

### Using PyG DataLoader

```python
from torch_geometric.loader import DataLoader

def create_dataloaders(root, batch_size=32, num_workers=4):
    train_dataset = CricketDataset(root, split='train')
    val_dataset = CricketDataset(root, split='val')
    test_dataset = CricketDataset(root, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
```

### Batching Behavior

PyG's DataLoader automatically:
1. Combines multiple HeteroData objects into one batched HeteroData
2. Increments node indices to create a single disconnected graph
3. Maintains `batch` tensor to track which nodes belong to which sample

```python
for batch in train_loader:
    # batch is a HeteroData with combined graphs
    # batch['ball'].batch tells you which sample each ball belongs to

    logits = model(batch)  # [batch_size, 7]
    loss = criterion(logits, batch.y)
```

---

## 6. Feature Computation Helpers

### State Features

```python
def compute_state_features(deliveries, innings):
    runs = sum(d['runs_total'] for d in deliveries)
    wickets = sum(1 for d in deliveries if d['wicket'])
    balls = len(deliveries)
    overs = balls // 6 + (balls % 6) / 10

    # ... compute all state features

    return {
        'score_state': [runs/250, wickets/10, balls/120, innings_num],
        'chase_state': [runs_needed/250, rrr/20, is_chase],
        'phase_state': [is_pp, is_middle, is_death, ball_in_over/6],
        'time_pressure': [balls_left/120, urgency, is_final],
        'wicket_buffer': [wickets_in_hand/10, is_tail],
    }
```

### Actor Features

```python
def compute_actor_features(deliveries, current_ball, entity_mapper):
    striker = current_ball['batsman']
    bowler = current_ball['bowler']

    # Compute striker stats from history
    striker_balls = [d for d in deliveries if d['batsman'] == striker]
    striker_runs = sum(d['runs_batsman'] for d in striker_balls)
    # ... more stats

    return {
        'striker_id': entity_mapper.get_player_id(striker),
        'striker_state': [runs/100, balls/60, sr/200, dots_pct, is_set, bdry/10],
        'bowler_id': entity_mapper.get_player_id(bowler),
        'bowler_state': [balls/24, runs/50, wkts/5, econ/15, dots_pct, threat],
        'partnership': [p_runs/100, p_balls/60, p_rr/10, stability],
    }
```

### Outcome to Class

```python
def outcome_to_class(delivery):
    """Convert delivery outcome to class index."""
    if delivery['wicket']:
        return 6  # wicket
    runs = delivery['runs_batsman']
    if runs == 0:
        return 0  # dot
    elif runs == 1:
        return 1
    elif runs == 2:
        return 2
    elif runs == 3:
        return 3
    elif runs == 4:
        return 4
    elif runs >= 6:
        return 5  # six (or more)
    return 0
```

---

## 7. Data Statistics

### Expected Dataset Size

For ~500 T20 matches:
- ~120 balls per innings × 2 innings × 500 matches = ~120,000 samples
- Each sample: ~80-200 nodes, ~500-2000 edges

### Memory Considerations

- Using InMemoryDataset for simplicity
- For larger datasets, switch to on-disk Dataset with `get(idx)`
- Consider NeighborLoader for very large history graphs
