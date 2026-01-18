# Simulation & Applications Guide

This document covers how to use the trained Cricket Ball Prediction GNN model for simulations and strategic applications.

---

## 1. End-of-Over Score Simulation

Use Monte Carlo simulation to predict score distributions for an over:

```python
import numpy as np

def simulate_over(model, initial_state, n_simulations=1000):
    """Monte Carlo simulation of one over.

    Args:
        model: Trained GNN model with predict_proba method
        initial_state: Current match state (graph features)
        n_simulations: Number of Monte Carlo samples

    Returns:
        Dictionary with simulation statistics
    """
    scores = []
    wickets = []

    for _ in range(n_simulations):
        state = initial_state.copy()
        over_runs = 0
        over_wickets = 0

        for ball in range(6):
            # Get probability distribution from model
            probs = model.predict_proba(state)

            # Sample outcome from distribution
            outcome = np.random.choice(7, p=probs)

            # Update state and accumulators
            # Runs: [Dot, Single, Two, Three, Four, Six, Wicket]
            runs = [0, 1, 2, 3, 4, 6, 0][outcome]
            over_runs += runs
            if outcome == 6:  # Wicket
                over_wickets += 1

            state = update_state(state, outcome)

        scores.append(over_runs)
        wickets.append(over_wickets)

    return {
        'mean_runs': np.mean(scores),
        'std_runs': np.std(scores),
        'wicket_prob': np.mean([w > 0 for w in wickets]),
        'distribution': np.histogram(scores, bins=range(0, 40)),
        'percentiles': {
            '10th': np.percentile(scores, 10),
            '50th': np.percentile(scores, 50),
            '90th': np.percentile(scores, 90)
        }
    }
```

---

## 2. Match Outcome Prediction

Simulate remaining overs from current state to get distribution of final scores:

```python
def simulate_innings(model, current_state, overs_remaining, n_simulations=500):
    """Simulate remaining innings from current state.

    Args:
        model: Trained GNN model
        current_state: Current match state
        overs_remaining: Number of overs left
        n_simulations: Monte Carlo samples

    Returns:
        Dictionary with final score distribution
    """
    final_scores = []

    for _ in range(n_simulations):
        state = current_state.copy()
        total_runs = state['current_score']
        wickets = state['wickets_fallen']

        for over in range(overs_remaining):
            if wickets >= 10:  # All out
                break

            over_result = simulate_over(model, state, n_simulations=1)
            total_runs += over_result['mean_runs']

            # Update wickets (simplified)
            if np.random.random() < over_result['wicket_prob']:
                wickets += 1

            state = update_state_for_over(state, over_result)

        final_scores.append(total_runs)

    return {
        'mean_score': np.mean(final_scores),
        'std_score': np.std(final_scores),
        'score_distribution': final_scores,
        'confidence_interval': (
            np.percentile(final_scores, 5),
            np.percentile(final_scores, 95)
        )
    }
```

**Use Cases:**
- Win probability graphs
- Required run rate analysis
- Optimal declaration timing (in longer formats)

---

## 3. Strategic Analysis

### Bowler Effectiveness Analysis

```python
def compare_bowler_effectiveness(model, base_state, bowler_types):
    """Compare model predictions for different bowler types.

    Args:
        model: Trained GNN model
        base_state: Current match state (without bowler info)
        bowler_types: List of bowler configurations to compare

    Returns:
        Dictionary mapping bowler type to prediction metrics
    """
    results = {}

    for bowler_type in bowler_types:
        state = base_state.copy()
        state['bowler_type'] = bowler_type

        probs = model.predict_proba(state)

        # Calculate metrics
        expected_runs = sum(p * r for p, r in zip(
            probs, [0, 1, 2, 3, 4, 6, 0]
        ))
        wicket_prob = probs[6]  # Wicket class
        boundary_prob = probs[4] + probs[5]  # Four + Six

        results[bowler_type] = {
            'expected_runs': expected_runs,
            'wicket_probability': wicket_prob,
            'boundary_probability': boundary_prob,
            'dot_probability': probs[0]
        }

    return results

# Example usage
comparison = compare_bowler_effectiveness(
    model,
    current_state,
    ['pace', 'spin', 'medium']
)
# Output: Which bowler type minimizes runs / maximizes wicket chance?
```

### Batting Aggression Analysis

```python
def analyze_phase_predictions(model, base_state, phases):
    """How do predictions change through innings phases?

    Args:
        model: Trained GNN model
        base_state: Base match state
        phases: List of phase names ['powerplay', 'middle', 'death']

    Returns:
        Phase-by-phase prediction comparison
    """
    results = {}

    phase_configs = {
        'powerplay': {'over': 3, 'field_restrictions': True},
        'middle': {'over': 12, 'field_restrictions': False},
        'death': {'over': 18, 'field_restrictions': False}
    }

    for phase in phases:
        state = base_state.copy()
        state.update(phase_configs[phase])

        probs = model.predict_proba(state)

        results[phase] = {
            'expected_runs': sum(p * r for p, r in zip(
                probs, [0, 1, 2, 3, 4, 6, 0]
            )),
            'wicket_risk': probs[6],
            'boundary_chance': probs[4] + probs[5],
            'full_distribution': probs
        }

    return results
```

### Matchup Analysis

```python
def analyze_matchup(model, striker_id, bowler_id, match_context):
    """Analyze specific striker-bowler combination.

    Args:
        model: Trained GNN model
        striker_id: Batsman identifier
        bowler_id: Bowler identifier
        match_context: Additional context (venue, pitch, etc.)

    Returns:
        Matchup prediction metrics
    """
    state = build_state(
        striker=striker_id,
        bowler=bowler_id,
        **match_context
    )

    probs = model.predict_proba(state)

    # Calculate advantage metric
    baseline_probs = get_baseline_probs(model, match_context)

    advantage_score = calculate_advantage(probs, baseline_probs)

    return {
        'predictions': probs,
        'expected_runs': sum(p * r for p, r in zip(probs, [0,1,2,3,4,6,0])),
        'advantage': advantage_score,  # Positive = batsman favored
        'recommendation': 'batsman' if advantage_score > 0 else 'bowler'
    }
```

---

## 4. Applications Summary

| Application | How Model Helps |
|-------------|-----------------|
| **Fantasy Cricket** | Identify players in favorable matchups |
| **Betting Markets** | Compare model odds to market odds for edge |
| **Commentary Support** | "Model gives 25% wicket chance this over" |
| **Team Strategy** | Optimal field placement (which outcomes to defend) |
| **Player Valuation** | How much does player X shift win probability? |
| **What-If Analysis** | "If we'd used spinner at over 15, what changes?" |

---

## 5. Simulation Validation

To trust simulations, validate that model outputs match historical patterns:

### Score Distribution Validation

```python
def validate_score_distribution(model, historical_data, n_simulations=1000):
    """Compare simulated vs historical score distributions.

    Args:
        model: Trained GNN model
        historical_data: Historical match data
        n_simulations: Simulations per historical state

    Returns:
        Validation metrics
    """
    simulated_scores = []
    actual_scores = []

    for match in historical_data:
        # Get state at start of over
        state = match['initial_state']
        actual_over_runs = match['actual_over_runs']

        # Simulate
        sim_result = simulate_over(model, state, n_simulations)

        simulated_scores.append(sim_result['mean_runs'])
        actual_scores.append(actual_over_runs)

    # Calculate validation metrics
    from scipy import stats

    correlation = np.corrcoef(simulated_scores, actual_scores)[0, 1]
    mae = np.mean(np.abs(np.array(simulated_scores) - np.array(actual_scores)))

    # KS test for distribution similarity
    ks_stat, ks_pvalue = stats.ks_2samp(simulated_scores, actual_scores)

    return {
        'correlation': correlation,
        'mae': mae,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'distributions_similar': ks_pvalue > 0.05
    }
```

### Wicket Timing Validation

```python
def validate_wicket_timing(model, historical_data):
    """Check if simulated wicket timing matches real patterns.

    Returns:
        Comparison of predicted vs actual wicket rates by phase
    """
    phases = ['powerplay', 'middle', 'death']
    results = {}

    for phase in phases:
        phase_data = filter_by_phase(historical_data, phase)

        predicted_wicket_rates = []
        actual_wicket_rates = []

        for match in phase_data:
            probs = model.predict_proba(match['state'])
            predicted_wicket_rates.append(probs[6])
            actual_wicket_rates.append(match['was_wicket'])

        results[phase] = {
            'predicted_rate': np.mean(predicted_wicket_rates),
            'actual_rate': np.mean(actual_wicket_rates),
            'calibration_error': abs(
                np.mean(predicted_wicket_rates) - np.mean(actual_wicket_rates)
            )
        }

    return results
```

### Run Rate by Phase Validation

```python
def validate_run_rates(model, historical_data):
    """Verify simulated run rates match historical by innings phase."""
    phases = ['powerplay', 'middle', 'death']

    results = {}
    for phase in phases:
        phase_data = filter_by_phase(historical_data, phase)

        predicted_runs = []
        actual_runs = []

        for ball in phase_data:
            probs = model.predict_proba(ball['state'])
            expected = sum(p * r for p, r in zip(probs, [0,1,2,3,4,6,0]))
            predicted_runs.append(expected)
            actual_runs.append(ball['actual_runs'])

        results[phase] = {
            'predicted_rpo': np.mean(predicted_runs) * 6,  # Runs per over
            'actual_rpo': np.mean(actual_runs) * 6,
            'error': abs(np.mean(predicted_runs) - np.mean(actual_runs)) * 6
        }

    return results
```

---

## 6. What "Good Enough" Means for Simulations

For Monte Carlo simulations to be useful, you need:

1. **Calibrated Probabilities (ECE < 0.10)**
   - If model says 20% wicket prediction → should actually happen ~20% of the time
   - Check with `validate_wicket_timing()`

2. **Reasonable Class Coverage**
   - All 7 classes should be predicted, not just majorities
   - Check per-class F1 scores in WandB

3. **Contextual Sensitivity**
   - Different predictions for powerplay vs death overs
   - Check with `analyze_phase_predictions()`

4. **Distribution Match**
   - Simulated score distributions should match historical
   - Check with `validate_score_distribution()`

**Key Insight:** Accuracy matters less than probability quality for simulations. A model with 30% accuracy but well-calibrated probabilities is more useful for simulations than one with 35% accuracy and poor calibration.

---

## 7. Production Deployment Considerations

### Inference Speed

```python
# Batch predictions for efficiency
def batch_simulate(model, states, n_simulations=100):
    """Run simulations in batches for speed."""
    # Batch all initial predictions
    all_probs = model.predict_proba_batch(states)

    # Run simulations using cached probabilities
    # ...
```

### Model Serving

```python
# Example Flask API for predictions
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model('best_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    state = request.json
    probs = model.predict_proba(state)

    return jsonify({
        'probabilities': probs.tolist(),
        'expected_runs': sum(p * r for p, r in zip(probs, [0,1,2,3,4,6,0])),
        'top_prediction': ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket'][np.argmax(probs)]
    })
```

### Monitoring in Production

Track these metrics in production:
- Prediction latency (p50, p99)
- Class distribution drift
- Calibration over time
- User feedback correlation
