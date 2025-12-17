# Manifold Transformations: A Geometric Deep Learning Perspective

## The Big Picture

From a geometric deep learning perspective, HRM performs a fundamentally different transformation of the data manifold compared to Transformers:

- **Transformers**: Fixed-depth diffeomorphism, mapping input manifold to output manifold in exactly L steps
- **HRM**: Adaptive-depth flow, allowing the manifold to evolve for as long as needed to reach a "solution manifold"

The key insight: **HRM creates a dimensionality hierarchy** that mirrors the biological brain - higher modules operate in higher-dimensional spaces for flexibility.

---

## First Principles: Data Manifolds and Neural Networks

### Part 1: What Is a Data Manifold?

A **manifold** is a smooth surface that locally looks like flat Euclidean space, but globally can have complex structure.

#### Concrete Example: The Earth

The Earth's surface is a 2D manifold embedded in 3D space:
- Locally: Any small patch looks flat (like a map)
- Globally: It's a sphere (you can't flatten it without distortion)

#### Data Manifold Intuition

High-dimensional data (like images or text) typically lies on a lower-dimensional manifold:

```
Image space: 256×256×3 = 196,608 dimensions

But natural images don't fill this space uniformly!
- Random pixels → noise (not on manifold)
- Cat photos → clustered on a "cat manifold"
- Text → clustered on a "text manifold"
```

**The manifold hypothesis**: Real data lies on or near a low-dimensional manifold within the high-dimensional ambient space.

### Part 2: Neural Networks as Manifold Transformations

Each neural network layer transforms the data manifold:

```
Input manifold M₀
        ↓ Layer 1
Transformed manifold M₁
        ↓ Layer 2
Transformed manifold M₂
        ↓ ...
        ↓ Layer L
Output manifold M_L
```

**Goal of learning**: Find transformations that:
1. Separate different classes (disentangle)
2. Group similar points together (clustering)
3. Make the prediction task easy on the final manifold

### Part 3: Transformers as Fixed Diffeomorphisms

A Transformer with L layers applies L diffeomorphisms (smooth, invertible maps):

```
M₀ →[f₁]→ M₁ →[f₂]→ M₂ → ... →[f_L]→ M_L
```

**The limitation**: The "amount of transformation" is fixed. Simple and complex inputs both get exactly L transformations.

**Geometric analogy**: It's like having exactly L "sculpting operations" to shape clay. Sometimes L is enough, sometimes it's not.

---

## How HRM Transforms the Manifold

### The Hierarchical Flow

HRM creates a different kind of transformation - an **iterative flow** that can continue until convergence:

```
Input manifold M₀
        ↓
    ┌─────────────────────────────────┐
    │  Cycle 1:                       │
    │  M₀ →[L]→ M₁ →[L]→ M₂ →[L]→ M₃  │
    │         (T steps, H fixed)       │
    │             ↓ H updates          │
    │  M₃ becomes input for Cycle 2    │
    └─────────────────────────────────┘
        ↓
    ┌─────────────────────────────────┐
    │  Cycle 2:                       │
    │  M₃ →[L]→ M₄ →[L]→ M₅ →[L]→ M₆  │
    │         (T steps, new H)         │
    │             ↓ H updates          │
    │  M₆ becomes input for Cycle 3    │
    └─────────────────────────────────┘
        ↓
       ...continues until ACT halts
```

### Key Difference: Variable Depth

- **Easy problem**: ACT halts after 1-2 segments. Manifold transformed minimally.
- **Hard problem**: ACT continues for 8+ segments. Manifold transformed extensively.

**Geometric analogy**: Instead of a fixed number of sculpting operations, you sculpt until the shape is "correct" (verified by ACT).

---

## The Dimensionality Hierarchy

### What the Paper Discovered

The paper's most striking result is the **emergent dimensionality hierarchy** in trained HRM:

```
Participation Ratio (PR):
- L-module (z_L): PR ≈ 30
- H-module (z_H): PR ≈ 90

Ratio: z_H/z_L ≈ 3.0
```

This matches the biological brain! In mouse cortex:
```
Higher cortical areas (associative): High PR
Lower cortical areas (sensory): Low PR

Ratio ≈ 2.25
```

### What Is Participation Ratio?

**Participation Ratio (PR)** measures the "effective dimensionality" of a representation:

```
PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²

Where λᵢ are eigenvalues of the covariance matrix
```

But what does this actually mean? Let's build it from first principles.

#### Step 1: What Are We Measuring?

We want to know: **How many dimensions does the neural network actually USE?**

A network might have 512-dimensional hidden states, but if 500 of those dimensions are always near zero, it's really only using ~12 dimensions.

#### Step 2: Covariance Matrix of WHAT?

The covariance matrix is computed over **the collection of hidden states the network produces**.

**Concretely for HRM:**

```
Run the model on many different inputs (e.g., 1000 Sudoku puzzles)
At each timestep, record the hidden state z (a 512-dim vector, say)

For L-module, collect: z_L after step 1, step 2, ..., for all puzzles
For H-module, collect: z_H after cycle 1, cycle 2, ..., for all puzzles

This gives you a matrix of hidden states:

Z = [ z₁ ]      ← hidden state from puzzle 1, step 1
    [ z₂ ]      ← hidden state from puzzle 1, step 2
    [ z₃ ]      ← hidden state from puzzle 2, step 1
    [ ... ]
    [ zₙ ]      ← hidden state from puzzle 1000, final step

Z has shape: (n_samples × hidden_dim), e.g., (50000 × 512)
```

**The covariance matrix Σ** measures how the dimensions vary together:

```
Σ = (1/n) Σᵢ (zᵢ - μ)(zᵢ - μ)ᵀ

Where μ = mean of all hidden states

Σ has shape: (hidden_dim × hidden_dim), e.g., (512 × 512)

Σ[j,k] = "How much do dimensions j and k vary together?"
```

#### Step 3: Eigenvalues Tell You About Spread

The eigenvalues λ₁, λ₂, ..., λ_d of the covariance matrix tell you **how much variance exists along each principal direction**:

```
Principal Component Analysis (PCA) view:

λ₁ = variance along the "most important" direction
λ₂ = variance along the "second most important" direction
...
λ_d = variance along the "least important" direction

If all λᵢ are equal: data spreads equally in all directions (sphere)
If λ₁ >> λ₂ >> ... : data is concentrated along first few directions (ellipse/tube)
```

#### Step 4: Participation Ratio Formula

```
PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²
```

**Why this formula?**

Case 1: All variance in ONE dimension
```
λ = [1, 0, 0, ..., 0]

PR = (1)² / (1²) = 1

"Only 1 dimension participates"
```

Case 2: Variance EQUALLY spread across d dimensions
```
λ = [1/d, 1/d, 1/d, ..., 1/d]   (d equal values)

PR = (d × 1/d)² / (d × (1/d)²)
   = 1² / (d × 1/d²)
   = 1 / (1/d)
   = d

"All d dimensions participate equally"
```

Case 3: Variance in k dimensions, zero in others
```
λ = [1/k, 1/k, ..., 1/k, 0, 0, ..., 0]   (k nonzero, d-k zero)

PR = (k × 1/k)² / (k × (1/k)²)
   = 1 / (1/k)
   = k

"k dimensions participate"
```

**PR is the "effective number of dimensions" the representation uses.**

#### Intuition Summary

**Low PR** (e.g., 1-10):
- Variance concentrated in few dimensions
- Representation is "compressed"
- Like a thin tube in high-dimensional space
- The network has learned a specialized, efficient encoding

**High PR** (e.g., 50-100):
- Variance spread across many dimensions
- Representation is "expansive"
- Like a fat blob in high-dimensional space
- The network uses many dimensions flexibly

### Why Dimensionality Matters for Reasoning

**Higher dimensionality = More flexibility = More tasks can be represented**

The prefrontal cortex (reasoning center) has very high-dimensional representations because it needs to:
- Handle many different tasks
- Represent abstract relationships
- Maintain flexible working memory

**Lower dimensionality = More specialized = Better at specific tasks**

Sensory cortex has lower-dimensional representations because it:
- Processes specific types of input
- Extracts particular features
- Doesn't need multi-task flexibility

---

## HRM's Emergent Hierarchy

### The Observation

After training, HRM develops this structure naturally:

```
┌─────────────────────────────────────┐
│         H-Module (z_H)              │
│                                     │
│  ●  ●  ●  ●  ●  ●  ●  ●            │
│    ●  ●  ●  ●  ●  ●  ●             │  High-dimensional
│  ●  ●  ●  ●  ●  ●  ●  ●            │  (PR ≈ 90)
│    ●  ●  ●  ●  ●  ●  ●             │  Many active dimensions
│  ●  ●  ●  ●  ●  ●  ●  ●            │
│                                     │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│         L-Module (z_L)              │
│                                     │
│      ●●●●                           │  Low-dimensional
│      ●●●●                           │  (PR ≈ 30)
│      ●●●●                           │  Fewer active dimensions
│                                     │
└─────────────────────────────────────┘
```

### Why This Emerges

**H-module needs high dimensionality because**:
- It must encode different "strategies" for different puzzles
- It represents abstract state (what plan we're following)
- It must flexibly influence L-module in many different ways

**L-module needs lower dimensionality because**:
- It executes specific operations within a strategy
- It converges to local equilibria (which are low-dimensional attractors)
- It processes detailed constraints (more specialized)

### Contrast: Neural Collapse in Standard Networks

Conventional deep networks often exhibit **neural collapse**:
- Final layer representations collapse to class means
- Dimensionality decreases toward the output
- The representation becomes "dead" - minimal information beyond class label

**HRM avoids neural collapse** because:
- The H-module must remain expressive to guide L-module
- ACT requires rich state to decide halt/continue
- Hierarchical structure maintains diversity

---

## Geometric Interpretation: The Solution Manifold

### The Reasoning Task as Geometry

Consider a Sudoku puzzle:

**Input space**: 81-dimensional (one dimension per cell, values 0-9)
**Solution space**: A discrete subset (exactly one correct solution)

But in the embedding space, this becomes:

```
All possible Sudoku states (continuous embedding)
│
├── Invalid states (constraint violations)
│       ↓ far from solution manifold
│
├── Partially valid states (some constraints met)
│       ↓ approaching solution manifold
│
└── Valid solutions (all constraints met)
        ↓ ON the solution manifold
```

### How HRM Navigates This

**L-module**: Moves toward the solution manifold (given current strategy from H)

```
z_L trajectory within one cycle:

    z_L⁰  →  z_L¹  →  z_L²  →  z_L*

    ●                          ●
     ↘                        ↗
       ↘                    ↗
         →→→→→→→→→→→→→→→→→

    Start        (gradient flow toward local equilibrium)        Converged
```

**H-module**: Adjusts which part of the solution manifold L-module targets

```
z_H trajectory across cycles:

    z_H⁰  →  z_H¹  →  z_H²  →  ...

    Different equilibria targeted by L-module at each H-state:

    z_H⁰ → L converges to z_L*(0)  (trying strategy A)
    z_H¹ → L converges to z_L*(1)  (refining based on A's result)
    z_H² → L converges to z_L*(2)  (backtracking if needed)
```

### The Key Geometric Insight

**Standard networks**: Fixed mapping from input to output. Either the mapping hits the solution manifold or it doesn't.

**HRM**: Iterative search toward the solution manifold. The hierarchy allows:
1. **Local search** (L-module): Move toward nearest valid state
2. **Global guidance** (H-module): Adjust which "region" of solution space to target

This is like having both:
- A local optimizer (gradient descent toward nearby solutions)
- A meta-optimizer (adjusting which basin to descend into)

---

## Participation Ratio: Detailed Analysis

### How PR Is Calculated

Given neural trajectories from solving multiple problems:

```
1. Collect hidden states: Z = [z₁, z₂, ..., z_n]  (across tasks and timesteps)
2. Compute covariance: Σ = (1/n) Σᵢ (zᵢ - μ)(zᵢ - μ)ᵀ
3. Compute eigenvalues: λ₁ ≥ λ₂ ≥ ... ≥ λ_d
4. Calculate PR: (Σᵢ λᵢ)² / Σᵢ λᵢ²
```

### What PR Values Mean

**PR = 1**: All variance in one dimension (maximally compressed)
```
Data looks like:  ────────●────────  (a line)
```

**PR = d** (hidden dimension): Variance equally spread (maximally expanded)
```
Data looks like:  ●●●●●●●●●●●●●●●  (fills the space)
```

**PR = 30** (L-module):
```
Data occupies ~30 effective dimensions
Specialized, task-specific processing
```

**PR = 90** (H-module):
```
Data occupies ~90 effective dimensions
Flexible, multi-task representation
```

### The Scaling Phenomenon

The paper shows (Figure 8c) that H-module PR **scales with task diversity**:

```
Tasks   H-module PR   L-module PR
10      40            25
50      70            28
100     90            30
```

**H-module adapts** to use more dimensions when facing more diverse tasks.
**L-module stays stable** - it's doing similar operations regardless of task variety.

This is exactly what neuroscience predicts about cortical hierarchy!

---

## Brain Correspondence: The Empirical Evidence

### Mouse Cortex Data (Posani et al.)

The paper compares HRM to mouse brain recordings:

```
Brain Region          Position in Hierarchy    PR
──────────────────────────────────────────────────
SSp-n (sensory)       Low (1)                  ~2.0
VISp (visual)         Low-mid (5)              ~2.5
MOs (motor planning)  Mid-high (25)            ~4.0
ACAd (associative)    High (40)                ~5.0
```

**Correlation**: ρ = 0.79 (strong positive)

Higher in the hierarchy → Higher PR → More flexible representation

### HRM Matches This

```
Module        Position in Hierarchy    PR
──────────────────────────────────────────────────
L-module      Low (execution)          ~30
H-module      High (planning)          ~90

Ratio: 90/30 = 3.0
Brain ratio: ~5.0/2.0 = 2.5
```

The ratios are remarkably similar!

### Control: Untrained Network

To verify this is learned (not architectural), the paper tests untrained HRM:

```
Untrained Network:
L-module PR ≈ 42
H-module PR ≈ 40

No hierarchy! Both modules have similar dimensionality.
```

**The dimensionality hierarchy emerges through training** - it's not an artifact of architecture.

---

## Geometric Interpretation of Intermediate States

### Visualizing the Solution Process

The paper shows (Figure 7) how predictions evolve over timesteps:

**Maze-Hard**:
```
Step 0: Multiple tentative paths
Step 2: Some paths eliminated
Step 4: Rough outline of solution
Step 6: Complete optimal path
```

**Sudoku-Extreme**:
```
Step 0: Random-ish fills
Step 2: Some constraints applied
Step 4: Partial solution (conflicts visible in red)
Step 6: Backtracking (grey cells = changed)
Step 8: Correct solution
```

**ARC-AGI**:
```
Step 0: Raw transformation attempt
Step 2-4: Incremental refinement
Step 6: Converged to correct pattern
```

### Geometric View

Each step moves on the data manifold:

```
Step 0      Step 2      Step 4      Step 6
   ●           ●           ●           ★
    ↘         ↙↗          ↗
      ↘     ↗            ↗
        ↘↗              ↗
         ↗↘            ↗
       ↗    ↘        ↗
     ●        →→→→→→★

Initial    Exploring   Refining   Solution!
state      (possibly   (converging
           wrong       toward
           direction)  solution)
```

The H-module enables "course corrections" - changing direction when a strategy isn't working.

---

## Implications for Geometric Deep Learning

### Traditional View: Networks as Functions

```
f: Input Space → Output Space
```

Fixed transformation, applied uniformly.

### HRM View: Networks as Dynamical Systems

```
dz/dt = F(z, x)

Evolution continues until equilibrium or halting criterion
```

**Variable transformation**, adapted to input complexity.

### The Geometric Advantage

**Hard problems** occupy "further" regions of the manifold from the solution:
- Need more transformation to reach solution manifold
- HRM can continue iterating
- Transformer is stuck with fixed depth

**Easy problems** are "close" to solution manifold:
- Minimal transformation needed
- HRM halts early (saves compute)
- Transformer wastes capacity

---

## Summary: Manifold Perspective on HRM

| Aspect | Transformer | HRM |
|--------|-------------|-----|
| Transformation | Fixed L diffeomorphisms | Adaptive flow to equilibrium |
| Depth | Fixed: L | Variable: N×T×M |
| Manifold navigation | Direct mapping | Iterative search |
| Dimensionality | Collapses toward output | Hierarchical (H > L) |
| Biological match | Poor | Strong (matches cortex) |
| Hard problems | Same transform as easy | More iterations |

**The key geometric insight**: HRM creates a hierarchical flow on the data manifold, with:
- High-dimensional planning space (H-module) for flexibility
- Low-dimensional execution space (L-module) for stability
- Adaptive duration to reach the solution manifold

This matches how the brain organizes computation - and explains why HRM succeeds where fixed-depth models fail.

---

## Concrete Example: Text Completion on the Manifold

Let's visualize what the L-module and H-module are doing on the data manifold using a text completion task. This makes the abstract geometry concrete.

### The Task

Complete the sentence: "The chef carefully _____ the ingredients before _____"

### The Embedding Space View

Each partial completion lives as a point in high-dimensional embedding space. Let's project down to 2D for visualization:

```
                          EMBEDDING SPACE (simplified to 2D)

    "cooked"●                                    ●"assembled"
              ↖                                ↗
                ↖                            ↗
    "prepared"●───↖──────────────────────↗───●"mixed"
                    ↖                  ↗
                      ↖    ●START    ↗
                        ↖   ↓      ↗
                          ↘↓    ↗
                           ●●●●
                          "chopped"
                          "sliced"
                          "diced"
                          "minced"
                          (cluster of similar meanings)
```

### What L-Module Does: Local Manifold Movement

The L-module moves the representation **locally** toward coherent completions:

```
CYCLE 1, L-module steps (H says: "food preparation context"):

Step 1: START → moves toward cooking verbs region
        z_L: [0.5, 0.3] → [0.4, 0.5]

Step 2: Refines toward "cutting" subregion (based on "carefully")
        z_L: [0.4, 0.5] → [0.3, 0.6]

Step 3: Converges to "chopped" vicinity
        z_L: [0.3, 0.6] → [0.25, 0.65]

Step 4: Equilibrium reached
        z_L: [0.25, 0.65] → [0.25, 0.65]  (no more movement)

L-module converged! It found a local attractor: the "precise cutting" region
```

**Geometrically**: L-module followed the gradient on the manifold toward the nearest coherent completion, given H's context.

### What H-Module Does: Global Manifold Steering

The H-module **changes which region** of the manifold L-module targets:

```
CYCLE 1 RESULT:
L converged to "chopped" region
H observes: "Good, but what comes after chopping?"

H-module update:
z_H: "food prep" → "food prep + sequence awareness"

CYCLE 2, L-module steps (H now says: "need temporal sequence"):

Step 1: L restarts from "chopped" but now seeks what comes AFTER
        z_L: [0.25, 0.65] → [0.3, 0.7]  (moving toward "cooking" region)

Step 2: Refines toward cooking actions
        z_L: [0.3, 0.7] → [0.4, 0.75]

Step 3: Converges to "cooking" vicinity
        z_L: [0.4, 0.75] → [0.45, 0.78]

Step 4: Equilibrium reached
        z_L: [0.45, 0.78] → [0.45, 0.78]

L-module converged to a DIFFERENT equilibrium because H changed!
```

**Geometrically**: H-module shifted the "target basin" on the manifold. L-module descended into a different valley.

### The Full Picture

```
                     MANIFOLD TRAJECTORY

    "cooked"●
              ↖
                ↖  ←←←←←←←←←←←←←←←←←←←←●z_H² (sequence context)
    "prepared"●───↖──────────────↗↗↗↗↗↗↗↗
                    ↖          ↗
                      ↖●START↗
                        ↘↓  ↗
                         ↘↓↗
                          ●z_L* (Cycle 1: "chopped")
                          ↓
                          ↓ (H updates!)
                          ↓
                          ●───→→→●z_L** (Cycle 2: "cooking")

Cycle 1: L descends to "chopped" equilibrium
H updates: "Now think about what comes next"
Cycle 2: L descends to "cooking" equilibrium (DIFFERENT basin!)

Final output: "The chef carefully chopped the ingredients before cooking"
```

### Why This Requires Hierarchy

**Without H-module** (simple iteration):
```
L keeps refining "chopped" → "sliced" → "diced" → "minced"
Gets stuck in the "cutting verbs" basin
Never realizes it needs to think about "what comes after"

Output: "The chef carefully chopped the ingredients before chopping"
        (Incoherent! Just repeated the same concept)
```

**With H-module**:
```
L converges to "chopped"
H sees this and updates context: "OK, first blank filled, now second blank"
L reconverges to "cooking" (different basin!)

Output: "The chef carefully chopped the ingredients before cooking"
        (Coherent! Different concepts for different blanks)
```

---

## Concrete Example: Cricket Ball Sequence on the Manifold

Now let's see the same dynamics with cricket ball-by-ball prediction.

### The Task

Predict outcome of ball 47.5 in a T20 chase:
- Score: 142/4, Target: 165, Need 23 from 15 balls
- Batsman A: 45(32), set and aggressive
- Batsman B: 8(5), newer but capable
- Bowler: Death specialist, bowling yorkers

### The Cricket State Space

Each match state is a point in embedding space. Different regions represent different "types" of situations:

```
                    CRICKET STATE MANIFOLD (2D projection)

        "Desperate slog"●                    ●"Cruise control"
                          ↖                ↗
                            ↖            ↗
        "Calculated risk"●────↖────────↗────●"Consolidation"
                                ↖    ↗
                        Need:   ●●●●
                        23 from 15
                        (we are HERE)
                                ↓
                "Tight finish"●●●●●●●●●●●●●●●●
                (cluster of similar pressure situations)
```

### What L-Module Does: Tactical Analysis

L-module analyzes the SPECIFIC delivery:

```
CYCLE 1, L-module steps (H says: "tight chase, maintain balance"):

Step 1: Process match state
        z_L encodes: "Need 1.53 RPB, 4 wickets in hand"
        Moves toward "achievable but not easy" region

Step 2: Process batsman matchup
        z_L refines: "Set batsman vs death bowler"
        Moves toward "contest is even" region

Step 3: Process delivery expectations
        z_L refines: "Yorker likely, boundary difficult"
        Moves toward "rotate strike or defend" region

Step 4: Converge to tactical assessment
        z_L*: "Single most likely, boundary possible, wicket risk moderate"

L-module equilibrium: P(0)=25%, P(1)=40%, P(2)=10%, P(4)=15%, P(W)=10%
```

**Geometrically**: L-module descended through the "tactical execution" manifold to find the most likely outcomes given current context.

### What H-Module Does: Strategic Adjustment

H-module considers the BIGGER picture:

```
After CYCLE 1:
L found: "Singles are safe, boundaries risky but possible"

H-module observes this AND considers strategic implications:

H update computation:
- "23 from 15 at 1 run per ball = 8 runs short"
- "Need at least 2 boundaries in remaining balls"
- "But losing wicket now would be catastrophic"
- "Batsman A is set - he should take risks, not B"

z_H update: "balanced" → "selective aggression from set batsman"

CYCLE 2, L-module steps (H now says: "set batsman can attack"):

Step 1: Reprocess with aggression bias
        z_L shifts toward "attacking outcomes more likely"

Step 2: BUT also incorporate yorker difficulty
        z_L refines: "attack if overpitched, defend if perfect"

Step 3: Integrate experience (set batsman handles pressure)
        z_L refines: "experience reduces wicket probability"

Step 4: New equilibrium
        z_L**: "Attack with controlled risk"

L-module NEW equilibrium: P(0)=20%, P(1)=35%, P(2)=12%, P(4)=25%, P(W)=8%
```

**Key change**: Boundary probability UP (15%→25%), wicket probability DOWN (10%→8%)

H-module didn't change the FACTS - same match state. It changed the INTERPRETATION: "this batsman can afford to attack."

### Visualizing the Cricket Manifold Trajectory

```
                    CRICKET MANIFOLD TRAJECTORY

    "All-out attack"●
                      ↖
                        ↖
    "Selective attack"●───●z_H² (strategic shift)
                          ↑
                          ↑ (H updates strategy!)
                          ↑
    "Conservative"●───────●z_H¹ (initial assessment)
                          ↓
                          ↓
                START●    ↓
                     ↘    ↓
                       ↘  ↓
                         ●z_L* (Cycle 1: balanced tactics)
                         ↓
                         ↓ (H says: "set batsman, attack")
                         ↓
                         ●→→→→●z_L** (Cycle 2: aggressive tactics)

z_L* prediction:  P(4)=15%, P(W)=10%   (conservative)
z_L** prediction: P(4)=25%, P(W)=8%    (aggressive for set batsman)
```

### The Dimensionality Story in Cricket

**L-module (Low PR ≈ 30)** represents:
- Specific tactical situations
- Ball-by-ball probabilities
- Immediate delivery analysis
- Narrow, specialized processing

```
L-module dimensions might encode:
- Dimension 1-5: Run probabilities (0,1,2,4,6)
- Dimension 6-10: Wicket probabilities by type
- Dimension 11-15: Shot type likelihoods
- Dimension 16-20: Field position relevance
- Dimension 21-30: Bowler-batsman matchup features

Specialized! Each dimension has clear tactical meaning.
```

**H-module (High PR ≈ 90)** represents:
- Strategic phase (powerplay, middle, death)
- Risk tolerance (ahead/behind, wickets in hand)
- Momentum and pressure
- Long-range planning
- FLEXIBLE, abstract concepts

```
H-module dimensions might encode:
- Some dimensions: "Desperation level" (many dimensions, continuous)
- Some dimensions: "Batsman confidence" (many dimensions, contextual)
- Some dimensions: "Match turning point proximity"
- Some dimensions: "Historical similar situations"
- Some dimensions: "Team tendencies under pressure"
- Many more dimensions: Abstract strategic concepts we can't name!

Flexible! Uses many dimensions because strategy is complex and varied.
```

### Why Cricket NEEDS the Hierarchy

**Ball 47.5 vs Ball 1.1:**

```
Ball 1.1 (opening ball of innings):
- Context is generic
- No pressure yet
- Historical averages dominate
- L-module: standard "first ball" equilibrium
- H-module: generic "opening phase" strategy
- ACT: halt after 1-2 segments (routine)

Ball 47.5 (tight finish):
- Context is highly specific
- Extreme pressure
- Every factor matters
- L-module: needs multiple cycles to integrate all factors
- H-module: continuously adjusting strategy as L provides tactical info
- ACT: continue for 6+ segments (crucial decision)
```

**The hierarchy enables different AMOUNTS of reasoning for different situations** - exactly what cricket requires.

### Summary: What Each Module Does to the Manifold

| Module | Manifold Action | Cricket Analog |
|--------|-----------------|----------------|
| L-module | Descends toward local equilibrium | Analyzes THIS ball's tactics |
| H-module | Shifts which equilibrium L targets | Adjusts overall strategy |
| L convergence | Finds coherent tactical assessment | "Given strategy X, outcomes are Y" |
| H update | Changes strategic context | "Actually, we should be more aggressive" |
| Multiple cycles | Explores different strategy-tactic combos | "Try defensive... no, try attacking" |

**The manifold view**: L-module does local gradient descent on the "coherent prediction" manifold. H-module moves the manifold itself (changes the landscape L descends on). Together, they navigate toward the "correct prediction" region through hierarchical search.
