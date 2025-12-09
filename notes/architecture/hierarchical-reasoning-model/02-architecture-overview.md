# Architecture Overview: The Two-Module Brain

## The Big Picture

HRM has four learnable components:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Input     │     │  Low-Level  │     │ High-Level  │     │    Output    │
│  Network    │ ──► │   Module    │ ◄─► │   Module    │ ──► │    Head      │
│    f_I      │     │    f_L      │     │    f_H      │     │     f_O      │
└─────────────┘     └─────────────┘     └─────────────┘     └──────────────┘
     θ_I                 θ_L                 θ_H                  θ_O
```

**Key insight**: The magic happens in the interaction between f_L (fast, detailed) and f_H (slow, abstract).

---

## Abstraction Ladder

### L1: Abstract Principle
> **Effective reasoning requires both high-level strategy and low-level execution, operating at different speeds.**

This mirrors the brain: prefrontal cortex (slow planning) guides motor cortex (fast execution).

### L2: Framework Level
Two coupled recurrent modules:
- **H-module**: Updates slowly (every T steps), maintains abstract plan
- **L-module**: Updates rapidly (every step), executes detailed computation

### L3: Method Level
Within each "cycle":
1. L-module runs T steps with fixed H-state context
2. L-module converges toward a local equilibrium
3. H-module updates once using L's final state
4. L-module resets for a new computational phase

### L4: Component Level
Both modules are **encoder-only Transformers** with modern enhancements:
- Rotary Positional Encoding (RoPE)
- Gated Linear Units (GLU)
- RMSNorm (not LayerNorm)
- No bias terms
- Post-Norm architecture

(See detailed explanation of each enhancement below)

### L5: Concrete Implementation

```python
def hrm(z, x, N=2, T=2):
    x = input_embedding(x)
    zH, zL = z

    with torch.no_grad():  # No gradients through unrolling
        for _i in range(N * T - 1):
            zL = L_net(zL, zH, x)
            if (_i + 1) % T == 0:
                zH = H_net(zH, zL)

    # Only last step gets gradients (one-step approximation)
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)

    return (zH, zL), output_head(zH)
```

---

## What is an Encoder-Only Transformer?

### Background: Encoder vs Decoder

The original Transformer (Vaswani et al., 2017) had two parts:

```
ENCODER-DECODER TRANSFORMER (for translation):

Encoder: Processes input sequence (e.g., French sentence)
         - Each position attends to ALL other positions (bidirectional)

Decoder: Generates output sequence (e.g., English sentence)
         - Each position attends only to PREVIOUS positions (causal/masked)
         - Also attends to encoder output (cross-attention)
```

**Encoder-only** means: Just the encoder part, no decoder, no causal masking.

```
ENCODER-ONLY (like BERT, and HRM's modules):

Input: [x₁, x₂, x₃, ..., xₙ]

Every position attends to EVERY other position (including future):

Position 3 can see: x₁, x₂, x₃, x₄, ..., xₙ (all of them!)
```

### Why Encoder-Only for HRM?

HRM processes **complete grids** (Sudoku, ARC, Maze), not sequential generation:

```
Sudoku input (flattened 9×9 = 81 tokens):

[5, 3, 0, 0, 7, 0, 0, 0, 0,   ← Row 1
 6, 0, 0, 1, 9, 5, 0, 0, 0,   ← Row 2
 0, 9, 8, 0, 0, 0, 0, 6, 0,   ← Row 3
 ...]

Cell at position 45 needs to "see" cells at positions 1, 9, 36, 72...
(same row, column, and box constraints)

Encoder-only: Every cell attends to every other cell ✓
Decoder-only: Cell 45 could only see cells 1-44 ✗
```

### Cricket Example: Encoder-Only vs Decoder-Only

```
Cricket balls in an over (6 balls):

Input: [ball₁, ball₂, ball₃, ball₄, ball₅, ball₆]

ENCODER-ONLY attention:
- ball₆ can see balls 1-5 (what happened before)
- ball₁ can see balls 2-6 (what happened after) ← useful for analysis!
- ball₃ can see both directions

Good for: ANALYZING a completed over, understanding patterns

DECODER-ONLY (causal) attention:
- ball₆ can see balls 1-5 only
- ball₁ can see nothing before it
- ball₃ can see balls 1-2 only

Good for: PREDICTING the next ball (can't peek at future!)
```

### What an Encoder-Only Transformer Block Looks Like

```
Input: z (sequence of vectors, shape: [seq_len, hidden_dim])
       │
       ▼
┌──────────────────────────────────────┐
│  Multi-Head Self-Attention           │
│  (every position attends to every    │
│   other position - no masking)       │
└──────────────────────────────────────┘
       │
       ▼
    Add & Norm (residual connection + normalization)
       │
       ▼
┌──────────────────────────────────────┐
│  Feed-Forward Network                │
│  (applied independently to each      │
│   position)                          │
└──────────────────────────────────────┘
       │
       ▼
    Add & Norm (residual connection + normalization)
       │
       ▼
Output: z' (same shape as input)
```

---

## The Modern Enhancements Explained

### 1. RoPE (Rotary Positional Encoding)

**The problem**: Attention has no inherent notion of position. "cat sat mat" and "mat sat cat" produce the same attention patterns without positional information.

**Old solution (original Transformer)**: Add fixed sinusoidal vectors to input:
```
x_i' = x_i + PE(i)

Where PE(i) is a fixed pattern based on position i
```

**RoPE solution**: Instead of adding position info, **rotate** the query/key vectors:

```
Traditional: q_i · k_j  (dot product ignores positions)

RoPE: rotate(q_i, i) · rotate(k_j, j)

The rotation is designed so that:
rotate(q, i) · rotate(k, j) depends on (i - j)

This means attention naturally captures RELATIVE position!
```

**Why RoPE is better**:
- Relative positions are often more important than absolute
- "word 5 is 3 positions before word 8" matters more than "word 5 is at position 5"
- Generalizes better to different sequence lengths

### 2. GLU (Gated Linear Units)

**The problem**: Standard feed-forward networks apply a fixed transformation:
```
FFN(x) = W₂ · ReLU(W₁ · x)

Every input gets transformed the same way (just scaled by ReLU)
```

**GLU solution**: Let the network **choose** how much to transform:

```
GLU(x) = (W₁ · x) ⊙ σ(W₂ · x)
              │         │
         "content"   "gate" (0 to 1)

⊙ = element-wise multiplication
σ = sigmoid (squashes to 0-1)
```

**What this does**:
- The "gate" decides which parts of the transformation to keep
- Some dimensions might be gated to 0 (ignored)
- Others gated to 1 (fully used)
- The network learns WHEN to apply WHICH transformations

**Analogy**: Instead of "always apply this filter," it's "decide whether to apply this filter based on what you see."

### 3. RMSNorm (vs LayerNorm)

**LayerNorm** (original Transformer):
```
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β

1. Subtract mean (center the data)
2. Divide by standard deviation (normalize scale)
3. Apply learnable scale (γ) and shift (β)
```

**RMSNorm** (simpler, faster):
```
RMSNorm(x) = x / RMS(x) * γ

Where RMS(x) = √(mean(x²))

1. Divide by root-mean-square (normalize scale only)
2. Apply learnable scale (γ)
3. NO mean subtraction, NO shift
```

**Why RMSNorm**:
- Empirically works just as well
- Computationally cheaper (no mean calculation)
- Fewer parameters (no β)
- Used in LLaMA and other modern architectures

### 4. No Bias Terms

**Standard linear layer**:
```
y = Wx + b

W = weight matrix
b = bias vector (added to every output)
```

**Without bias**:
```
y = Wx

No bias term
```

**Why remove bias**:
- Modern architectures find it unnecessary when using normalization
- Slightly fewer parameters
- RMSNorm can compensate for any needed shifts
- Empirically doesn't hurt performance

### 5. Post-Norm Architecture

**Pre-Norm** (common in recent models like GPT):
```
x → Norm → Attention → Add(x) → Norm → FFN → Add(x) → output

Normalize BEFORE each sublayer
```

**Post-Norm** (original Transformer, and HRM):
```
x → Attention → Add(x) → Norm → FFN → Add(x) → Norm → output

Normalize AFTER each sublayer
```

**Why HRM uses Post-Norm**:
- Better for recurrent/iterative computation
- Helps with the equilibrium convergence (one-step gradient approximation)
- More stable for very deep effective computation
- Aligns with Deep Equilibrium Model theory

---

## First Principles: Understanding the Hierarchy

### Part 1: Why Two Modules?

Consider how you solve a Sudoku puzzle:

**High-level (strategic)**:
- "I should focus on this 3×3 block first"
- "This row is almost complete"
- "Let me try putting 5 here and see what happens"

**Low-level (tactical)**:
- "Row 1 already has 1,3,5,7,9 so this cell can only be 2,4,6,8"
- "If I put 5 here, then column 3 can't have another 5"
- "Checking all constraints... no conflicts"

The strategic thinking changes slowly (you stick with a plan for a while). The tactical checking is rapid (many quick constraint evaluations).

**HRM separates these two types of computation.**

### Part 2: The Timescale Separation

Brain oscillations provide the biological analogy:

| Brain Rhythm | Frequency | Function | HRM Analog |
|-------------|-----------|----------|------------|
| Theta waves | 4-8 Hz | Working memory, planning | H-module |
| Gamma waves | 30-100 Hz | Perception, detailed processing | L-module |

Theta is ~10× slower than gamma. In HRM, H-module updates every T steps (typically T=8), making it ~8× slower than L-module.

### Part 3: Information Flow

```
One Forward Pass (N=2 cycles, T=4 steps per cycle):
═══════════════════════════════════════════════════

Cycle 1:
─────────────────────────────────────────────────
Step 1:  z_L = f_L(z_L⁰, z_H⁰, x̃)     z_H stays fixed
Step 2:  z_L = f_L(z_L¹, z_H⁰, x̃)     z_H stays fixed
Step 3:  z_L = f_L(z_L², z_H⁰, x̃)     z_H stays fixed
Step 4:  z_L = f_L(z_L³, z_H⁰, x̃)     z_H = f_H(z_H⁰, z_L⁴)  ← H updates!

Cycle 2:
─────────────────────────────────────────────────
Step 5:  z_L = f_L(z_L⁴, z_H¹, x̃)     z_H stays fixed (new value)
Step 6:  z_L = f_L(z_L⁵, z_H¹, x̃)     z_H stays fixed
Step 7:  z_L = f_L(z_L⁶, z_H¹, x̃)     z_H stays fixed
Step 8:  z_L = f_L(z_L⁷, z_H¹, x̃)     z_H = f_H(z_H¹, z_L⁸)  ← H updates!

Output: ŷ = f_O(z_H²)
```

**Total computational depth**: N × T = 2 × 4 = 8 steps

But this is just ONE forward pass. With deep supervision, HRM can do M forward passes (segments), giving total depth N × T × M.

---

## The Four Components in Detail

### 1. Input Network (f_I)

**Purpose**: Convert discrete tokens to vector representations.

```
x = (x₁, x₂, ..., x_l)  →  x̃ = f_I(x; θ_I)

Where:
- x_i ∈ {1, 2, ..., V} are token indices (vocabulary V)
- x̃ ∈ R^{l × d} is a sequence of d-dimensional vectors
```

For Sudoku/ARC: This is a simple embedding layer, converting grid cells to vectors.

### 2. Low-Level Module (f_L)

**Purpose**: Fast, detailed computation. Updates every timestep.

**Inputs**:
- z_L^{i-1}: Previous L-state (its own memory)
- z_H^{i-1}: Current H-state (context/guidance from above)
- x̃: Input representation

**Architecture**: Encoder-only Transformer block

```
z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x̃; θ_L)

Implementation:
combined = z_L^{i-1} + z_H^{i-1} + x̃  # Element-wise addition
z_L^i = TransformerBlock(combined)
```

**Key property**: Updates rapidly, converges within T steps to a "local equilibrium" given fixed z_H.

### 3. High-Level Module (f_H)

**Purpose**: Slow, abstract planning. Updates every T steps.

**Inputs**:
- z_H^{i-1}: Previous H-state (its own memory)
- z_L^{i-1}: Final L-state from the completed cycle

**Architecture**: Encoder-only Transformer block

```
z_H^i = f_H(z_H^{i-1}, z_L^{i-1}; θ_H)  if i ≡ 0 (mod T)
      = z_H^{i-1}                        otherwise

Implementation:
combined = z_H^{i-1} + z_L^{i-1}  # Element-wise addition
z_H^i = TransformerBlock(combined)
```

**Key property**: After updating, it provides a NEW context to L-module, causing L to restart its convergence toward a DIFFERENT equilibrium.

### 4. Output Head (f_O)

**Purpose**: Convert final hidden state to predictions.

```
ŷ = f_O(z_H^{NT}; θ_O) = softmax(θ_O · z_H^{NT})

Where:
- z_H^{NT} is the H-state after all N×T steps
- ŷ ∈ R^{l' × V} are token probabilities for output sequence
```

The output comes from H-module only (not L-module). This makes sense: the "plan" produces the answer, not the detailed computation.

---

## Deep Supervision: Multiple Segments

One forward pass might not be enough. HRM uses **deep supervision**:

```python
# Deep Supervision Training
for x, y_true in train_dataloader:
    z = z_init  # Initialize hidden states

    for step in range(M_supervision):
        z, y_hat = hrm(z, x)           # Forward pass
        loss = cross_entropy(y_hat, y_true)  # Compute loss

        z = z.detach()  # Crucial: stop gradients between segments
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Key insight**: The `z.detach()` means gradients don't flow between segments. Each segment learns independently to improve the answer. This is like "giving feedback" periodically rather than waiting until the end.

**Total computational depth**: N × T × M

With N=2, T=8, M=8: That's 128 "thinking steps" - far more than a typical Transformer's depth!

---

## Adaptive Computational Time (ACT)

Not all problems need the same amount of thinking. ACT learns to halt early when confident:

### The Q-Learning Approach

A Q-head predicts whether to halt or continue:

```
Q̂^m = σ(θ_Q · z_H^{mNT})

Q̂^m = (Q̂_halt, Q̂_continue)
```

**Decision rule**:
- If Q̂_halt > Q̂_continue and m ≥ M_min: HALT
- Otherwise: CONTINUE

**Reward structure** (episodic MDP):
- Halt and correct: reward = 1
- Halt and wrong: reward = 0
- Continue: reward = 0 (but might get 1 later)

### Why This Is "Thinking, Fast and Slow"

Easy problem (like adding 2+2):
- After segment 1, Q̂_halt > Q̂_continue
- Model halts immediately
- Minimal computation

Hard problem (like complex Sudoku):
- Q̂_continue stays high for many segments
- Model keeps "thinking"
- More computation for harder problems

**This is adaptive depth**: the model learns how long to think!

---

## Concrete Example: Sudoku Processing

Let's trace through how HRM might solve a Sudoku:

```
Input: 9×9 grid with some cells filled

Segment 1:
├── Cycle 1: L-module scans for obvious fills, H-module notes "row 3 is constrained"
├── Cycle 2: L-module fills obvious cells, H-module notes "try block 5 next"
└── Output: Partially filled grid (maybe 60% complete)

Segment 2:
├── Cycle 1: L-module finds harder constraints, H-module notes "conflict if 7 here"
├── Cycle 2: L-module tries alternatives, H-module tracks "backtracking needed"
└── Output: More filled (maybe 80% complete)

Segment 3:
├── Cycle 1: L-module resolves remaining ambiguities
├── Cycle 2: L-module verifies all constraints
└── Output: Complete solution

Q-head: Q̂_halt > Q̂_continue → HALT
```

Each segment refines the solution. The H-module maintains the "strategy" while the L-module does the detailed constraint checking.

---

## Architecture Comparison

### HRM vs Standard Transformer

| Aspect | Transformer | HRM |
|--------|-------------|-----|
| Depth | Fixed (L layers) | Variable (N×T×M) |
| Memory | O(L) | O(1) for gradients |
| All-to-all attention | Yes (per layer) | Yes (within each module) |
| Recurrence | No | Yes (both modules) |
| Output from | Last layer | H-module |

### HRM vs Standard RNN

| Aspect | RNN | HRM |
|--------|-----|-----|
| Single recurrent state | Yes | No (two hierarchical states) |
| Premature convergence | Yes (major problem) | Solved via hierarchical convergence |
| Training | BPTT (O(T) memory) | One-step gradient (O(1)) |
| Depth control | Manual (T steps) | Adaptive (ACT) |

### HRM vs Universal Transformer

| Aspect | Universal Transformer | HRM |
|--------|----------------------|-----|
| Recurrence | Single loop over layers | Two-level hierarchy |
| Depth | ACT-based | Hierarchical + ACT |
| Convergence | Can still be premature | Hierarchical convergence avoids it |
| Training | Standard BPTT | One-step gradient |

---

## Why The Hierarchy Matters: A Physical Analogy

**Single-level recurrence** (like standard RNN):

Imagine stirring a cup of coffee. At first, there's lots of motion. But eventually, the coffee settles to a uniform state. No amount of "more time" adds complexity - you've reached equilibrium.

**Hierarchical recurrence** (like HRM):

Imagine an orchestra. The conductor (H-module) gives high-level direction: "Now we play the loud section." The musicians (L-module) execute rapidly within that direction. When the conductor signals a new section, everything changes - new tempo, new dynamics, new complexity.

The coffee cup always settles. The orchestra can keep producing rich, varied music indefinitely because the conductor keeps providing new direction.

**HRM's H-module is the conductor**: It resets L-module's "direction" periodically, preventing premature settling.

---

## Summary: The Architecture in One Picture

```
                    ┌────────────────────────────────────────┐
                    │           Deep Supervision             │
                    │    (Multiple Forward Pass Segments)    │
                    └────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────────────┐
                    │           One Forward Pass             │
                    │         (N High-Level Cycles)          │
                    └────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌─────────────────┐                 ┌─────────────────┐
         │   H-Module      │                 │   L-Module      │
         │ (Slow Planning) │◄────────────────│ (Fast Compute)  │
         │ Updates: 1/T    │────────────────►│ Updates: 1/step │
         └─────────────────┘                 └─────────────────┘
                    │                                   │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────────┐
                    │            Output Head                 │
                    │     ŷ = softmax(θ_O · z_H^final)       │
                    └────────────────────────────────────────┘
```

**Total computational depth** = N (H-cycles) × T (L-steps per cycle) × M (segments)

With typical values N=2, T=8, M=8: **128 effective layers of computation**, all with O(1) gradient memory!

---

## Understanding N, T, M: The Three Levels of Depth

### The Visual Structure

```
M = 3 SEGMENTS (Deep Supervision)
════════════════════════════════════════════════════════════════════════

SEGMENT 1                    SEGMENT 2                    SEGMENT 3
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│                      │    │                      │    │                      │
│  N=2 CYCLES          │    │  N=2 CYCLES          │    │  N=2 CYCLES          │
│  ┌────────┬────────┐ │    │  ┌────────┬────────┐ │    │  ┌────────┬────────┐ │
│  │Cycle 1 │Cycle 2 │ │    │  │Cycle 1 │Cycle 2 │ │    │  │Cycle 1 │Cycle 2 │ │
│  │        │        │ │    │  │        │        │ │    │  │        │        │ │
│  │ T=4    │ T=4    │ │    │  │ T=4    │ T=4    │ │    │  │ T=4    │ T=4    │ │
│  │ steps  │ steps  │ │    │  │ steps  │ steps  │ │    │  │ steps  │ steps  │ │
│  │        │        │ │    │  │        │        │ │    │  │        │        │ │
│  └────────┴────────┘ │    │  └────────┴────────┘ │    │  └────────┴────────┘ │
│                      │    │                      │    │                      │
└──────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
           │                           │                           │
           ▼                           ▼                           ▼
       loss₁ → ∇θ                 loss₂ → ∇θ                  loss₃ → ∇θ
           │                           │                           │
       z.detach()                 z.detach()                  z.detach()
           │                           │                           │
           └───────────────────────────┴───────────────────────────┘
```

### Breaking It Down

**T = Steps per L-module cycle**
- T is how many times the L-module updates before H-module gets to update once
- Think of it as: "How many quick thoughts before reconsidering strategy?"
- With T=4: L updates 4 times, then H updates once

```
Within ONE CYCLE (T=4 steps):

Step 1: L-module processes → z_L¹     H-module: sleeping (uses old z_H)
Step 2: L-module processes → z_L²     H-module: sleeping
Step 3: L-module processes → z_L³     H-module: sleeping
Step 4: L-module processes → z_L⁴     H-module: WAKES UP, updates → z_H¹
        ─────────────────────────────────────────────────────────────
        End of cycle. z_H changed! L-module now has new context.
```

**N = Number of H-L cycles per forward pass**
- N is how many times H-module updates in one forward pass
- Think of it as: "How many strategic reconsiderations per 'thinking session'?"
- With N=2 and T=4: Total of 2×4=8 L-module steps, 2 H-module updates

```
ONE FORWARD PASS (N=2 cycles, T=4 steps each):

Cycle 1 (steps 1-4):
├── L runs 4 times with z_H⁰
├── L converges toward equilibrium given z_H⁰
└── H updates: z_H⁰ → z_H¹

Cycle 2 (steps 5-8):
├── L runs 4 times with z_H¹ (NEW context!)
├── L converges toward DIFFERENT equilibrium given z_H¹
└── H updates: z_H¹ → z_H²

Output: prediction based on z_H²
```

**M = Number of segments (with deep supervision)**
- M is how many times we repeat the entire forward pass
- Think of it as: "How many drafts before the final answer?"
- After each segment: compute loss, backprop, update weights, DETACH state

```
M=3 SEGMENTS:

Segment 1: [N×T steps] → ŷ₁ → loss₁ → gradients → z.detach()
                                                        ↓
Segment 2: [N×T steps] → ŷ₂ → loss₂ → gradients → z.detach()
                                                        ↓
Segment 3: [N×T steps] → ŷ₃ → loss₃ → gradients → DONE

Each segment starts from where the previous ended (but without gradient connection)
```

### The Math

**Total L-module updates per input** = N × T × M
**Total H-module updates per input** = N × M
**Effective computational depth** = N × T × M

With typical values (N=2, T=8, M=8):
- L-module updates: 2 × 8 × 8 = 128 times
- H-module updates: 2 × 8 = 16 times
- But gradients only flow through the LAST step of each segment!

---

## Deep Dive: z.detach() and Why It's Crucial

### What Does detach() Do?

In PyTorch, every tensor can track the operations that created it, forming a **computation graph**:

```python
a = torch.tensor([1.0], requires_grad=True)
b = a * 2  # b remembers: "I was created by a * 2"
c = b + 3  # c remembers: "I was created by b + 3"

c.backward()  # Follows the graph: c → b → a, computes gradients
```

**detach()** breaks this connection:

```python
a = torch.tensor([1.0], requires_grad=True)
b = a * 2
c = b.detach()  # c is a COPY of b, but "forgets" how it was made
d = c + 3

d.backward()  # ERROR! d → c → ??? (c doesn't know about a)
```

### Why HRM Uses detach() Between Segments

**Without detach():**

```
Segment 1: x → [N×T steps] → z¹ → loss₁
                              ↓
Segment 2: z¹ → [N×T steps] → z² → loss₂
                              ↓
Segment 3: z² → [N×T steps] → z³ → loss₃

When loss₃.backward() runs:
- Gradients flow back through segment 3 (N×T steps)
- Gradients flow back through segment 2 (N×T steps)
- Gradients flow back through segment 1 (N×T steps)

Memory required: O(M × N × T) - HUGE!
```

**With detach():**

```
Segment 1: x → [N×T steps] → z¹ → loss₁
                              ↓
                         z¹.detach() ──→ z¹' (copy, no gradient connection)
                              ↓
Segment 2: z¹' → [N×T steps] → z² → loss₂
                              ↓
                         z².detach() ──→ z²' (copy, no gradient connection)
                              ↓
Segment 3: z²' → [N×T steps] → z³ → loss₃

When loss₃.backward() runs:
- Gradients flow back through segment 3 ONLY (N×T steps)
- z²' blocks further backprop (it was detached)

Memory required: O(N × T) - CONSTANT regardless of M!
```

### The Implementation Pattern

```python
# Deep Supervision Training Loop
z = z_init.clone()

for segment in range(M):
    # Forward pass through one segment
    z, y_hat = hrm(z, x)  # z is updated through N×T steps

    # Compute loss on current prediction
    loss = cross_entropy(y_hat, y_true)

    # Backprop - but ONLY through this segment
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # CRUCIAL: Break gradient connection to previous segment
    z = z.detach()  # z is now a "fresh start" for the next segment
```

### Why This Still Works

You might worry: "If we don't backprop through earlier segments, how does the model learn to use them?"

**Answer 1: Each segment learns independently**
- Segment 1 learns: "Given input x, how to produce good z¹"
- Segment 2 learns: "Given z¹ (whatever it is), how to improve to z²"
- Each segment improves on its predecessor, without needing the full history

**Answer 2: The one-step gradient approximation**
- Within each segment, we use the one-step gradient (see hierarchical convergence doc)
- This approximates the full gradient at the equilibrium point
- We don't need the full trajectory, just the final state

**Answer 3: Biological plausibility**
- The brain doesn't do backpropagation through time
- Learning is based on recent activity and outcomes
- Deep supervision mirrors this: "Did this thinking step help? Learn from it."

---

## Q-Learning and ACT: Adaptive Computational Time

### The Problem ACT Solves

Fixed computation depth wastes resources:

```
Easy problem (2+2=?):
- A 128-step model still runs 128 steps
- Answer is obvious after step 1
- Steps 2-128 are wasted computation

Hard problem (complex Sudoku):
- Same 128 steps might not be enough
- Model is forced to give answer at step 128
- Might need step 200 to actually solve it
```

### The Solution: Learn When to Stop

HRM adds a **Q-head** that predicts whether stopping now will lead to a correct answer:

```
After each segment m:

z_H → Q-head → Q̂(s, a) = [Q̂_halt, Q̂_continue]

Q̂_halt = "Expected reward if I stop now"
Q̂_continue = "Expected reward if I keep going"

Decision: If Q̂_halt > Q̂_continue → STOP
          Otherwise → CONTINUE
```

### What is Q-Learning? (First Principles)

**Q-Learning** comes from reinforcement learning. The "Q" stands for "Quality":

```
Q(state, action) = Expected future reward starting from 'state' and taking 'action'
```

**Example: Chess**
```
State: Current board position
Actions: All legal moves
Q(position, move) = "How good is making this move in this position?"

High Q → This move likely leads to winning
Low Q → This move likely leads to losing
```

**For HRM:**
```
State: Current hidden state z_H after segment m
Actions: {halt, continue}

Q(z_H, halt) = "If I stop now, will my answer be correct?" (0 or 1)
Q(z_H, continue) = "If I keep thinking, will I eventually be correct?" (0 or 1)
```

### The Reward Structure

HRM uses an **episodic MDP** (Markov Decision Process):

```
Episode = Processing one puzzle from start to answer

Rewards:
- Halt and CORRECT: reward = 1  ← This is what we want!
- Halt and WRONG: reward = 0   ← Stopped too early
- Continue: reward = 0         ← No immediate reward, but maybe later...

Total reward = reward at halt (since continue gives 0)
```

### How the Q-Head Is Trained

Using **temporal difference learning**:

```
At segment m, we have:
- Q̂(z_H^m, halt): Predicted Q for halting
- Q̂(z_H^m, continue): Predicted Q for continuing

After we know the actual outcome:
- If we halted at m and got correct: Target = 1
- If we halted at m and got wrong: Target = 0
- If we continued to m+1: Target = max(Q̂(z_H^{m+1}, halt), Q̂(z_H^{m+1}, continue))

Loss = (Q̂ - Target)²
```

This is the classic **Bellman equation**: Q should equal reward + best future Q.

### The Gating Function

To ensure smooth learning, HRM uses a gating mechanism:

```
Before segment M_min: Always continue (don't halt too early)
After segment M_min: Can halt if Q̂_halt > Q̂_continue

Probability of halting: g(Q̂) = σ(α · (Q̂_halt - Q̂_continue - ε))

Where:
- σ = sigmoid (0 to 1)
- α = inverse temperature (higher = sharper decision)
- ε = margin (small bias toward continuing)
```

### Inference Time Behavior

```
Segment 1: Process → Q̂_continue > Q̂_halt → CONTINUE
Segment 2: Process → Q̂_continue > Q̂_halt → CONTINUE
Segment 3: Process → Q̂_halt > Q̂_continue → HALT! Output answer.

The model learned: "After 3 segments, I'm confident enough to stop."
```

For a harder problem:
```
Segments 1-5: Q̂_continue keeps winning → CONTINUE
Segment 6: Q̂_halt finally wins → HALT! Output answer.

The model learned: "This one needed more thinking."
```

---

## Addressing the Critique: Hierarchical vs Iterative Refinement

### The Critique

Some have questioned whether HRM's hierarchy is fundamentally different from simple iterative refinement:

> "Isn't this just running the same computation multiple times? What makes the hierarchy special?"

### The Short Answer

**No, it's not the same.** The hierarchy provides something iterative refinement cannot: **prevention of premature convergence** and **separation of timescales**.

### The Long Answer

**Simple Iterative Refinement:**

```python
# Iterative refinement (like Universal Transformer or just running RNN longer)
z = init
for step in range(many_steps):
    z = refine(z, x)  # Same operation every step
output = decode(z)
```

Problem: The state z converges to a fixed point and stops changing:

```
Step 1:  z = [0.5, -0.3, 0.8]   (varied)
Step 10: z = [0.4, -0.1, 0.5]   (less varied)
Step 30: z = [0.35, 0.0, 0.3]   (converging)
Step 50: z = [0.35, 0.0, 0.3]   (STUCK - no more change!)
```

After ~30 steps, the model has effectively stopped computing. More iterations add nothing.

**HRM's Hierarchical Structure:**

```python
# HRM's two-level hierarchy
z_L, z_H = init
for cycle in range(N):
    for step in range(T):
        z_L = L_refine(z_L, z_H, x)  # L converges given fixed z_H
    z_H = H_update(z_H, z_L)         # H changes! New target for L!
output = decode(z_H)
```

The key difference: **z_L's equilibrium DEPENDS on z_H**

```
Cycle 1 (z_H = "Strategy A"):
  Step 1:  z_L = [0.5, -0.3, 0.8]
  Step 4:  z_L = [0.2, 0.1, 0.4]  ← Converged to equilibrium for Strategy A

Cycle 2 (z_H = "Strategy B"): ← H-module changed the strategy!
  Step 5:  z_L = [0.2, 0.1, 0.4]  ← Starts from old value
  Step 6:  z_L = [0.6, -0.2, 0.7] ← But new z_H means new target!
  Step 8:  z_L = [0.8, 0.0, 0.9]  ← Converged to DIFFERENT equilibrium

Cycle 3 (z_H = "Strategy C"):
  ...and so on
```

### Why the Hierarchy Matters: Mathematical View

**Fixed point depends on context:**

For L-module, the fixed point z_L* satisfies:
```
z_L* = f_L(z_L*, z_H, x)
```

When z_H changes, z_L* changes too! The L-module must reconverge.

**The nested fixed-point structure:**

Let F(z_H) = "the equilibrium L reaches given z_H"

Then H-module computes:
```
z_H^{k+1} = f_H(z_H^k, F(z_H^k))
```

This is a **meta-level iteration** where each evaluation of F involves L converging to its own equilibrium.

### What Iterative Refinement Cannot Do

1. **No timescale separation**: Every step is the same speed. No "slow strategic" vs "fast tactical"

2. **Premature convergence**: State collapses to fixed point, additional computation is wasted

3. **No strategy switching**: Can't "change approach" midway through reasoning

4. **Gradient issues**: Long iterative chains have vanishing/exploding gradients

### What Hierarchy Provides

1. **Timescale separation**: H (slow) guides L (fast), matching biological brain organization

2. **Prevented convergence**: L converges within cycles, but H resets the target

3. **Strategy switching**: H can change "what we're trying to do" between cycles

4. **Gradient efficiency**: One-step approximation + deep supervision = O(1) memory

### Empirical Evidence (from the paper)

The paper measures **forward residual** (how much the state changes each step):

```
Standard RNN:
  Steps 1-30:  Residual decays exponentially
  Steps 30+:   Residual ≈ 0 (dead computation)

Deep Network:
  Input layers: High residual
  Middle layers: Very low (vanishing gradients)
  Output layers: High residual

HRM:
  Within cycles: Residual decays (L converging)
  At cycle boundaries: Residual SPIKES (H reset the target!)
  Sustained high computation throughout
```

The hierarchy keeps computation "alive" far longer than iterative refinement.

### The Biological Parallel

This isn't just a theoretical nicety - it matches how brains work:

```
Prefrontal Cortex (like H-module):
- Updates slowly (theta waves, 4-8 Hz)
- Maintains goals, plans, strategies
- High-dimensional representations (flexible)

Sensory/Motor Cortex (like L-module):
- Updates rapidly (gamma waves, 30-100 Hz)
- Executes specific computations
- Lower-dimensional representations (specialized)

The hierarchy is not arbitrary - it's how biological intelligence organizes computation!
```

### Summary: Why Hierarchy > Simple Iteration

| Property | Simple Iteration | HRM Hierarchy |
|----------|------------------|---------------|
| Convergence | Premature (dies after ~30 steps) | Sustained (H resets L's target) |
| Timescales | Single | Two (slow H, fast L) |
| Strategy changes | Impossible (same operation) | Natural (H-module updates) |
| Gradient memory | O(T) for T steps | O(1) with one-step approx |
| Biological match | Poor | Strong (matches cortex) |
| Effective depth | Limited by convergence | N × T × M (128+ steps) |

**The hierarchy isn't just "more computation" - it's a fundamentally different computational structure that enables sustained, strategy-guided reasoning.**
