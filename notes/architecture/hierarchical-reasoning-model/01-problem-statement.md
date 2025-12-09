# The Problem: Transformers Are Paradoxically Shallow

## The Core Issue

Despite being called "deep learning," modern LLMs (built on Transformers) are **paradoxically shallow** when it comes to reasoning. This isn't about parameter count or training data - it's a fundamental architectural limitation.

**The shocking result**: Transformers cannot solve problems that require polynomial-time computation, regardless of how many parameters or examples you give them.

---

## Problem Abstraction Ladder

### L1: Abstract Principle
> **True reasoning requires iterative refinement - the ability to "think" for as long as needed, revising intermediate conclusions based on new constraints.**

A Sudoku puzzle requires checking constraints, making tentative assignments, discovering conflicts, and backtracking. This process can't be compressed into a fixed number of "layers."

### L2: Framework Level
The limitation manifests in three ways:
- **Computational**: Fixed depth = fixed complexity class = some problems are literally impossible
- **Learning**: Chain-of-Thought is a workaround that externalizes reasoning, but is brittle and slow
- **Efficiency**: CoT requires massive pretraining and generates many tokens for complex problems

### L3: Method Level - Comparing Approaches

| Architecture | Depth | Complexity Class | Can Solve Sudoku? |
|-------------|-------|------------------|-------------------|
| Transformer (L layers) | Fixed: L | AC⁰ or TC⁰ | No (0% on hard) |
| RNN (T steps) | Fixed: T | Theoretically higher, but premature convergence | Poorly |
| Transformer + CoT | Effectively O(n) | Higher (via external tokens) | Partially |
| **HRM** | **Adaptive: N×T×M** | **Turing-complete** | **Yes (55%+ on extreme)** |

### L4: Concrete Impact

A Transformer with 100 layers processes information through exactly 100 transformations. Whether the problem is trivial or extremely hard, it gets the same computational budget.

This is like forcing a human to solve every math problem in exactly 100 mental steps - impossible for complex problems.

### L5: Specific Example

**Sudoku-Extreme** (hardest puzzles requiring backtracking):
- **o3-mini-high (with CoT)**: 0% accuracy
- **Claude 3.7 (with CoT)**: 0% accuracy
- **DeepSeek R1 (with CoT)**: 0% accuracy
- **HRM (27M params, 1000 examples)**: 55% accuracy

The LLMs aren't just worse - they **completely fail**. This isn't a scaling issue; it's an architectural impossibility.

---

## First Principles: Understanding Computational Depth

### Part 1: What is "Computational Depth"?

**Computational depth** is how many sequential operations a model can perform to transform input to output.

#### Concrete Example: Mental Math

Consider computing 847 × 293 in your head:

**Method 1: One-shot** (no depth)
- Look at 847 × 293
- Somehow "see" the answer: 248,171
- This requires memorizing all possible multiplications - impossible!

**Method 2: Iterative** (deep computation)
```
Step 1: 847 × 3 = 2,541
Step 2: 847 × 90 = 76,230
Step 3: 847 × 200 = 169,400
Step 4: 2,541 + 76,230 + 169,400 = 248,171
```

Method 2 requires multiple steps, but each step is simple. This is "depth" - breaking a hard problem into easy sequential steps.

### Part 2: Why Transformers Have Fixed Depth

A Transformer layer does this:

```
Input: x (a sequence of vectors)
       ↓
   [Self-Attention]  ← All positions talk to each other
       ↓
   [Feed-Forward]    ← Each position independently transformed
       ↓
Output: x' (transformed sequence)
```

If you have L layers, you apply this transformation L times:

```
x → Layer₁ → x₁ → Layer₂ → x₂ → ... → Layer_L → y
```

**The depth is exactly L**, regardless of the input. A 100-layer Transformer always does 100 transformations.

### Part 3: Complexity Classes - The Mathematical Limitation

**Complexity classes** categorize problems by how much computation they require.

#### Key Classes:

**AC⁰** (Alternating Circuits, constant depth):
- **A** = Alternating (layers alternate between AND and OR gates)
- **C** = Circuits
- **⁰** = Constant depth (doesn't grow with input size)
- Problems solvable with constant depth AND/OR circuits
- Examples: OR-ing bits together
- Cannot solve: Multiplication, parity (XOR of all bits)

```
AC⁰ circuit for OR of n bits:

Input:  x₁  x₂  x₃  x₄  ...  xₙ
         \   \   \   /    /
          \   \  |  /   /
           ────OR gate────
                |
             Output

Depth = 1 (constant), regardless of n
```

**TC⁰** (Threshold Circuits, constant depth):
- **T** = Threshold (gates can count and compare to a threshold)
- **C** = Circuits
- **⁰** = Constant depth
- More powerful than AC⁰ - can do counting and threshold operations
- Can solve: Addition (via carry-lookahead), sorting, even multiplication
- Still cannot solve: Iterative/sequential reasoning problems

```
TC⁰ adds THRESHOLD gates:

Input:  x₁  x₂  x₃  x₄  ...  xₙ
         \   \   |   /    /
          \   \  |  /   /
        ─THRESHOLD ≥ k─
                |
         Output (1 if ≥k inputs are 1)
```

**P** (Polynomial time):
- Problems solvable in polynomial time (n, n², n³, etc.)
- Includes: Sudoku (with backtracking), pathfinding, constraint satisfaction
- Requires: Variable depth depending on input

**The key theorem**: Standard Transformers are in AC⁰ or TC⁰. They **cannot** solve problems that require polynomial time, no matter how many parameters they have.

---

### Part 4: Why Addition Is Constant Depth But Multiplication (in AC⁰) Isn't

#### Addition IS Constant Depth (in TC⁰)

Consider adding two n-bit numbers:

```
  1011  (11)
+ 0110  (6)
──────
 10001 (17)
```

**Key insight**: With clever circuit design (carry-lookahead), you can compute all carries in parallel:

```
For each bit position i:
- Generate (Gᵢ): Would this position create a carry? Gᵢ = Aᵢ AND Bᵢ
- Propagate (Pᵢ): Would this position pass a carry? Pᵢ = Aᵢ XOR Bᵢ

Carry into position i:
Cᵢ = Gᵢ₋₁ OR (Pᵢ₋₁ AND Gᵢ₋₂) OR (Pᵢ₋₁ AND Pᵢ₋₂ AND Gᵢ₋₃) OR ...
```

This is a big OR of ANDs - **constant depth** with threshold gates.

#### Multiplication Is NOT in AC⁰

Consider multiplying two n-bit numbers - you need to add n partial products:

```
    1011  (11)
  × 0110  (6)
  ──────
    0000  (11 × 0)
   1011   (11 × 1, shifted)
  1011    (11 × 1, shifted)
 0000     (11 × 0, shifted)
──────────
  1000010 (66)
```

Adding n numbers requires O(log n) depth even with tree-structured addition - not constant!

**Formal result**: Multiplication requires computing parity of certain bit combinations. Parity is provably NOT in AC⁰ (Furst-Saxe-Sipser theorem, 1984).

#### Parity: The Canonical Hard Problem for AC⁰

**Parity of n bits**: Output 1 if odd number of 1s, else 0.

```
Parity(0,0,0) = 0
Parity(1,0,0) = 1
Parity(1,1,0) = 0
Parity(1,1,1) = 1
```

**Why it's hard for constant-depth circuits**:
- Every input bit affects the output
- Flipping ANY single input bit flips the output
- You can't "summarize" partial information - must consider ALL bits together
- With constant depth, each gate can only "see" a limited neighborhood

**Theorem** (Furst-Saxe-Sipser, 1984): Parity is not in AC⁰.

---

### Part 5: Why TC⁰ Can Count But Cannot "Reason"

#### What TC⁰ CAN Do

**Counting**: "Are there at least k 1s among these n bits?"
```
THRESHOLD-k(x₁, x₂, ..., xₙ) = 1 if Σxᵢ ≥ k
```
This is a single gate in TC⁰!

**Majority**: "Are more than half the bits 1?" - just THRESHOLD-⌈n/2⌉

**Addition**: Carry-lookahead uses threshold-like comparisons

**Multiplication**: Can actually be done in TC⁰ (unlike AC⁰)!

#### What TC⁰ CANNOT Do: Iterative Problems

Problems where step k depends on step k-1's result:

**Example: Following a path in a graph**
```
Graph:  A → B → C → D → E

Query: Is there a path from A to E?

Step 1: A connects to B ✓
Step 2: B connects to C ✓  (depends on knowing B from step 1!)
Step 3: C connects to D ✓  (depends on knowing C from step 2!)
Step 4: D connects to E ✓
Result: YES
```

With **constant depth**, you can check paths of constant length, but NOT paths of length O(n).

**The "reasoning" gap**: Real reasoning requires:
1. **State-dependent computation**: What you do at step k depends on step k-1
2. **Unbounded iteration**: Number of steps scales with problem size
3. **Backtracking**: Try one approach, detect failure, try another

None of these fit in constant-depth circuits!

---

### Part 6: More Polynomial-Time Problems Transformers Cannot Solve

Beyond Sudoku, here are concrete examples from research:

#### Graph Problems

**Graph Connectivity**: Given a graph, determine if two nodes are connected
- Requires: Following paths of potentially O(n) length
- Transformers: Cannot reliably determine connectivity in large graphs

**Shortest Path**: Find optimal route between two nodes
- Requires: BFS/Dijkstra - inherently iterative
- LLMs: 0% on 30×30 mazes (HRM: 74.5%)

#### Arithmetic Problems

**Multi-digit Multiplication**: 847 × 293 = ?
- Requires: Carrying across positions - inherently sequential
- Transformers: Accuracy drops sharply with digit count

**Modular Exponentiation**: Compute a^b mod n for large numbers
- Requires: Sequential squaring and multiplication
- Fixed depth cannot handle arbitrary exponent sizes

#### String/Sequence Problems

**Longest Common Subsequence**: Find longest sequence in both strings
- Requires: Dynamic programming with O(n²) dependencies

**Parity over long sequences**: Is the count of 1s odd or even?
- Proven impossible for AC⁰ (and thus limited Transformers)

#### Planning Problems

**Tower of Hanoi**: Move disks following rules
- Optimal solution requires 2^n - 1 moves
- Cannot be "pattern matched" - requires recursive reasoning

**Boolean Satisfiability (SAT)**: Find variable assignment making formula true
- Requires: Backtracking search
- Transformers fail on complex formulas

**These are exactly the problems where LLMs fail most spectacularly.**

The paper shows this empirically: increasing Transformer width does nothing for Sudoku, but increasing depth helps somewhat (Figure 2). However, even very deep Transformers saturate far below good performance.

---

## The Chain-of-Thought Workaround

### What CoT Does

Chain-of-Thought "solves" the depth problem by:
1. Having the model generate intermediate tokens
2. Each generated token becomes input for the next
3. Effectively creating depth through sequence length

```
Input: "Solve this Sudoku..."
       ↓
Output: "Let me think step by step.
         First, cell (1,1) must be 5 because...
         Next, cell (1,2) could be 3 or 7, let me check...
         [continues for many tokens]
         The answer is: [solution]"
```

### Why CoT Is "A Crutch, Not A Solution"

The paper makes a strong claim: CoT is fundamentally limited.

#### Problem 1: Brittle Task Decomposition
CoT relies on the model knowing how to break down the problem. A single wrong step derails everything.

Example: If the model says "cell (1,1) must be 5" but it's actually 3, the entire reasoning chain is corrupted.

#### Problem 2: Tethered to Token-Level Patterns
CoT forces reasoning to happen in "language space." But reasoning isn't fundamentally linguistic - the brain reasons in abstract representations, not words.

The paper cites research showing: "language is a tool for human communication, not the substrate of thought itself."

#### Problem 3: Massive Data Requirements
CoT reasoning must be learned from examples. This requires:
- Huge pretraining corpora
- Potentially training on reasoning traces
- Many examples of similar problems

HRM achieves better results with 1000 examples vs. billions for LLMs.

#### Problem 4: High Latency
CoT generates many tokens. For a complex Sudoku:
- o3-mini-high might generate 10,000+ tokens
- Each token requires a full forward pass
- Result: Very slow inference

HRM solves problems in latent space - no token generation, much faster.

---

## The RNN Alternative (And Why It Also Fails)

### What RNNs Offer

RNNs have recurrence - they can process sequences of arbitrary length:

```
h₁ = f(x₁, h₀)
h₂ = f(x₂, h₁)
h₃ = f(x₃, h₂)
...
```

Theoretically, if you run an RNN for T steps, you get depth T.

### How RNN Steps Give Depth

```
RNN unrolled:

x₁ → [f, θ] → h₁
              ↓
x₂ ─────────→ [f, θ] → h₂
                       ↓
x₃ ────────────────→ [f, θ] → h₃
                              ↓
                             ...

Same function f with same weights θ applied T times = depth T
```

**Theoretically**: Running for T steps means T sequential transformations, so depth T.

---

### Universal Transformers: Looped Transformers

**Universal Transformers** (Dehghani et al., 2018) extend the original Transformer by adding recurrence:

#### Original Transformer (Vaswani et al., 2017 - "Attention Is All You Need")

```
Input → [Layer 1] → [Layer 2] → ... → [Layer L] → Output
           θ₁          θ₂                θ_L

Each layer has DIFFERENT weights: θ₁, θ₂, ..., θ_L
Depth is FIXED at L
```

#### Universal Transformer

```
Input → [Shared Layer] → [Shared Layer] → ... → [Shared Layer] → Output
              θ                θ                      θ

Same weights θ used for ALL iterations
Depth can be VARIABLE (with ACT halting)
```

#### Key Differences

| Aspect | Original Transformer | Universal Transformer |
|--------|---------------------|----------------------|
| Weights | Different per layer | **Shared across all iterations** |
| Depth | Fixed: L layers | **Variable** (with ACT) |
| Computation | Exactly L passes | **Adaptive** - more for harder inputs |
| Parameters | L × (per-layer params) | 1 × (per-layer params) |
| Complexity class | AC⁰/TC⁰ | **Turing-complete** (theoretically) |

#### The ACT (Adaptive Computation Time) Mechanism

```
Iteration 1: x → SharedLayer → x₁, compute ponder score p₁
Iteration 2: x₁ → SharedLayer → x₂, compute ponder score p₂
...
HALT when: cumulative ponder score exceeds threshold

Output: weighted average of x₁, x₂, ..., x_T
```

**Easy problem**: Halts after 3 iterations
**Hard problem**: Continues for 20 iterations

#### Theoretical Result

With unbounded iterations and sufficient width, Universal Transformers are **Turing-complete** - they can compute anything computable.

#### But Universal Transformers Still Have Problems

1. **Premature convergence**: The shared weights lead to early settling (same as RNNs)
2. **BPTT required**: Training through T iterations needs O(T) memory
3. **Practical depth limited**: Even with ACT, effective depth is bounded in practice

**This is exactly what HRM addresses!**

---

### The Premature Convergence Problem

In practice, both RNNs and Universal Transformers suffer from **premature convergence**:

```
Hidden states over time (RNN or Universal Transformer):

Step 1:   [0.2, -0.5, 0.8, 0.1, ...]  (active, varied)
Step 5:   [0.3, -0.2, 0.5, 0.2, ...]  (still changing)
Step 20:  [0.4, -0.1, 0.3, 0.25, ...] (slowing down)
Step 50:  [0.41, -0.09, 0.29, 0.26, ...] (almost frozen)
Step 100: [0.41, -0.09, 0.29, 0.26, ...] (converged - no more change!)
```

**What happens**: The hidden state settles toward a fixed point. Updates become tiny. Additional steps contribute nothing.

**Why this is catastrophic**: You might have 100 steps of depth, but effectively only 20 are doing useful computation. The model "gives up" before solving the problem.

**Effective depth << Actual depth**

### Universal Transformer Has The Same Problem

Because Universal Transformers use shared weights, they're effectively RNNs:

```
Universal Transformer iterations:

Iteration 1: x → SharedLayer → x₁  (big change)
Iteration 2: x₁ → SharedLayer → x₂ (smaller change)
Iteration 3: x₂ → SharedLayer → x₃ (even smaller)
...
Iteration 20+: almost no change (converged to fixed point)
```

The shared weights create a contraction mapping - same convergence issues as RNNs!

### Why Convergence Happens

Models with recurrence are typically trained to produce stable outputs. The gradients push toward:
- Smooth transitions
- Consistent hidden states
- Predictable dynamics (contraction mappings)

A **contraction mapping** f satisfies: `||f(h₁) - f(h₂)|| ≤ k × ||h₁ - h₂||` where k < 1

This means the function brings points closer together. Eventually, everything collapses to a unique fixed point.

**Good for**: Training stability, predictable behavior
**Bad for**: Computational depth, complex reasoning

---

## The BPTT Problem

### What Is BPTT?

**Backpropagation Through Time** is how RNNs are trained:

1. Unroll the RNN for T steps
2. Store all intermediate hidden states: h₁, h₂, ..., h_T
3. Compute loss at the end
4. Backpropagate gradients through all T steps

### Memory Requirement

BPTT requires O(T) memory - storing all hidden states for T timesteps.

For T = 1000 steps with hidden size 512:
- Need to store 1000 × 512 = 512,000 floats per sample
- With batch size 64: ~130 million floats = 500+ MB just for hidden states
- Plus gradients: doubles the memory

This forces:
- Small batch sizes (poor GPU utilization)
- Limited depth (can't do very long reasoning)
- Gradient instability (vanishing/exploding over many steps)

### Biological Implausibility

The brain definitely doesn't do BPTT:
- No "replay" of full activity history
- Credit assignment happens locally
- Learning is based on local, temporally-proximate signals

This suggests BPTT might not be the "right" way to train deep recurrent systems.

---

## What HRM Does Differently

HRM addresses all these problems:

| Problem | HRM Solution |
|---------|--------------|
| Fixed depth | Hierarchical recurrence: N cycles × T steps × M segments |
| Premature convergence | Hierarchical convergence: L-module reset by H-module |
| BPTT memory | One-step gradient: O(1) memory |
| CoT brittleness | Latent reasoning: no external tokens |
| Data hunger | Direct supervision: 1000 examples suffice |

The key insight: **Brain-inspired hierarchical structure solves multiple problems at once.**

---

## Why This Matters for Cricket Ball-by-Ball Prediction

Cricket prediction might seem "easier" than Sudoku, but consider:

1. **Constraint Satisfaction**: Match state must be consistent (runs, wickets, overs)
2. **Strategic Reasoning**: What should the batsman do given the field placement?
3. **Multi-step Planning**: How should an innings be paced over 50 overs?
4. **Counterfactual Reasoning**: What if the batsman had played differently?

A model that can truly "reason" about cricket - not just pattern-match - would need variable computational depth. Simple balls (predictable dot ball) need less thinking than complex situations (final over of a chase).

HRM's adaptive computation (via ACT) naturally allocates more "thinking time" to harder predictions.

---

## Summary

| Limitation | Why It Matters | HRM's Solution |
|------------|---------------|----------------|
| Transformers are fixed-depth | Can't solve polynomial-time problems | Recurrence enables arbitrary depth |
| RNNs converge prematurely | Effective depth << actual depth | Hierarchical convergence resets L-module |
| BPTT needs O(T) memory | Limits practical depth | One-step gradient approximation |
| CoT is brittle and slow | Fails on complex reasoning, high latency | Latent reasoning in hidden states |
| LLMs need massive data | Impractical for specialized domains | 1000 examples suffice |
