# GAT Architecture: Attention Over Graph Neighborhoods

## The Big Picture

GAT applies the attention mechanism from Transformers to graph-structured data:

```
┌─────────────────────────────────────────────────────────────┐
│                    Graph Attention Layer                     │
│                                                              │
│   Input: Node features h = {h₁, h₂, ..., hₙ}, Graph edges   │
│                              │                               │
│                              ▼                               │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  1. Linear Transform: W·hᵢ for all nodes             │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  2. Attention Scores: e_ij = a(W·hᵢ, W·hⱼ)           │  │
│   │     (only for neighbors j ∈ N_i)                     │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  3. Normalize: α_ij = softmax_j(e_ij)                │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  4. Aggregate: h'ᵢ = σ(Σⱼ α_ij · W·hⱼ)               │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                               │
│   Output: New node features h' = {h'₁, h'₂, ..., h'ₙ}       │
└─────────────────────────────────────────────────────────────┘
```

---

## First Principles: Building the Attention Mechanism

### Step 1: Linear Transformation

Every node's features are transformed by a shared weight matrix:

```
Input: h = {h₁, h₂, ..., hₙ},  hᵢ ∈ R^F  (F input features)

Transform: W ∈ R^(F' × F)  (shared across all nodes)

Output: W·hᵢ ∈ R^F'  for each node i
```

**Why this step?**
- Project features to a new space where attention makes sense
- Shared W means the transformation is the same for all nodes
- F' can be different from F (dimensionality change)

### Step 2: Compute Attention Scores

For each pair of connected nodes, compute how much node j should influence node i:

```
Attention mechanism: a : R^F' × R^F' → R

e_ij = a(W·hᵢ, W·hⱼ)

"How important is node j's features for computing node i's new representation?"
```

**GAT's specific attention mechanism**:

```
e_ij = LeakyReLU(aᵀ · [W·hᵢ || W·hⱼ])

Where:
- || denotes concatenation
- a ∈ R^(2F') is a learnable weight vector
- LeakyReLU has negative slope 0.2
```

**Unpacking this formula**:

```
Step-by-step for nodes i and j:

1. Transform both: W·hᵢ and W·hⱼ (each is F'-dimensional)

2. Concatenate: [W·hᵢ || W·hⱼ] ∈ R^(2F')

3. Dot product with a: aᵀ · [W·hᵢ || W·hⱼ] ∈ R (scalar)

4. Apply LeakyReLU: LeakyReLU(aᵀ · [W·hᵢ || W·hⱼ]) ∈ R

Result: Single scalar score e_ij
```

### Step 3: Masked Attention (Injecting Graph Structure)

**Critical**: We only compute attention between connected nodes.

```
Without masking (full attention):
- Every node attends to every other node
- Graph structure is ignored
- O(n²) computation

With masking:
- Node i only attends to neighbors N_i
- Graph structure determines which pairs are computed
- O(|E|) computation
```

**Implementation**:
```python
# Only compute e_ij where edge (i,j) exists
for i in nodes:
    for j in neighbors[i]:  # Only neighbors, not all nodes!
        e[i,j] = attention(W @ h[i], W @ h[j])
```

### Step 4: Normalize with Softmax

Attention scores are normalized so they sum to 1 for each node:

```
α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈N_i} exp(e_ik)

Properties:
- α_ij ≥ 0 for all j
- Σ_{j∈N_i} α_ij = 1
- Comparable across different nodes (despite different neighborhood sizes)
```

**Why softmax over neighbors only?**
```
Node i with neighbors {A, B, C}:

e_iA = 2.0
e_iB = 1.0
e_iC = 0.5

α_iA = exp(2.0) / (exp(2.0) + exp(1.0) + exp(0.5))
     = 7.39 / (7.39 + 2.72 + 1.65)
     = 7.39 / 11.76
     = 0.63

α_iB = 0.23
α_iC = 0.14

Sum: 0.63 + 0.23 + 0.14 = 1.0 ✓
```

### Step 5: Aggregate Weighted Neighbor Features

The final node representation is a weighted sum:

```
h'ᵢ = σ(Σ_{j∈N_i} α_ij · W·hⱼ)

Where σ is a nonlinearity (typically ELU or ReLU)
```

**Putting it all together**:
```
h'ᵢ = σ(Σ_{j∈N_i} [exp(LeakyReLU(aᵀ[W·hᵢ || W·hⱼ])) / Σ_k exp(LeakyReLU(aᵀ[W·hᵢ || W·hₖ]))] · W·hⱼ)
```

---

## Multi-Head Attention

### Why Multiple Heads?

Single attention head might be unstable or miss important patterns. Multiple heads provide:
- **Stabilization**: Different heads can capture different aspects
- **Expressiveness**: Multiple "views" of the neighborhood
- **Robustness**: If one head fails, others compensate

### Multi-Head Implementation

K independent attention mechanisms run in parallel:

```
For head k ∈ {1, 2, ..., K}:
- Separate weights: W^k, a^k
- Separate attention: α^k_ij
- Separate aggregation: head_k = σ(Σⱼ α^k_ij · W^k·hⱼ)

Combine heads by concatenation:
h'ᵢ = ||_{k=1}^K head_k
```

**Dimensions**:
```
Input: hᵢ ∈ R^F
Each head output: R^(F'/K)
Concatenated output: R^F' (K heads × F'/K each)
```

### Final Layer: Average Instead of Concatenate

For the last layer (classification), concatenation would give K× too many features. Instead, average:

```
h'ᵢ = σ(1/K · Σ_{k=1}^K Σ_{j∈N_i} α^k_ij · W^k·hⱼ)
```

---

## Visual Walkthrough: Single Node Update

```
                     Node 1's Neighborhood

        h₂ ─────────┐
           α₁₂=0.12 │
        h₃ ─────────┼───────────────┐
           α₁₃=0.13 │               │
        h₄ ─────────┤    Weighted   │
           α₁₄=0.14 │     Sum       ├──→ h'₁
        h₅ ─────────┤               │
           α₁₅=0.15 │               │
        h₆ ─────────┘               │
           α₁₆=0.16                 │
                                    │
        h₁ (self) ──────────────────┘
           α₁₁=0.30

Note: Self-loop included (node attends to itself)
      Attention weights sum to 1.0
```

**Multi-head version (K=3 heads)**:

```
Head 1:  h₂,h₃,h₄,h₅,h₆,h₁ ──[α¹_ij]──→ σ(Σ α¹·W¹·h) ─┐
                                                       │
Head 2:  h₂,h₃,h₄,h₅,h₆,h₁ ──[α²_ij]──→ σ(Σ α²·W²·h) ─┼──→ [concat] ──→ h'₁
                                                       │
Head 3:  h₂,h₃,h₄,h₅,h₆,h₁ ──[α³_ij]──→ σ(Σ α³·W³·h) ─┘

Each head has different attention patterns!
```

---

## The Full GAT Model

### Architecture for Node Classification

```
Input Graph: N nodes, F features each
│
▼
┌────────────────────────────────────────┐
│  GAT Layer 1                           │
│  - K=8 attention heads                 │
│  - F'=8 features per head              │
│  - Output: 64 features (8×8)           │
│  - ELU activation                      │
│  - Dropout on input and attention      │
└────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐
│  GAT Layer 2 (Classification)          │
│  - K=1 attention head (or K=8 averaged)│
│  - F'=C features (C = num classes)     │
│  - Softmax activation                  │
└────────────────────────────────────────┘
│
▼
Output: Class probabilities for each node
```

### Hyperparameters Used in Paper

**Transductive (Cora, Citeseer)**:
```
- 2 GAT layers
- Layer 1: K=8 heads, F'=8 features each → 64 total
- Layer 2: K=1 head, F'=C classes
- Dropout: p=0.6 on inputs AND attention
- L2 regularization: λ=0.0005
- Learning rate: 0.005
- Activation: ELU
```

**Inductive (PPI)**:
```
- 3 GAT layers
- Layer 1-2: K=4 heads, F'=256 features each → 1024 total
- Layer 3: K=6 heads, F'=121 features (averaged)
- Skip connections between layers
- No dropout (large training set)
- Learning rate: 0.005
- Activation: ELU
```

---

## Comparison: GAT vs Transformer Attention

### Similarities

| Aspect | Transformer | GAT |
|--------|-------------|-----|
| Core operation | Scaled dot-product attention | Additive attention |
| Multi-head | Yes (K heads concatenated) | Yes (K heads concatenated) |
| Softmax normalization | Yes | Yes |
| Learned query/key/value | Q, K, V matrices | W matrix + a vector |

### Key Differences

**1. Attention Scope**:
```
Transformer: All-to-all (every position attends to every other)
     ┌─────────────────┐
     │ ● ● ● ● ● ● ● ● │  Every token attends to every token
     └─────────────────┘

GAT: Masked (only neighbors)
     ┌─────────────────┐
     │ ●─●   ●─●─●     │  Only connected nodes attend
     │   │     │       │
     │   ●─────●       │
     └─────────────────┘
```

**2. Attention Formula**:
```
Transformer (scaled dot-product):
α_ij = softmax(Q_i · K_j^T / √d)

GAT (additive):
α_ij = softmax(LeakyReLU(a^T · [W·hᵢ || W·hⱼ]))
```

**3. Position Information**:
```
Transformer: Needs explicit positional encoding (sinusoidal/learned)
GAT: Structure encoded implicitly via adjacency (which nodes connect)
```

**4. Computational Complexity**:
```
Transformer: O(n²) in sequence length
GAT: O(|V|·F·F' + |E|·F') - scales with edges, not all pairs
```

---

## Implementation Details

### The Attention Mechanism in Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.6, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        # Shared linear transformation
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Attention parameters (one per head)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: Node features [N, in_features]
        adj: Adjacency matrix [N, N] (sparse or dense)
        """
        N = h.size(0)

        # Step 1: Linear transform
        Wh = self.W(h)  # [N, n_heads * out_features]
        Wh = Wh.view(N, self.n_heads, self.out_features)  # [N, heads, F']

        # Step 2: Compute attention scores
        # For each head, compute a^T [Wh_i || Wh_j] for all edges
        Wh_i = Wh.unsqueeze(2)  # [N, heads, 1, F']
        Wh_j = Wh.unsqueeze(1)  # [N, 1, heads, F'] -> broadcast to [N, N, heads, F']

        # Concatenate and apply attention
        concat_features = torch.cat([
            Wh_i.expand(-1, -1, N, -1),  # [N, heads, N, F']
            Wh_j.expand(-1, N, -1, -1).transpose(1, 2)  # [N, heads, N, F']
        ], dim=-1)  # [N, heads, N, 2*F']

        e = (concat_features * self.a.unsqueeze(0).unsqueeze(2)).sum(dim=-1)  # [N, heads, N]
        e = self.leaky_relu(e)

        # Step 3: Masked softmax (only over neighbors)
        mask = (adj == 0).unsqueeze(1)  # [N, 1, N]
        e = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=-1)  # [N, heads, N]
        alpha = self.dropout(alpha)

        # Step 4: Aggregate
        h_prime = torch.bmm(alpha.transpose(0, 1), Wh.transpose(0, 1))  # [heads, N, F']
        h_prime = h_prime.transpose(0, 1)  # [N, heads, F']

        if self.concat:
            return h_prime.view(N, -1)  # [N, heads * F']
        else:
            return h_prime.mean(dim=1)  # [N, F']
```

### Sparse Implementation for Large Graphs

For large graphs, storing full N×N attention matrix is infeasible. Use sparse operations:

```python
def sparse_attention(edge_index, Wh, a):
    """
    edge_index: [2, E] tensor of edge indices
    Wh: [N, F'] transformed features
    a: [2*F'] attention vector
    """
    source, target = edge_index

    # Get features for edge endpoints
    Wh_i = Wh[source]  # [E, F']
    Wh_j = Wh[target]  # [E, F']

    # Compute attention scores for each edge
    concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [E, 2*F']
    e = F.leaky_relu(concat @ a, negative_slope=0.2)  # [E]

    # Sparse softmax (normalize per source node)
    alpha = sparse_softmax(e, source, num_nodes=Wh.size(0))  # [E]

    # Sparse aggregation
    h_prime = scatter_add(alpha.unsqueeze(-1) * Wh_j, source, dim=0)  # [N, F']

    return h_prime
```

---

## Why GAT Works: Key Insights

### 1. Content-Based Aggregation

GCN: "Average your neighbors (weighted by degree)"
GAT: "Selectively listen to relevant neighbors"

### 2. Implicit Feature Selection

High attention = important neighbor
Low attention ≈ 0 = neighbor effectively ignored

This is like **soft neighbor selection** without explicit thresholding.

### 3. Inductive Capability

The attention mechanism (W, a) is shared across all nodes and edges:
- Train on Graph A
- Apply to Graph B (completely unseen structure)
- Works because attention is computed from features, not memorized structure

### 4. Interpretability

Attention weights α_ij tell us:
- Which neighbors influenced each prediction
- What the model considers important
- Useful for debugging and understanding

---

## Limitations and Considerations

### 1. Over-Smoothing with Depth

Like GCNs, stacking many GAT layers causes over-smoothing:
```
Layer 1: Nodes aggregate 1-hop neighbors
Layer 2: Nodes aggregate 2-hop neighbors
...
Layer k: All nodes become similar (over-smoothed)
```

Solution: Skip connections, careful depth selection (typically 2-3 layers)

### 2. Computational Cost for Dense Graphs

```
Sparse graphs (social networks): |E| << N²  → GAT efficient
Dense graphs (fully connected): |E| ≈ N²   → GAT expensive

For dense graphs, consider approximations or sampling
```

### 3. Static Attention

GAT computes attention once per layer, not iteratively refined:
```
GAT: α_ij computed from input features only
vs.
Some methods: α_ij refined over multiple iterations
```

### 4. Edge Features Not Used

Original GAT only uses node features for attention:
```
e_ij = a^T [W·hᵢ || W·hⱼ]  ← Only node features!

What about edge features (relationship types, weights)?
Extensions like EGAT add: e_ij = a^T [W·hᵢ || W·hⱼ || W_e·eᵢⱼ]
```

---

## Summary: The GAT Layer

```
Input:  Node features h ∈ R^(N×F), Adjacency A ∈ R^(N×N)

┌─────────────────────────────────────────────────────────┐
│  1. Transform:    Wh = W · h                   [shared] │
│  2. Attention:    e_ij = LeakyReLU(a^T[Whᵢ||Whⱼ])      │
│  3. Mask:         e_ij = -∞ if (i,j) ∉ E              │
│  4. Normalize:    α_ij = softmax_j(e_ij)               │
│  5. Aggregate:    h'ᵢ = σ(Σⱼ α_ij · Whⱼ)              │
│  6. Multi-head:   h'ᵢ = ||_k σ(Σⱼ α^k_ij · W^k·hⱼ)    │
└─────────────────────────────────────────────────────────┘

Output: New node features h' ∈ R^(N×F')

Key properties:
- O(|V|FF' + |E|F') complexity
- Inductive (works on unseen graphs)
- Interpretable (attention weights)
- Permutation equivariant (order-independent)
```
