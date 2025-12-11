"""Hierarchical Graph Attention Network for within-ball attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .graph import NUM_NODES, NodeType


class NodeProjection(nn.Module):
    """Project heterogeneous node features to common dimension."""

    def __init__(self, node_dims: dict[int, int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict()

        for node_type, dim in node_dims.items():
            self.projections[str(node_type)] = nn.Linear(dim, hidden_dim)

    def forward(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """Project each node based on its type."""
        output = torch.zeros(x.shape[0], list(self.projections.values())[0].out_features)
        output = output.to(x.device)

        for node_type, proj in self.projections.items():
            mask = node_types == int(node_type)
            if mask.any():
                output[mask] = proj(x[mask])

        return output


class HierarchicalGATLayer(nn.Module):
    """Single GAT layer with attention extraction."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        # GATv2 for dynamic attention
        self.gat = GATv2Conv(
            in_dim,
            out_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with optional attention extraction.

        Returns:
            x: Updated node features
            attention: Attention weights if requested
        """
        if return_attention:
            x_new, (edge_idx, attention) = self.gat(
                x, edge_index, return_attention_weights=True
            )
        else:
            x_new = self.gat(x, edge_index)
            attention = None

        # Residual + norm
        if x.shape == x_new.shape:
            x_new = x_new + x
        x_new = self.norm(x_new)

        return x_new, attention


class HierarchicalGAT(nn.Module):
    """
    Hierarchical GAT for within-ball graph attention.

    Architecture:
    - Layer 1: Global context nodes update match state nodes
    - Layer 2: Match state nodes update actor nodes
    - Layer 3: Actor nodes update dynamics nodes
    - Layer 4: Final message passing across all nodes
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection for each node type
        from .graph import NODE_DIMS
        self.input_proj = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in NODE_DIMS.values()
        ])

        # GAT layers
        self.layers = nn.ModuleList([
            HierarchicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Forward pass through hierarchical GAT.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity
            batch: Batch indices for pooling
            return_attention: Whether to return attention weights

        Returns:
            output: Graph-level representation [batch_size, hidden_dim]
            attention_dict: Layer-wise attention if requested
        """
        attention_dict = {} if return_attention else None

        # Process through GAT layers
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, edge_index, return_attention)
            if return_attention and attn is not None:
                attention_dict[f"layer_{i}"] = attn

        # Global pooling
        if batch is not None:
            output = global_mean_pool(x, batch)
        else:
            output = x.mean(dim=0, keepdim=True)

        output = self.output_proj(output)

        return output, attention_dict

    def get_node_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, float]:
        """
        Get attention weights per node type for interpretability.

        Returns dict with node type names and their importance scores.
        """
        _, attention_dict = self.forward(x, edge_index, return_attention=True)

        if attention_dict is None:
            return {}

        # Aggregate attention by node type
        node_importance = {n.name.lower(): 0.0 for n in NodeType}

        for layer_name, attn in attention_dict.items():
            if attn is None:
                continue
            # attn shape: [num_edges, num_heads]
            attn_mean = attn.mean(dim=1)  # Average across heads

            # Sum attention received by each node
            edge_index_local = edge_index  # Assuming same edge_index
            for i, weight in enumerate(attn_mean):
                target_node = edge_index_local[1, i].item()
                if target_node < NUM_NODES:
                    node_type = NodeType(target_node)
                    node_importance[node_type.name.lower()] += weight.item()

        # Normalize
        total = sum(node_importance.values())
        if total > 0:
            node_importance = {k: v / total for k, v in node_importance.items()}

        return node_importance
