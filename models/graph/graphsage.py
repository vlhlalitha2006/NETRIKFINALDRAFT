from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


HIDDEN_DIM = 32
OUTPUT_DIM = 32
DROPOUT = 0.2


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, HIDDEN_DIM)
        self.conv2 = SAGEConv(HIDDEN_DIM, OUTPUT_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        return x


class GraphSAGENodeClassifier(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim=in_dim)
        self.classifier = nn.Linear(OUTPUT_DIM, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x, edge_index)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits, embeddings
