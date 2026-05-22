import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        adj: [N, N], binary or weighted adjacency matrix
        """
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh = adj @ x / deg
        out = self.self_lin(x) + self.neigh_lin(neigh)
        return F.relu(out)


class GNNOC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        bottleneck: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = SimpleGraphConv(input_dim, hidden_dim)
        self.conv2 = SimpleGraphConv(hidden_dim, bottleneck)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_nodes(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv1(node_feat, adj)
        h = self.dropout(h)
        z_node = self.conv2(h, adj)
        return z_node

    def graph_pool(self, z_node: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        if batch is None:
            return z_node.mean(dim=0, keepdim=True)

        num_graphs = int(batch.max().item()) + 1
        z_graph = []

        for gid in range(num_graphs):
            mask = batch == gid
            z_graph.append(z_node[mask].mean(dim=0))

        return torch.stack(z_graph, dim=0)

    def forward(
        self,
        node_feat: torch.Tensor,
        adj: torch.Tensor,
        batch: torch.Tensor = None,
    ):

        z_node = self.encode_nodes(node_feat, adj)
        z_graph = self.graph_pool(z_node, batch=batch)

        if batch is None:
            x_graph = node_feat.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(batch.max().item()) + 1
            x_graph = []
            for gid in range(num_graphs):
                mask = batch == gid
                x_graph.append(node_feat[mask].mean(dim=0))
            x_graph = torch.stack(x_graph, dim=0)

        recon = self.decoder(z_graph)
        s = F.mse_loss(recon, x_graph, reduction="none").mean(dim=-1)

        return recon, z_graph, s
