"""ProNet interaction layers.

This module implements the core layers for ProNet including custom convolution
and interaction blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, inits
from torch_sparse import matmul


class Linear(nn.Module):
    """Linear layer with configurable weight initialization.
    
    Similar to PyG's Linear but with control over initialization method.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        bias: Whether to include bias term
        weight_initializer: Initialization method ('glorot' or 'zeros')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: str = "glorot",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == "glorot":
            inits.glorot(self.weight)
        elif self.weight_initializer == "zeros":
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class TwoLinear(nn.Module):
    """Two stacked linear layers with optional activation.
    
    Args:
        in_channels: Input feature dimension
        middle_channels: Hidden feature dimension
        out_channels: Output feature dimension
        bias: Whether to include bias terms
        act: Whether to apply activation functions (SiLU/Swish)
    """
    
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        bias: bool = False,
        act: bool = False,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = F.silu(x)
        x = self.lin2(x)
        if self.act:
            x = F.silu(x)
        return x


class EdgeGraphConv(MessagePassing):
    """Graph convolution with Hadamard product between node and edge features.
    
    Similar to PyG's GraphConv but performs element-wise multiplication
    between node features and edge features during message passing.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge features (num_edges, in_channels)
            size: Size of the output (optional)
            
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        """Message function: element-wise multiplication."""
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        """Combined message and aggregation for efficiency."""
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(nn.Module):
    """ProNet interaction block with three parallel message passing streams.
    
    Stream 0: (dist, theta, phi) geometric embeddings
    Stream 1: Euler angle embeddings (3 angles for backbone)
    Stream 2: Sequence positional embeddings
    
    Args:
        hidden_channels: Hidden feature dimension
        output_channels: Output feature dimension
        num_radial: Number of radial basis functions
        num_spherical: Number of spherical harmonic degrees
        num_layers: Number of refinement layers
        mid_emb: Middle embedding dimension for feature projection
        act: Activation function (default: F.silu)
        num_pos_emb: Dimension of positional embeddings
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        num_radial: int,
        num_spherical: int,
        num_layers: int,
        mid_emb: int,
        act=F.silu,
        num_pos_emb: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        # Three parallel EdgeGraphConv layers
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        # Feature projection layers
        self.lin_feature0 = TwoLinear(
            num_radial * num_spherical ** 2, mid_emb, hidden_channels
        )
        # For backbone: 3 Euler angles
        self.lin_feature1 = TwoLinear(
            3 * num_radial * num_spherical, mid_emb, hidden_channels
        )
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        # Node feature processing
        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        # Output processing per stream
        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        # Concatenation and refinement layers
        self.lins_cat = nn.ModuleList()
        self.lins_cat.append(Linear(3 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        """Forward pass through interaction block.
        
        Args:
            x: Node features (num_nodes, hidden_channels)
            feature0: (dist, theta, phi) embeddings (num_edges, num_radial * num_spherical^2)
            feature1: Euler angle embeddings (num_edges, 3 * num_radial * num_spherical)
            pos_emb: Positional embeddings (num_edges, num_pos_emb)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes,)
            
        Returns:
            Updated node features (num_nodes, hidden_channels)
        """
        # Process node features
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))

        # Stream 0: (dist, theta, phi) geometric features
        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        # Stream 1: Euler angle features
        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        # Stream 2: Positional features
        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        # Concatenate three streams
        h = torch.cat((h0, h1, h2), 1)
        
        # Refine concatenated features
        for lin in self.lins_cat:
            h = self.act(lin(h))

        # Residual connection
        h = h + x_lin_2

        # Final refinement
        for lin in self.lins:
            h = self.act(lin(h))
        h = self.final(h)
        
        return h
