"""ProNet encoder for backbone-level protein representations.

This module implements ProNet (from DIVE) adapted for ProteinWorkshop's framework.
Uses CA, N, C atom coordinates to compute geometric features with spherical harmonics.
"""
from typing import Set, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_scatter import scatter

from proteinworkshop.models.graph_encoders.layers.pronet import (
    InteractionBlock,
    Linear,
)
from proteinworkshop.models.graph_encoders.layers.pronet_features import (
    AngleEmbedding,
    ThetaPhiEmbedding,
)
from proteinworkshop.custom_types import EncoderOutput


class ProNetModel(nn.Module):
    """ProNet encoder for backbone-level protein representations.
    
    Uses CA, N, C atom coordinates to compute:
    - Geometric features (theta, phi angles)
    - Euler angles between local frames
    - Spherical harmonic encodings
    
    Leverages ProteinWorkshop's existing node features (amino acid type, dihedrals, sidechain_torsions).
    
    Args:
        num_blocks: Number of interaction blocks
        hidden_channels: Hidden embedding dimension
        mid_emb: Intermediate embedding dimension for geometric features
        num_radial: Number of radial basis functions
        num_spherical: Number of spherical harmonic degrees
        cutoff: Cutoff distance for normalization (not used for edges, only for feature scaling)
        int_emb_layers: Number of layers in each interaction block
        out_layers: Number of output layers
        num_pos_emb: Dimension of positional embeddings
        dropout: Dropout probability
        pool: Pooling method for graph-level embeddings ('sum', 'mean', 'max')
        data_augment_eachlayer: If True, add Gaussian noise to node features
        euler_noise: If True, add noise to Euler angles
    """
    
    def __init__(
        self,
        num_blocks: int = 4,
        hidden_channels: int = 128,
        mid_emb: int = 64,
        num_radial: int = 6,
        num_spherical: int = 2,
        cutoff: float = 10.0,
        int_emb_layers: int = 3,
        out_layers: int = 2,
        num_pos_emb: int = 16,
        dropout: float = 0.0,
        pool: str = "sum",
        data_augment_eachlayer: bool = False,
        euler_noise: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_pos_emb = num_pos_emb
        self.hidden_channels = hidden_channels
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.pool = pool

        # Geometric feature encoders (ProNet-specific)
        self.theta_phi_emb = ThetaPhiEmbedding(num_radial, num_spherical, cutoff)
        self.angle_emb = AngleEmbedding(num_radial, num_spherical, cutoff)

        # Node feature embedding
        # Input: amino_acid_one_hot (20) + dihedrals (6) + sidechain_torsions (8)
        # Total: 34 node features
        # Use LazyLinear to automatically infer input dimension from batch.x
        self.embedding = nn.LazyLinear(hidden_channels)

        # Interaction blocks
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=F.silu,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output layers
        self.lins_out = nn.ModuleList()
        for _ in range(out_layers - 1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, hidden_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        if hasattr(self.embedding, "reset_parameters"):
            self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for ProNet backbone encoder.
        
        Returns:
            Set of required attribute names:
            - x: Node features (amino acid one-hot + dihedrals + sidechain_torsions)
            - pos: CA coordinates (automatically set from batch.coords[:, 1, :])
            - coords: AtomTensor with shape (num_residues, 37, 3) for extracting N, C
            - edge_index: Pre-computed edges (from ProteinWorkshop feature config)
            - batch: Batch assignment vector
        """
        return {"x", "pos", "coords", "edge_index", "batch"}

    def _compute_positional_embedding(
        self, edge_index: torch.Tensor, num_pos_emb: int = 16
    ) -> torch.Tensor:
        """Compute sinusoidal positional embeddings from sequence distance.
        
        From https://github.com/jingraham/neurips19-graph-protein-design
        
        Args:
            edge_index: Edge indices (2, num_edges)
            num_pos_emb: Dimension of positional embedding
            
        Returns:
            Positional embeddings (num_edges, num_pos_emb)
        """
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(
                0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device
            )
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _compute_theta_phi(
        self, pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute theta and phi angles for edges.
        
        Theta: Angle between edge vector and previous node
        Phi: Dihedral angle involving the edge
        
        Args:
            pos: Node positions (num_nodes, 3)
            edge_index: Edge indices (2, num_edges)
            num_nodes: Number of nodes
            
        Returns:
            Tuple of (theta, phi) angles in radians
        """
        j, i = edge_index

        # Reference nodes for angle calculation
        refi0 = (i - 1) % num_nodes
        refi1 = (i + 1) % num_nodes

        # Theta: angle between (j-i) and (refi0-i)
        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        # Phi: dihedral angle
        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(
            dim=-1
        ) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)

        return theta, phi

    def _compute_euler_angles(
        self,
        pos: torch.Tensor,
        pos_n: torch.Tensor,
        pos_c: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Euler angles between local coordinate frames.
        
        Builds local frames from (CA, N, C) atoms and computes rotation between them.
        
        Args:
            pos: CA coordinates (num_nodes, 3)
            pos_n: N coordinates (num_nodes, 3)
            pos_c: C coordinates (num_nodes, 3)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Tuple of three Euler angles (angle1, angle2, angle3)
        """
        j, i = edge_index
        device = pos.device

        # Build local coordinate frame for node i
        Or1_x = pos_n[i] - pos[i]
        Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
        Or1_z_length = Or1_z.norm(dim=1) + 1e-7

        # Build local coordinate frame for node j
        Or2_x = pos_n[j] - pos[j]
        Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
        Or2_z_length = Or2_z.norm(dim=1) + 1e-7

        # Compute rotation between frames
        Or1_Or2_N = torch.cross(Or1_z, Or2_z)

        angle1 = torch.atan2(
            (torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1) / Or1_z_length,
            (Or1_x * Or1_Or2_N).sum(dim=-1),
        )
        angle2 = torch.atan2(
            torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1)
        )
        angle3 = torch.atan2(
            (torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1) / Or2_z_length,
            (Or1_Or2_N * Or2_x).sum(dim=-1),
        )

        # Optional: Add noise for data augmentation
        if self.euler_noise:
            euler_noise = torch.clip(
                torch.empty(3, len(angle1)).to(device).normal_(mean=0.0, std=0.025),
                min=-0.1,
                max=0.1,
            )
            angle1 += euler_noise[0]
            angle2 += euler_noise[1]
            angle3 += euler_noise[2]

        return angle1, angle2, angle3

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Forward pass for backbone-level ProNet.
        
        Workflow:
        1. Use ProteinWorkshop's pre-computed node features (batch.x)
        2. Extract CA, N, C coordinates from batch.coords
        3. Use pre-computed edges from batch.edge_index
        4. Compute ProNet-specific geometric features
        5. Apply interaction blocks
        6. Pool to graph level
        
        Args:
            batch: Protein batch with required attributes
            
        Returns:
            EncoderOutput containing node and graph embeddings
        """
        # Extract coordinates and batch info
        pos = batch.pos  # CA coordinates (num_nodes, 3)
        # Extract N and C coordinates from batch.coords
        # AtomTensor shape: (num_residues, 37, 3)
        # Atom ordering: 0=N, 1=CA, 2=C, 3=O
        pos_n = batch.coords[:, 0, :]  # N coordinates (num_nodes, 3)
        pos_c = batch.coords[:, 2, :]  # C coordinates (num_nodes, 3)
        batch_idx = batch.batch  # Batch assignment (num_nodes,)
        device = pos.device

        # Use ProteinWorkshop's node features directly
        # batch.x already contains: amino_acid_one_hot + dihedrals + sidechain_torsions
        x = self.embedding(batch.x)  # (num_nodes, hidden_channels)

        # Use pre-computed edges from ProteinWorkshop (e.g., knn_30)
        edge_index = batch.edge_index  # (2, num_edges)
        j, i = edge_index  # Source and target nodes

        # Compute distances
        dist = (pos[i] - pos[j]).norm(dim=1)  # (num_edges,)

        # Compute geometric features (ProNet-specific)
        pos_emb = self._compute_positional_embedding(edge_index, self.num_pos_emb)
        theta, phi = self._compute_theta_phi(pos, edge_index, len(pos))

        # Encode (dist, theta, phi) with spherical harmonics
        feature0 = self.theta_phi_emb(dist, theta, phi)

        # Compute Euler angles between local frames
        angle1, angle2, angle3 = self._compute_euler_angles(
            pos, pos_n, pos_c, edge_index
        )

        # Encode three Euler angles
        feature1 = torch.cat(
            [
                self.angle_emb(dist, angle1),
                self.angle_emb(dist, angle2),
                self.angle_emb(dist, angle3),
            ],
            dim=1,
        )

        # Apply interaction blocks
        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                # Add Gaussian noise to features for augmentation
                gaussian_noise = torch.clip(
                    torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025),
                    min=-0.1,
                    max=0.1,
                )
                x += gaussian_noise
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch_idx)

        # Pool to graph level
        if self.pool == "sum":
            graph_embedding = scatter(x, batch_idx, dim=0, reduce="sum")
        elif self.pool == "mean":
            graph_embedding = scatter(x, batch_idx, dim=0, reduce="mean")
        elif self.pool == "max":
            graph_embedding = scatter(x, batch_idx, dim=0, reduce="max")
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool}")

        # Apply output layers
        for lin in self.lins_out:
            graph_embedding = self.relu(lin(graph_embedding))
            graph_embedding = self.dropout(graph_embedding)
        graph_embedding = self.lin_out(graph_embedding)

        return EncoderOutput(
            {
                "node_embedding": x,
                "graph_embedding": graph_embedding,
            }
        )

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
