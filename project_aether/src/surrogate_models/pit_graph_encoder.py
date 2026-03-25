"""
Project AETHER: Physics-Informed Transformer (PIT) Graph Encoder
Optimized for Unstructured Hypersonic Meshes.

Standard vision transformers extract patches from structured 2D grids. In CFD and 
DSMC, the data lives on highly irregular, unstructured volumetric meshes. This module 
implements a Graph Neural Network (GNN) based tokenizer. 

It uses a Message Passing Neural Network (MPNN) to aggregate local flow gradients, 
pressure potentials, and boundary proximity into high-dimensional latent vectors. 
These node embeddings become the "tokens" that are subsequently fed into the 
FP8 Transformer for sequence modeling and temporal prediction.
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn.models import MLP
except ImportError:
    print("Warning: torch_geometric is required for the PIT Graph Encoder.")
    # Mocking classes for environments without torch_geometric installed
    class MessagePassing(nn.Module):
        def __init__(self, aggr='mean'): super().__init__()
    class MLP(nn.Module):
        def __init__(self, channel_list, dropout=0.0): super().__init__()

class FluidDynamicsMessagePassing(MessagePassing):
    """
    Custom Message Passing layer tailored for fluid dynamics on unstructured grids.
    Calculates interactions between adjacent fluid cells/particles, effectively 
    learning the local spatial gradients and flux exchanges before global attention.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        # We use 'mean' aggregation to be invariant to the number of neighbors 
        # (cell faces vary in unstructured meshes). 'add' can cause overflow in dense regions.
        super().__init__(aggr='mean')
        
        # Message function: Learns how much information flows from node j to node i
        # based on their states and the geometric geometry of the face connecting them.
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(), # Swish activation, common in physics-informed networks for smooth 2nd derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Update function: Updates the cell's state based on its previous state 
        # and the incoming aggregated messages from its neighbors.
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Starts the message passing process.
        x: [num_nodes, node_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        # propagate internally calls self.message() and self.update()
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages from node j to node i.
        x_i: Target node features
        x_j: Source node features
        edge_attr: Edge features (e.g., face area, normal vector, distance)
        """
        # Concatenate target state, source state, and geometric edge features
        msg_inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_inputs)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Updates node embeddings using the aggregated messages.
        aggr_out: Aggregated messages from neighbors
        x: Current node features
        """
        # Combine current state with neighborhood flux information
        update_inputs = torch.cat([x, aggr_out], dim=-1)
        # Residual connection to preserve original state information
        return x + self.update_mlp(update_inputs)

class GraphTokenEncoder(nn.Module):
    """
    The complete Graph-to-Sequence Tokenizer for the Physics-Informed Transformer.
    Converts physical variables defined on an unstructured mesh into deep latent 
    tokens ready for global Star Attention.
    """
    def __init__(self, input_node_dim: int, input_edge_dim: int, embed_dim: int, num_mpnn_layers: int = 3):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Initial projection of physical variables (rho, u, v, w, p, T) to latent space
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # Initial projection of edge geometry (dx, dy, dz, normal_x, normal_y, normal_z)
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # Stack of Message Passing layers to build local context (receptive field)
        self.mpnn_layers = nn.ModuleList([
            FluidDynamicsMessagePassing(node_dim=embed_dim, edge_dim=embed_dim, hidden_dim=embed_dim)
            for _ in range(num_mpnn_layers)
        ])
        
        # Final LayerNorm before feeding into the FP8 Transformer
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Physical state vector at each mesh node [num_nodes, input_node_dim].
            edge_index: Connectivity graph of the mesh [2, num_edges].
            edge_attr: Geometric properties of the mesh faces/edges [num_edges, input_edge_dim].
            
        Returns:
            tokens: Latent sequence embeddings for the Transformer [1, num_nodes, embed_dim].
        """
        # 1. Project physical variables into latent dimensions
        h_nodes = self.node_encoder(x)
        h_edges = self.edge_encoder(edge_attr)
        
        # 2. Extract local spatial context via Message Passing (Graph Convolutions)
        # This acts similarly to a CNN extracting local features before a ViT, 
        # but operates strictly on the unstructured mesh topology.
        for mpnn in self.mpnn_layers:
            h_nodes = mpnn(h_nodes, edge_index, h_edges)
            
        h_nodes = self.final_norm(h_nodes)
        
        # 3. Add batch dimension to conform to standard Transformer inputs [Batch, Sequence, Dim]
        # In this context, num_nodes = Sequence Length
        tokens = h_nodes.unsqueeze(0) 
        
        return tokens

# --- Diagnostic / Testing Block ---
if __name__ == "__main__":
    # Mocking a small fluid domain (1000 nodes, 4000 edges)
    num_nodes = 1000
    num_edges = 4000
    
    # 6 physical variables: [rho, u, v, w, p, T]
    mock_nodes = torch.randn(num_nodes, 6) 
    
    # 4 geometric variables: [dx, dy, dz, face_area]
    mock_edges = torch.randn(num_edges, 4) 
    
    # Random connectivity
    mock_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    print(f"Initializing Graph Token Encoder for Unstructured Mesh...")
    encoder = GraphTokenEncoder(input_node_dim=6, input_edge_dim=4, embed_dim=1024, num_mpnn_layers=3)
    
    print("Tokenizing flow field...")
    transformer_tokens = encoder(mock_nodes, mock_edge_index, mock_edges)
    
    print(f"Output Tokens Shape (Ready for PIT): {transformer_tokens.shape}")
    # Expected: [1, 1000, 1024] -> [Batch, Seq_Len, Hidden_Dim]