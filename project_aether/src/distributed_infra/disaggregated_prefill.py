"""
Project AETHER: Disaggregated Serving - Prefill Node
Optimized for High-Compute H100 Clusters.

In a traditional LLM or Transformer setup, the "prefill" phase (processing the prompt) 
is highly compute-bound, while the "decode" phase (autoregressive generation) is 
memory-bandwidth bound. 

Project AETHER adapts this to computational physics:
1. Prefill Nodes (H100s): Run the heavy JAX/Triton physics engines (CFD, DSMC) 
   to simulate the initial flight conditions (the "prompt"). They tokenize this 
   state into a KV-cache.
2. Decode Nodes (A100s/L40s): Receive the KV-cache over the network and run the 
   quantized FP8 surrogate model to predict the future flight envelope in real-time.

This module implements the Prefill logic, responsible for the heavy physics compute
and the asynchronous transmission of the Physics KV-Cache to the Decode cluster.
"""

import torch
import torch.distributed as dist
import time
from typing import Dict, List, Tuple

# Import the tokenizer/encoder and physics modules (paths relative to the ideal directory)
try:
    from src.surrogate_models.pit_graph_encoder import GraphTokenEncoder
    from src.surrogate_models.pit_star_attention import PhysicsInformedTransformerLayer
except ImportError:
    print("Warning: Surrogate model modules not found. Using mock layers for prefill.")
    GraphTokenEncoder = None
    PhysicsInformedTransformerLayer = None

class DisaggregatedPrefillNode:
    """
    Manages the compute-heavy simulation phase and network transmission of the 
    physical KV cache.
    """
    def __init__(self, decode_rank_mapping: Dict[int, int], embed_dim: int = 1024):
        """
        Args:
            decode_rank_mapping: Dictionary mapping local Prefill GPU ranks to 
                                 their corresponding Decode GPU ranks over the network.
            embed_dim: Dimension of the Transformer latent space.
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for disaggregated serving.")
            
        self.rank = dist.get_rank()
        self.decode_targets = decode_rank_mapping
        self.embed_dim = embed_dim
        
        # Initialize the Graph Tokenizer
        if GraphTokenEncoder is not None:
            # 6 physical variables, 4 geometric edge features
            self.tokenizer = GraphTokenEncoder(input_node_dim=6, input_edge_dim=4, embed_dim=embed_dim).cuda()
            self.transformer_layer = PhysicsInformedTransformerLayer(hidden_dim=embed_dim, ffn_dim=embed_dim*4).cuda()
        else:
            self.tokenizer = None
            self.transformer_layer = None
            
        # Asynchronous network request handles
        self.active_transfers = []

    def _simulate_heavy_physics(self, num_initial_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the JAX/Triton solvers (CFD + DSMC + Plasma) to generate the ground-truth
        initial state of the hypersonic shock layer.
        
        Returns:
            nodes: [num_nodes, 6] physical variables.
            edges: [num_edges, 4] geometric properties.
            connectivity: [2, num_edges] graph structure.
        """
        # In a full deployment, this would invoke `src/physics_engine/ablation_monolithic_solver.py`
        # and `src/physics_engine/dsmc_boltzmann.py`.
        print(f"[Prefill Rank {self.rank}] Executing high-fidelity JAX physics engine for {num_initial_steps} steps...")
        time.sleep(1.5) # Mocking heavy compute latency
        
        # Mock resulting mesh state
        num_nodes = 50_000
        num_edges = 200_000
        nodes = torch.randn(num_nodes, 6, device='cuda')
        edges = torch.randn(num_edges, 4, device='cuda')
        connectivity = torch.randint(0, num_nodes, (2, num_edges), device='cuda')
        
        return nodes, edges, connectivity

    def compute_kv_cache(self, nodes: torch.Tensor, edges: torch.Tensor, connectivity: torch.Tensor) -> torch.Tensor:
        """
        Converts the raw physical mesh into deep latent tokens and performs the 
        initial Transformer forward pass to generate the Key/Value cache.
        """
        print(f"[Prefill Rank {self.rank}] Tokenizing unstructured mesh state...")
        
        if self.tokenizer is not None:
            # Generate the latent sequence: [1, SeqLen, EmbedDim]
            tokens = self.tokenizer(nodes, connectivity, edges)
            
            # Perform the Prefill forward pass through the first Transformer layer
            # In a real model, this extracts the intermediate K and V tensors from the attention blocks
            # For demonstration, we use the output state as the combined "context cache"
            prefill_context = self.transformer_layer(tokens)
        else:
            # Mock the KV cache tensor if imports are missing
            prefill_context = torch.randn(1, nodes.shape[0], self.embed_dim, device='cuda')
            
        return prefill_context

    def transmit_to_decode_cluster(self, kv_cache: torch.Tensor):
        """
        Asynchronously streams the computed KV cache over InfiniBand/RoCE to the 
        memory-bound Decode nodes. Uses `isend` so the H100 can immediately start 
        simulating the next ensemble member.
        """
        target_decode_rank = self.decode_targets.get(self.rank)
        
        if target_decode_rank is None:
            print(f"[Prefill Rank {self.rank}] No mapped Decode node. Skipping transmission.")
            return

        print(f"[Prefill Rank {self.rank}] Streaming {kv_cache.nelement() * kv_cache.element_size() / 1e6:.2f} MB "
              f"KV-Cache to Decode Node {target_decode_rank}...")
        
        # Clean up finished transfers
        self.active_transfers = [req for req in self.active_transfers if not req.is_completed()]
        
        # Non-blocking send (isend)
        # This allows the H100 to immediately return to doing heavy math while the NIC 
        # handles the data transfer in the background via DMA.
        req = dist.isend(tensor=kv_cache, dst=target_decode_rank)
        self.active_transfers.append(req)

    def run_prefill_pipeline(self, num_initial_steps: int = 100):
        """
        Main execution pipeline for the Prefill node.
        """
        # 1. Compute the physical truth
        nodes, edges, connectivity = self._simulate_heavy_physics(num_initial_steps)
        
        # 2. Tokenize and compute Transformer context
        kv_cache = self.compute_kv_cache(nodes, edges, connectivity)
        
        # 3. Offload memory to the predictor cluster
        self.transmit_to_decode_cluster(kv_cache)
        
        print(f"[Prefill Rank {self.rank}] Pipeline complete. Ready for next simulation phase.")


# --- Diagnostic / Testing Block ---
if __name__ == "__main__":
    import os
    
    # Mock distributed environment setup for local testing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '2' # 1 Prefill, 1 Decode
    
    try:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        
        # Map Prefill Rank 0 -> Decode Rank 1
        decode_map = {0: 1}
        
        prefill_node = DisaggregatedPrefillNode(decode_rank_mapping=decode_map)
        prefill_node.run_prefill_pipeline()
        
        dist.destroy_process_group()
    except Exception as e:
        print(f"Distributed test skipped/failed: {e}")
        print("To run this properly, use: torchrun --nproc_per_node=2 ...")