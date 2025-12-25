"""
Practical examples demonstrating the ProteinWorkshop featurization module.

This script provides hands-on examples for:
1. Basic featurization usage
2. Comparing different representations
3. Memory profiling
4. Performance optimization
5. Custom featurization pipelines
"""

import torch
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch


def example_1_basic_featurization():
    """Example 1: Basic featurization with minimal configuration."""
    print("\n" + "="*80)
    print("Example 1: Basic Featurization")
    print("="*80)
    
    # Create a simple featuriser
    featuriser = ProteinFeaturiser(
        representation="CA",
        scalar_node_features=["amino_acid_one_hot"],
        vector_node_features=[],
        edge_types=["knn_10"],
        scalar_edge_features=["edge_distance"],
        vector_edge_features=[],
    )
    
    # Create example batch
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    # Apply featurization
    print(f"\nInput batch:")
    print(f"  - Number of residues: {batch.coords.shape[0]}")
    print(f"  - Coordinate shape: {batch.coords.shape}")
    
    featurized = featuriser(batch)
    
    print(f"\nFeaturized batch:")
    print(f"  - Number of nodes: {featurized.num_nodes}")
    print(f"  - Node feature shape: {featurized.x.shape}")
    print(f"  - Number of edges: {featurized.edge_index.shape[1]}")
    print(f"  - Edge feature shape: {featurized.edge_attr.shape}")
    print(f"  - Node positions shape: {featurized.pos.shape}")
    
    return featurized


def example_2_representation_comparison():
    """Example 2: Compare different structural representations."""
    print("\n" + "="*80)
    print("Example 2: Representation Comparison (CA vs BB vs FA)")
    print("="*80)
    
    # Create example batch
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    representations = {
        "CA": "CA",
        "BB (Backbone)": "BB",
        # Note: FA (Full Atom) commented out as it requires more setup
        # "FA (Full Atom)": "FA",
    }
    
    results = {}
    
    for name, repr_type in representations.items():
        try:
            featuriser = ProteinFeaturiser(
                representation=repr_type,
                scalar_node_features=["amino_acid_one_hot"],
                vector_node_features=[],
                edge_types=["knn_10"],
                scalar_edge_features=["edge_distance"],
                vector_edge_features=[],
            )
            
            featurized = featuriser(batch.clone())
            
            results[name] = {
                'num_nodes': featurized.num_nodes,
                'num_edges': featurized.edge_index.shape[1],
                'node_feature_dim': featurized.x.shape[1],
            }
            
            print(f"\n{name}:")
            print(f"  - Nodes: {results[name]['num_nodes']}")
            print(f"  - Edges: {results[name]['num_edges']}")
            print(f"  - Node feature dim: {results[name]['node_feature_dim']}")
            
        except Exception as e:
            print(f"\n{name}: Failed - {str(e)}")
    
    return results


def example_3_feature_combinations():
    """Example 3: Different feature combinations."""
    print("\n" + "="*80)
    print("Example 3: Feature Combinations")
    print("="*80)
    
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    configs = {
        "Minimal": {
            "scalar_node_features": ["amino_acid_one_hot"],
            "vector_node_features": [],
        },
        "With Angles": {
            "scalar_node_features": ["amino_acid_one_hot", "dihedrals", "alpha", "kappa"],
            "vector_node_features": [],
        },
        "Equivariant": {
            "scalar_node_features": ["dihedrals"],
            "vector_node_features": ["orientation"],
        },
    }
    
    results = {}
    
    for name, config in configs.items():
        featuriser = ProteinFeaturiser(
            representation="CA",
            scalar_node_features=config["scalar_node_features"],
            vector_node_features=config["vector_node_features"],
            edge_types=["knn_16"],
            scalar_edge_features=["edge_distance"],
            vector_edge_features=["edge_vectors"] if config["vector_node_features"] else [],
        )
        
        featurized = featuriser(batch.clone())
        
        results[name] = {
            'node_feature_dim': featurized.x.shape[1],
            'has_vector_features': hasattr(featurized, 'x_vector_attr'),
        }
        
        print(f"\n{name}:")
        print(f"  - Scalar node features: {config['scalar_node_features']}")
        print(f"  - Vector node features: {config['vector_node_features']}")
        print(f"  - Total scalar feature dim: {results[name]['node_feature_dim']}")
        print(f"  - Has vector features: {results[name]['has_vector_features']}")
    
    return results


def example_4_edge_construction():
    """Example 4: Different edge construction strategies."""
    print("\n" + "="*80)
    print("Example 4: Edge Construction Strategies")
    print("="*80)
    
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    edge_configs = {
        "KNN-10": ["knn_10"],
        "KNN-30": ["knn_30"],
        "Epsilon-8": ["eps_8"],
        "Sequential Forward": ["seq_forward"],
        "Sequential Bidirectional": ["seq_forward", "seq_backward"],
        "Mixed (KNN + Sequential)": ["knn_16", "seq_forward"],
    }
    
    results = {}
    
    for name, edge_types in edge_configs.items():
        try:
            featuriser = ProteinFeaturiser(
                representation="CA",
                scalar_node_features=["amino_acid_one_hot"],
                vector_node_features=[],
                edge_types=edge_types,
                scalar_edge_features=["edge_distance"],
                vector_edge_features=[],
            )
            
            featurized = featuriser(batch.clone())
            
            num_edges = featurized.edge_index.shape[1]
            avg_degree = num_edges / featurized.num_nodes
            
            results[name] = {
                'num_edges': num_edges,
                'avg_degree': avg_degree,
            }
            
            print(f"\n{name}:")
            print(f"  - Edge types: {edge_types}")
            print(f"  - Number of edges: {num_edges}")
            print(f"  - Average degree: {avg_degree:.2f}")
            
        except Exception as e:
            print(f"\n{name}: Failed - {str(e)}")
    
    return results


def example_5_memory_profiling():
    """Example 5: Memory profiling of different configurations."""
    print("\n" + "="*80)
    print("Example 5: Memory Profiling")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory profiling.")
        return None
    
    device = torch.device('cuda')
    batch = create_example_batch().to(device)
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long, device=device)
    
    configs = {
        "Minimal (CA, KNN-10)": {
            "representation": "CA",
            "scalar_node_features": ["amino_acid_one_hot"],
            "edge_types": ["knn_10"],
        },
        "Full Features (CA, KNN-30)": {
            "representation": "CA",
            "scalar_node_features": ["amino_acid_one_hot", "dihedrals", "alpha", "kappa"],
            "edge_types": ["knn_30"],
        },
        "Backbone (BB, KNN-10)": {
            "representation": "BB",
            "scalar_node_features": ["amino_acid_one_hot"],
            "edge_types": ["knn_10"],
        },
    }
    
    results = {}
    
    for name, config in configs.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        featuriser = ProteinFeaturiser(
            representation=config["representation"],
            scalar_node_features=config["scalar_node_features"],
            vector_node_features=[],
            edge_types=config["edge_types"],
            scalar_edge_features=["edge_distance"],
            vector_edge_features=[],
        ).to(device)
        
        featurized = featuriser(batch.clone())
        torch.cuda.synchronize()
        
        mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        results[name] = {
            'memory_used_mb': mem_after - mem_before,
            'memory_peak_mb': mem_peak,
        }
        
        print(f"\n{name}:")
        print(f"  - Memory used: {results[name]['memory_used_mb']:.2f} MB")
        print(f"  - Peak memory: {results[name]['memory_peak_mb']:.2f} MB")
        
        # Cleanup
        del featurised
        torch.cuda.empty_cache()
    
    return results


def example_6_performance_benchmarking():
    """Example 6: Performance benchmarking."""
    print("\n" + "="*80)
    print("Example 6: Performance Benchmarking")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch = create_example_batch().to(device)
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long, device=device)
    
    configs = {
        "Minimal": {
            "representation": "CA",
            "scalar_node_features": ["amino_acid_one_hot"],
            "edge_types": ["knn_10"],
        },
        "Standard": {
            "representation": "CA",
            "scalar_node_features": ["amino_acid_one_hot", "dihedrals"],
            "edge_types": ["knn_16"],
        },
        "Full": {
            "representation": "CA",
            "scalar_node_features": ["amino_acid_one_hot", "dihedrals", "alpha", "kappa"],
            "edge_types": ["knn_30"],
        },
    }
    
    num_iterations = 50
    results = {}
    
    for name, config in configs.items():
        featuriser = ProteinFeaturiser(
            representation=config["representation"],
            scalar_node_features=config["scalar_node_features"],
            vector_node_features=[],
            edge_types=config["edge_types"],
            scalar_edge_features=["edge_distance"],
            vector_edge_features=[],
        ).to(device)
        
        # Warmup
        for _ in range(5):
            _ = featuriser(batch.clone())
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = featuriser(batch.clone())
            if device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        time_per_iter = (elapsed / num_iterations) * 1000  # ms
        throughput = num_iterations / elapsed
        
        results[name] = {
            'time_per_iter_ms': time_per_iter,
            'throughput': throughput,
        }
        
        print(f"\n{name}:")
        print(f"  - Time per iteration: {time_per_iter:.2f} ms")
        print(f"  - Throughput: {throughput:.2f} batches/sec")
    
    return results


def example_7_custom_featurization_pipeline():
    """Example 7: Creating a custom featurization pipeline."""
    print("\n" + "="*80)
    print("Example 7: Custom Featurization Pipeline")
    print("="*80)
    
    class CustomFeaturiser(torch.nn.Module):
        """Custom featuriser with progressive complexity."""
        
        def __init__(self):
            super().__init__()
            self.coarse_featuriser = ProteinFeaturiser(
                representation="CA",
                scalar_node_features=["amino_acid_one_hot"],
                vector_node_features=[],
                edge_types=["knn_10"],
                scalar_edge_features=["edge_distance"],
                vector_edge_features=[],
            )
            
            self.fine_featuriser = ProteinFeaturiser(
                representation="CA",
                scalar_node_features=["amino_acid_one_hot", "dihedrals", "alpha"],
                vector_node_features=[],
                edge_types=["knn_30"],
                scalar_edge_features=["edge_distance", "edge_type"],
                vector_edge_features=[],
            )
        
        def forward(self, batch, mode='coarse'):
            """Apply featurization based on mode."""
            if mode == 'coarse':
                return self.coarse_featuriser(batch)
            else:
                return self.fine_featuriser(batch)
    
    # Create custom featuriser
    custom_featuriser = CustomFeaturiser()
    
    # Create batch
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    # Apply coarse featurization
    coarse_result = custom_featuriser(batch.clone(), mode='coarse')
    print("\nCoarse mode:")
    print(f"  - Nodes: {coarse_result.num_nodes}")
    print(f"  - Edges: {coarse_result.edge_index.shape[1]}")
    print(f"  - Feature dim: {coarse_result.x.shape[1]}")
    
    # Apply fine featurization
    fine_result = custom_featuriser(batch.clone(), mode='fine')
    print("\nFine mode:")
    print(f"  - Nodes: {fine_result.num_nodes}")
    print(f"  - Edges: {fine_result.edge_index.shape[1]}")
    print(f"  - Feature dim: {fine_result.x.shape[1]}")
    
    print("\nUse case: Start training with coarse mode for quick convergence,")
    print("then switch to fine mode for better final performance.")
    
    return custom_featuriser


def example_8_batch_processing():
    """Example 8: Efficient batch processing."""
    print("\n" + "="*80)
    print("Example 8: Efficient Batch Processing")
    print("="*80)
    
    from torch_geometric.data import Batch as PyGBatch
    
    # Create multiple protein graphs
    graphs = [create_example_batch() for _ in range(4)]
    for g in graphs:
        g.seq_pos = torch.arange(g.coords.shape[0], dtype=torch.long)
    
    # Batch them
    batched = PyGBatch.from_data_list(graphs)
    
    print(f"\nProcessing batch of {len(graphs)} proteins:")
    print(f"  - Total residues: {batched.coords.shape[0]}")
    print(f"  - Batch indices shape: {batched.batch.shape}")
    
    # Apply featurization to entire batch at once
    featuriser = ProteinFeaturiser(
        representation="CA",
        scalar_node_features=["amino_acid_one_hot", "dihedrals"],
        vector_node_features=[],
        edge_types=["knn_16"],
        scalar_edge_features=["edge_distance"],
        vector_edge_features=[],
    )
    
    featurized_batch = featuriser(batched)
    
    print(f"\nFeaturized batch:")
    print(f"  - Total nodes: {featurized_batch.num_nodes}")
    print(f"  - Total edges: {featurized_batch.edge_index.shape[1]}")
    print(f"  - Node features shape: {featurized_batch.x.shape}")
    print(f"  - Maintains batch structure: {hasattr(featurized_batch, 'batch')}")
    
    print("\nNote: Batching allows efficient parallel processing of multiple proteins!")
    
    return featurized_batch


def run_all_examples():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# ProteinWorkshop Featurization Examples")
    print("#"*80)
    
    try:
        example_1_basic_featurization()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_representation_comparison()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_feature_combinations()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_edge_construction()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_memory_profiling()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_performance_benchmarking()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    try:
        example_7_custom_featurization_pipeline()
    except Exception as e:
        print(f"Example 7 failed: {e}")
    
    try:
        example_8_batch_processing()
    except Exception as e:
        print(f"Example 8 failed: {e}")
    
    print("\n" + "#"*80)
    print("# All examples completed!")
    print("#"*80)


if __name__ == "__main__":
    # Run individual examples
    # example_1_basic_featurization()
    # example_2_representation_comparison()
    # example_3_feature_combinations()
    # example_4_edge_construction()
    # example_5_memory_profiling()
    # example_6_performance_benchmarking()
    # example_7_custom_featurization_pipeline()
    # example_8_batch_processing()
    
    # Or run all examples
    run_all_examples()
