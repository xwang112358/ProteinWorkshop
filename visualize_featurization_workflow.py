"""
Visualization script for the ProteinWorkshop featurization workflow.

This script creates visual diagrams to help understand the featurization pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_workflow_diagram():
    """Create a visual workflow diagram of the featurization pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'ProteinWorkshop Featurization Workflow', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Define colors
    color_input = '#e8f4f8'
    color_node = '#b8e6f0'
    color_repr = '#90d5e8'
    color_edge = '#68c4e0'
    color_output = '#40b3d8'
    
    # Stage 1: Input
    box1 = FancyBboxPatch((0.5, 9.5), 9, 1, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 10, 'Raw Protein Data\n(coords: N×37×3, residue_type, batch, ...)', 
            ha='center', va='center', fontsize=10)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((5, 9.3), (5, 8.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Stage 2: Scalar Node Features
    box2 = FancyBboxPatch((0.5, 7.2), 9, 1.4, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_node, linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 8.3, 'Step 1: Compute Scalar Node Features', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 7.9, 'amino_acid_one_hot (23D) | dihedrals (6D) | alpha (2D)', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 7.5, 'kappa (2D) | sidechain_torsions (8D) | positional_encoding (16D)', 
            ha='center', va='center', fontsize=9)
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5, 7.0), (5, 6.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Stage 3: Representation Transform
    box3 = FancyBboxPatch((0.5, 5.0), 9, 1.3, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_repr, linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 6.0, 'Step 2: Transform Representation', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 5.6, 'CA (1 node/residue) | BB (4 nodes/residue) | FA (all atoms)', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 5.25, 'Extract/expand coordinates → Update pos attribute', 
            ha='center', va='center', fontsize=9)
    
    # Arrow 3
    arrow3 = FancyArrowPatch((5, 4.8), (5, 4.3),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Stage 4: Vector Node Features
    box4 = FancyBboxPatch((0.5, 3.4), 9, 0.7, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_node, linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 3.95, 'Step 3: Compute Vector Node Features (Optional)', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 3.6, 'orientation (forward/backward unit vectors)', 
            ha='center', va='center', fontsize=9)
    
    # Arrow 4
    arrow4 = FancyArrowPatch((5, 3.2), (5, 2.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Stage 5: Edge Construction
    box5 = FancyBboxPatch((0.5, 1.7), 9, 0.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_edge, linewidth=2)
    ax.add_patch(box5)
    ax.text(5, 2.3, 'Step 4: Construct Edges', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, 1.95, 'KNN (knn_10, knn_16, knn_30) | Epsilon (eps_8) | Sequential (seq_forward/backward)', 
            ha='center', va='center', fontsize=9)
    
    # Arrow 5
    arrow5 = FancyArrowPatch((2.5, 1.5), (2.5, 0.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    arrow5b = FancyArrowPatch((7.5, 1.5), (7.5, 0.7),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5b)
    
    # Stage 6a: Scalar Edge Features
    box6a = FancyBboxPatch((0.3, -0.3), 4.4, 0.8, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor=color_edge, linewidth=2)
    ax.add_patch(box6a)
    ax.text(2.5, 0.3, 'Step 5a: Scalar Edge Features', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(2.5, -0.05, 'edge_distance | edge_type\nsequence_distance | pos_emb', 
            ha='center', va='center', fontsize=8)
    
    # Stage 6b: Vector Edge Features
    box6b = FancyBboxPatch((5.3, -0.3), 4.4, 0.8, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor=color_edge, linewidth=2)
    ax.add_patch(box6b)
    ax.text(7.5, 0.3, 'Step 5b: Vector Edge Features', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(7.5, -0.05, 'edge_vectors\n(unit directional vectors)', 
            ha='center', va='center', fontsize=8)
    
    # Arrows to output
    arrow6a = FancyArrowPatch((2.5, -0.5), (3.5, -1.3),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow6a)
    arrow6b = FancyArrowPatch((7.5, -0.5), (6.5, -1.3),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow6b)
    
    # Stage 7: Output
    box7 = FancyBboxPatch((0.5, -2.5), 9, 1, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(box7)
    ax.text(5, -1.7, 'Featurized Graph', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(5, -2.1, 'x (node features) | pos (coordinates) | edge_index | edge_attr', 
            ha='center', va='center', fontsize=9)
    ax.text(5, -2.35, 'x_vector_attr (optional) | edge_vector_attr (optional)', 
            ha='center', va='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    return fig


def create_representation_comparison():
    """Create a visual comparison of different representations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CA representation
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('CA (C-alpha only)\nMemory: 1×  |  Nodes: N', fontsize=12, fontweight='bold')
    
    # Draw CA atoms
    for i in range(5):
        circle = plt.Circle((2*i+1, 1.5), 0.3, color='#3498db', ec='black', linewidth=2)
        ax.add_patch(circle)
        if i > 0:
            ax.plot([2*(i-1)+1.3, 2*i+0.7], [1.5, 1.5], 'k-', linewidth=2)
        ax.text(2*i+1, 1.5, 'Cα', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # BB representation
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('BB (Backbone)\nMemory: 4×  |  Nodes: 4N', fontsize=12, fontweight='bold')
    
    # Draw backbone atoms
    atoms = ['N', 'Cα', 'C', 'O']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for res in range(2):
        for i, (atom, color) in enumerate(zip(atoms, colors)):
            x = res*5 + i*1.2 + 0.5
            circle = plt.Circle((x, 1.5), 0.25, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            if i > 0 or res > 0:
                prev_x = res*5 + (i-1)*1.2 + 0.5 if i > 0 else (res-1)*5 + 3*1.2 + 0.5
                ax.plot([prev_x+0.25, x-0.25], [1.5, 1.5], 'k-', linewidth=1.5)
            ax.text(x, 1.5, atom, ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    
    # FA representation
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('FA (Full Atom)\nMemory: ~20×  |  Nodes: ~20N', fontsize=12, fontweight='bold')
    
    # Draw full atom representation (simplified)
    np.random.seed(42)
    for res in range(2):
        # Backbone
        for i in range(4):
            x = res*4.5 + i*0.8 + 0.5
            circle = plt.Circle((x, 1.5), 0.2, color='#3498db', ec='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Sidechain (simplified as cluster)
        sc_x = res*4.5 + 2
        sc_y = 2.3
        for _ in range(6):
            offset_x = np.random.uniform(-0.4, 0.4)
            offset_y = np.random.uniform(-0.3, 0.3)
            circle = plt.Circle((sc_x+offset_x, sc_y+offset_y), 0.15, 
                              color='#95a5a6', ec='black', linewidth=1)
            ax.add_patch(circle)
    
    ax.text(5, 0.5, 'Each residue = backbone + all sidechain atoms', 
            ha='center', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig


def create_edge_type_comparison():
    """Create a visual comparison of different edge types."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate sample node positions
    np.random.seed(42)
    n_nodes = 15
    positions = np.random.rand(n_nodes, 2) * 8 + 1
    
    def plot_graph(ax, title, edge_func, description):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Draw edges first
        edges = edge_func(positions)
        for i, j in edges:
            ax.plot([positions[i, 0], positions[j, 0]], 
                   [positions[i, 1], positions[j, 1]], 
                   'gray', alpha=0.5, linewidth=1)
        
        # Draw nodes
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='#3498db', s=200, zorder=10, edgecolors='black', linewidth=2)
        
        # Add description
        ax.text(5, 0.3, description, ha='center', va='center', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return len(edges)
    
    # KNN-3
    def knn_edges(pos, k=3):
        edges = []
        for i in range(len(pos)):
            dists = np.sum((pos - pos[i])**2, axis=1)
            nearest = np.argsort(dists)[1:k+1]  # Exclude self
            for j in nearest:
                edges.append((i, j))
        return edges
    
    n_edges = plot_graph(axes[0, 0], 'KNN-3 Edges', 
                        lambda p: knn_edges(p, k=3),
                        f'~{3*n_nodes} edges | Fixed degree | Spatial proximity')
    
    # Epsilon
    def eps_edges(pos, eps=2.0):
        edges = []
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                if np.linalg.norm(pos[i] - pos[j]) < eps:
                    edges.append((i, j))
                    edges.append((j, i))
        return edges
    
    n_edges = plot_graph(axes[0, 1], 'Epsilon (r=2.0) Edges', 
                        lambda p: eps_edges(p, eps=2.0),
                        f'Variable degree | Distance threshold | Local interactions')
    
    # Sequential
    def seq_edges(pos):
        # Assume nodes are ordered sequentially
        sorted_idx = np.argsort(positions[:, 0])
        edges = []
        for i in range(len(sorted_idx)-1):
            edges.append((sorted_idx[i], sorted_idx[i+1]))
        return edges
    
    n_edges = plot_graph(axes[1, 0], 'Sequential Forward Edges', 
                        seq_edges,
                        f'~{n_nodes} edges | Sequence connectivity | Minimal')
    
    # Mixed (KNN + Sequential)
    def mixed_edges(pos):
        knn = knn_edges(pos, k=2)
        seq = seq_edges(pos)
        return knn + seq
    
    n_edges = plot_graph(axes[1, 1], 'Mixed (KNN-2 + Sequential)', 
                        mixed_edges,
                        f'~{2*n_nodes + n_nodes} edges | Spatial + sequence info')
    
    plt.tight_layout()
    return fig


def create_memory_comparison_chart():
    """Create a bar chart comparing memory usage."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = [
        'CA + knn_10\nMinimal',
        'CA + knn_16\nStandard', 
        'CA + knn_30\nFull',
        'BB + knn_10\nMinimal',
        'BB + knn_16\nStandard',
        'BB + knn_30\nFull',
        'FA + knn_10\nMinimal'
    ]
    
    memory = [1.0, 1.6, 3.0, 4.0, 6.4, 12.0, 20.0]  # Relative memory
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    bars = ax.bar(configs, memory, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Relative Memory Usage', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(memory) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add reference line
    ax.axhline(y=4.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Recommended for 16GB GPU')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def create_performance_comparison_chart():
    """Create a scatter plot of memory vs speed."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    configs = [
        ('CA + knn_10\nMinimal', 1.0, 100, 200),
        ('CA + knn_16\nStandard', 1.6, 80, 200),
        ('CA + knn_30\nFull', 3.0, 50, 200),
        ('BB + knn_10', 4.0, 70, 150),
        ('BB + knn_16', 6.4, 40, 150),
        ('BB + knn_30', 12.0, 20, 150),
    ]
    
    for name, memory, speed, size in configs:
        ax.scatter(memory, speed, s=size, alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(name, (memory, speed), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Relative Memory Usage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Speed (batches/min)', fontsize=12, fontweight='bold')
    ax.set_title('Memory vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add pareto frontier
    pareto_points = [(1.0, 100), (1.6, 80), (3.0, 50), (4.0, 70)]
    pareto_points.sort()
    memory_pareto = [p[0] for p in pareto_points]
    speed_pareto = [p[1] for p in pareto_points]
    ax.plot(memory_pareto, speed_pareto, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    ax.legend()
    
    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    print("Generating ProteinWorkshop Featurization Visualizations...")
    
    # Create output directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate workflow diagram
    print("1. Creating workflow diagram...")
    fig1 = create_workflow_diagram()
    fig1.savefig('visualizations/featurization_workflow.png', dpi=300, bbox_inches='tight')
    print("   Saved: visualizations/featurization_workflow.png")
    
    # Generate representation comparison
    print("2. Creating representation comparison...")
    fig2 = create_representation_comparison()
    fig2.savefig('visualizations/representation_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: visualizations/representation_comparison.png")
    
    # Generate edge type comparison
    print("3. Creating edge type comparison...")
    fig3 = create_edge_type_comparison()
    fig3.savefig('visualizations/edge_type_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: visualizations/edge_type_comparison.png")
    
    # Generate memory comparison
    print("4. Creating memory comparison chart...")
    fig4 = create_memory_comparison_chart()
    fig4.savefig('visualizations/memory_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: visualizations/memory_comparison.png")
    
    # Generate performance comparison
    print("5. Creating performance comparison chart...")
    fig5 = create_performance_comparison_chart()
    fig5.savefig('visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: visualizations/performance_comparison.png")
    
    print("\nAll visualizations generated successfully!")
    print("Check the 'visualizations/' directory for the output files.")
    
    # Show plots (optional - comment out if running in non-interactive mode)
    # plt.show()


if __name__ == "__main__":
    main()
