"""
Create publication-quality diagrams for Pipeline 18 presentation.

Generates 5 key visualizations:
1. Intermediate node histogram concept
2. Pipeline flow diagram
3. Neural network architecture
4. Anomaly detection quadrant plot
5. Degree binning strategy

Usage:
    python create_pipeline_diagrams.py

Output:
    Saves PNG files to visualizations/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Create output directory
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)


def create_intermediate_signature_diagram():
    """
    Diagram A: Intermediate Node Histogram Concept

    Shows:
    - Metapath structure (Compound -> Gene -> Pathway)
    - Example Gene connectivity patterns
    - 2D histogram visualization
    - How histogram captures topology
    """
    fig = plt.figure(figsize=(14, 8))

    # Create GridSpec for layout
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Panel A: Metapath structure
    ax_metapath = fig.add_subplot(gs[0, :])
    ax_metapath.set_xlim(0, 12)
    ax_metapath.set_ylim(0, 4)
    ax_metapath.axis('off')

    # Draw nodes
    compound_x, compound_y = 1.5, 2
    gene_x, gene_y = 6, 2
    pathway_x, pathway_y = 10.5, 2

    # Compound
    compound = Circle((compound_x, compound_y), 0.6, color='#3498db',
                      alpha=0.8, zorder=3)
    ax_metapath.add_patch(compound)
    ax_metapath.text(compound_x, compound_y, 'Compound\n(Metformin)',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white', zorder=4)

    # Genes (multiple intermediate nodes)
    gene_positions = [(5.5, 3.2), (6, 2), (6.5, 0.8)]
    gene_colors = ['#e74c3c', '#e74c3c', '#e74c3c']
    gene_labels = ['Gene 1\n(in=120, out=3)',
                   'Gene 2\n(in=80, out=8)',
                   'Gene 3\n(in=50, out=12)']

    for (gx, gy), color, label in zip(gene_positions, gene_colors, gene_labels):
        gene = Circle((gx, gy), 0.4, color=color, alpha=0.8, zorder=3)
        ax_metapath.add_patch(gene)
        ax_metapath.text(gx, gy, label, ha='center', va='center',
                         fontsize=7, color='white', zorder=4)

    # Pathway
    pathway = Circle((pathway_x, pathway_y), 0.6, color='#2ecc71',
                     alpha=0.8, zorder=3)
    ax_metapath.add_patch(pathway)
    ax_metapath.text(pathway_x, pathway_y, 'Pathway\n(Insulin\nSignaling)',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white', zorder=4)

    # Draw edges
    for gx, gy in gene_positions:
        # Compound to Gene
        arrow1 = FancyArrowPatch((compound_x + 0.6, compound_y),
                                (gx - 0.4, gy),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color='#34495e', alpha=0.6,
                                zorder=2)
        ax_metapath.add_patch(arrow1)

        # Gene to Pathway
        arrow2 = FancyArrowPatch((gx + 0.4, gy),
                                (pathway_x - 0.6, pathway_y),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color='#34495e', alpha=0.6,
                                zorder=2)
        ax_metapath.add_patch(arrow2)

    # Add labels
    ax_metapath.text(3.5, 3.5, 'Edge 1:\nCompound\nbinds Gene',
                     ha='center', fontsize=8, style='italic',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax_metapath.text(8.5, 3.5, 'Edge 2:\nGene participates\nin Pathway',
                     ha='center', fontsize=8, style='italic',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Panel B: Example intermediate connectivity table
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')

    table_data = [
        ['Gene', 'In-deg', 'Out-deg', 'Bin'],
        ['Gene 1', '120', '3', '(High, Low)'],
        ['Gene 2', '80', '8', '(Med, Med)'],
        ['Gene 3', '50', '12', '(Med, High)'],
        ['Gene 4', '95', '5', '(Med, Low)'],
        ['Gene 5', '110', '4', '(High, Low)'],
    ]

    table = ax_table.table(cellText=table_data, cellLoc='center',
                          loc='center', bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax_table.text(0.5, 0.95, 'Example Intermediate Nodes',
                  ha='center', fontsize=10, fontweight='bold',
                  transform=ax_table.transAxes)

    # Panel C: 2D histogram
    ax_hist = fig.add_subplot(gs[1, 1])

    # Create example histogram (5x5 for visibility)
    hist_data = np.array([
        [0, 2, 3, 1, 0],
        [1, 5, 8, 4, 1],
        [2, 12, 15, 6, 2],
        [1, 6, 4, 2, 0],
        [0, 1, 1, 0, 0]
    ])

    im = ax_hist.imshow(hist_data, cmap='YlOrRd', aspect='auto',
                        interpolation='nearest')

    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = ax_hist.text(j, i, str(hist_data[i, j]),
                               ha='center', va='center', color='black',
                               fontsize=9, fontweight='bold')

    ax_hist.set_xticks(range(5))
    ax_hist.set_yticks(range(5))
    ax_hist.set_xticklabels(['0-20', '21-50', '51-100', '101-200', '200+'],
                            fontsize=8)
    ax_hist.set_yticklabels(['0-2', '3-5', '6-10', '11-20', '20+'],
                            fontsize=8)
    ax_hist.set_xlabel('Gene In-Degree Bins', fontsize=9, fontweight='bold')
    ax_hist.set_ylabel('Gene Out-Degree Bins', fontsize=9, fontweight='bold')
    ax_hist.set_title('2D Histogram of Intermediate Degrees\n(5x5 bins shown, actual: 10x10)',
                      fontsize=10, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax_hist, fraction=0.046, pad=0.04)
    cbar.set_label('Gene Count', rotation=270, labelpad=15, fontsize=8)

    # Panel D: Flattened signature vector
    ax_vector = fig.add_subplot(gs[1, 2])
    ax_vector.axis('off')

    # Show flattened vector
    flat_hist = hist_data.flatten()
    vector_display = flat_hist[:15]  # Show first 15 elements

    ax_vector.text(0.5, 0.95, 'Flattened Feature Vector',
                   ha='center', fontsize=10, fontweight='bold',
                   transform=ax_vector.transAxes)

    # Draw vector elements
    y_start = 0.85
    for i, val in enumerate(vector_display):
        normalized_val = val / flat_hist.sum()
        ax_vector.text(0.1, y_start - i*0.05,
                      f'inter_sig_{i:02d}:',
                      fontsize=7, transform=ax_vector.transAxes)
        ax_vector.text(0.5, y_start - i*0.05,
                      f'{normalized_val:.4f}',
                      fontsize=7, transform=ax_vector.transAxes,
                      family='monospace')

    ax_vector.text(0.5, y_start - 15*0.05,
                   '... (100 features total)',
                   ha='center', fontsize=7, style='italic',
                   transform=ax_vector.transAxes)

    # Add interpretation box
    interpretation = (
        'Interpretation:\n'
        'Most genes: HIGH in-degree,\n'
        'LOW out-degree\n'
        '→ Bottleneck topology\n'
        '→ FEWER pathways expected'
    )
    ax_vector.text(0.5, 0.15, interpretation,
                   ha='center', va='center', fontsize=8,
                   transform=ax_vector.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Overall title
    fig.suptitle('Intermediate Node Degree Signature Construction',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_dir / 'diagram_a_intermediate_signature.png',
                dpi=300, bbox_inches='tight')
    print("Created: diagram_a_intermediate_signature.png")
    plt.close()


def create_pipeline_flow_diagram():
    """
    Diagram B: Pipeline Flow

    Shows:
    - Data flow through 18a -> 18f -> 18g -> 18h
    - Input/output at each stage
    - Memory reduction
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define box positions
    boxes = [
        {'name': '18a\nData Preparation', 'x': 2, 'y': 8, 'color': '#3498db'},
        {'name': '18f\nNeural Network\nTraining', 'x': 7, 'y': 8, 'color': '#e74c3c'},
        {'name': '18g\nVariance\nEstimation', 'x': 12, 'y': 8, 'color': '#9b59b6'},
        {'name': '18h\nAnomaly\nDetection', 'x': 7, 'y': 2, 'color': '#2ecc71'},
    ]

    # Draw boxes
    for box in boxes:
        rect = FancyBboxPatch((box['x']-1.2, box['y']-0.8), 2.4, 1.6,
                              boxstyle='round,pad=0.1',
                              edgecolor='black', facecolor=box['color'],
                              alpha=0.7, linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'], box['y'], box['name'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')

    # Add input/output labels
    # 18a
    ax.text(2, 9.5, 'Input: 2.7M pairs', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, 6.8, 'Output: 100 bins\n(1000x reduction)', ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 18f
    ax.text(7, 9.5, 'Input: 100 bins', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(7, 6.8, 'Output: Trained NN\n(50 KB model)', ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 18g
    ax.text(12, 9.5, 'Input: 20 permutations', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(12, 6.8, 'Output: Variance estimates\n(100 bins)', ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 18h
    ax.text(7, 0.5, 'Output: ~30K enriched pairs\n~1.5K novel discoveries',
            ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Draw arrows
    # 18a -> 18f
    arrow1 = FancyArrowPatch((4.2, 8), (5.8, 8),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='black')
    ax.add_patch(arrow1)
    ax.text(5, 8.3, 'Degree-binned\ntraining data', ha='center', fontsize=8,
            style='italic')

    # 18f -> 18g
    arrow2 = FancyArrowPatch((8.2, 8), (10.8, 8),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='black')
    ax.add_patch(arrow2)
    ax.text(9.5, 8.3, 'Trained\nmodel', ha='center', fontsize=8,
            style='italic')

    # 18f -> 18h
    arrow3 = FancyArrowPatch((7, 6.8), (7, 3.6),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='black')
    ax.add_patch(arrow3)
    ax.text(7.8, 5, 'Predictions', ha='center', fontsize=8,
            style='italic')

    # 18g -> 18h
    arrow4 = FancyArrowPatch((11, 7), (8.5, 3.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='black')
    ax.add_patch(arrow4)
    ax.text(10.5, 4.5, 'Variance\nestimates', ha='center', fontsize=8,
            style='italic')

    # Add process descriptions
    processes = [
        {'x': 2, 'y': 5.5, 'text': 'Process:\n• Bin by degree\n• Compute signatures\n• Aggregate counts'},
        {'x': 7, 'y': 5.5, 'text': 'Process:\n• Train NN (100 bins)\n• Learn non-linear\n  degree effects'},
        {'x': 12, 'y': 5.5, 'text': 'Process:\n• Validate on perms\n• Compute variance\n  per bin'},
        {'x': 2, 'y': 2, 'text': 'Process:\n• For each pair:\n• Predict expected\n• Compute Z-score\n• Compare to DWPC'},
    ]

    for proc in processes:
        ax.text(proc['x'], proc['y'], proc['text'],
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Add timeline
    ax.text(7, 0.2, 'Total Pipeline: ~4 hours | Peak Memory: 16 GB',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    # Title
    ax.text(7, 9.8, 'Pipeline 18: Complete Workflow',
            ha='center', fontsize=14, fontweight='bold')

    plt.savefig(output_dir / 'diagram_b_pipeline_flow.png',
                dpi=300, bbox_inches='tight')
    print("Created: diagram_b_pipeline_flow.png")
    plt.close()


def create_nn_architecture_diagram():
    """
    Diagram C: Neural Network Architecture

    Shows:
    - Layer dimensions
    - Activation functions
    - Input/output examples
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Layer specifications
    layers = [
        {'name': 'Input', 'neurons': 102, 'y': 10.5, 'color': '#3498db'},
        {'name': 'Dense 1', 'neurons': 128, 'y': 9, 'color': '#e74c3c'},
        {'name': 'ReLU + Dropout(0.1)', 'neurons': 128, 'y': 8.2, 'color': '#95a5a6'},
        {'name': 'Dense 2', 'neurons': 64, 'y': 7, 'color': '#e74c3c'},
        {'name': 'ReLU + Dropout(0.1)', 'neurons': 64, 'y': 6.2, 'color': '#95a5a6'},
        {'name': 'Dense 3', 'neurons': 32, 'y': 5, 'color': '#e74c3c'},
        {'name': 'ReLU + Dropout(0.1)', 'neurons': 32, 'y': 4.2, 'color': '#95a5a6'},
        {'name': 'Output Dense', 'neurons': 1, 'y': 3, 'color': '#e74c3c'},
        {'name': 'Softplus', 'neurons': 1, 'y': 2.2, 'color': '#95a5a6'},
        {'name': 'Output', 'neurons': 1, 'y': 1, 'color': '#2ecc71'},
    ]

    # Draw layers
    for layer in layers:
        width = min(layer['neurons'] / 20, 4)  # Scale width by neuron count
        rect = Rectangle((5 - width/2, layer['y'] - 0.25),
                         width, 0.5,
                         facecolor=layer['color'], edgecolor='black',
                         linewidth=2, alpha=0.7)
        ax.add_patch(rect)

        # Layer name
        ax.text(5, layer['y'], layer['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if layer['color'] != '#95a5a6' else 'black')

        # Neuron count (on right)
        ax.text(8, layer['y'], f'{layer["neurons"]} neurons',
                ha='left', va='center', fontsize=8)

    # Draw arrows between layers
    for i in range(len(layers) - 1):
        y1 = layers[i]['y'] - 0.25
        y2 = layers[i+1]['y'] + 0.25
        arrow = FancyArrowPatch((5, y1), (5, y2),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='black', alpha=0.5)
        ax.add_patch(arrow)

    # Add input example
    input_text = (
        'Input Example:\n'
        '  source_bin: 5\n'
        '  target_bin: 7\n'
        '  inter_sig_0: 0.012\n'
        '  inter_sig_1: 0.020\n'
        '  ...\n'
        '  inter_sig_99: 0.008'
    )
    ax.text(1, 10.5, input_text,
            ha='left', va='center', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Add output example
    output_text = (
        'Output Example:\n'
        '  pathway_count: 48.2\n'
        '  (non-negative)'
    )
    ax.text(1, 1, output_text,
            ha='left', va='center', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add annotations
    annotations = [
        {'y': 8.2, 'text': 'Non-linearity', 'x': 8.5},
        {'y': 6.2, 'text': 'Regularization', 'x': 8.5},
        {'y': 2.2, 'text': 'Ensure ≥ 0', 'x': 8.5},
    ]

    for ann in annotations:
        ax.text(ann['x'], ann['y'], ann['text'],
                ha='left', va='center', fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Add parameter count
    total_params = (102*128 + 128) + (128*64 + 64) + (64*32 + 32) + (32*1 + 1)
    ax.text(5, 0.3, f'Total Parameters: {total_params:,}',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    # Title
    ax.text(5, 11.5, 'Degree Signature Neural Network Architecture',
            ha='center', fontsize=14, fontweight='bold')

    plt.savefig(output_dir / 'diagram_c_nn_architecture.png',
                dpi=300, bbox_inches='tight')
    print("Created: diagram_c_nn_architecture.png")
    plt.close()


def create_quadrant_plot_diagram():
    """
    Diagram D: Anomaly Detection Quadrant Plot

    Shows:
    - DWPC vs Z-score scatter
    - Annotated quadrants
    - Example discoveries highlighted
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate synthetic data for visualization
    np.random.seed(42)
    n_points = 1000

    # Create four clusters for quadrants
    # Q1: High DWPC, High Z (validated)
    q1_dwpc = np.random.gamma(8, 2, 250)
    q1_z = np.random.gamma(6, 1.5, 250)

    # Q2: High DWPC, Low Z (degree-driven)
    q2_dwpc = np.random.gamma(8, 2, 250)
    q2_z = np.random.gamma(2, 1, 250)

    # Q3: Low DWPC, High Z (NOVEL)
    q3_dwpc = np.random.gamma(2, 1, 200)
    q3_z = np.random.gamma(6, 1.5, 200)

    # Q4: Low DWPC, Low Z (modest)
    q4_dwpc = np.random.gamma(2, 1, 300)
    q4_z = np.random.gamma(2, 1, 300)

    # Combine
    all_dwpc = np.concatenate([q1_dwpc, q2_dwpc, q3_dwpc, q4_dwpc])
    all_z = np.concatenate([q1_z, q2_z, q3_z, q4_z])

    # Compute thresholds (75th percentile)
    dwpc_threshold = np.percentile(all_dwpc, 75)
    z_threshold = np.percentile(all_z, 75)

    # Plot all points
    ax.scatter(all_dwpc, all_z, c='gray', alpha=0.3, s=20, edgecolors='none')

    # Highlight quadrants with different colors
    q1_mask = (all_dwpc >= dwpc_threshold) & (all_z >= z_threshold)
    q2_mask = (all_dwpc >= dwpc_threshold) & (all_z < z_threshold)
    q3_mask = (all_dwpc < dwpc_threshold) & (all_z >= z_threshold)
    q4_mask = (all_dwpc < dwpc_threshold) & (all_z < z_threshold)

    ax.scatter(all_dwpc[q1_mask], all_z[q1_mask], c='green', alpha=0.6,
               s=30, edgecolors='k', linewidth=0.5, label='Q1: Validated')
    ax.scatter(all_dwpc[q2_mask], all_z[q2_mask], c='blue', alpha=0.6,
               s=30, edgecolors='k', linewidth=0.5, label='Q2: Degree-driven')
    ax.scatter(all_dwpc[q3_mask], all_z[q3_mask], c='orange', alpha=0.7,
               s=40, edgecolors='k', linewidth=0.5, label='Q3: NOVEL')
    ax.scatter(all_dwpc[q4_mask], all_z[q4_mask], c='lightgray', alpha=0.4,
               s=20, edgecolors='none', label='Q4: Modest')

    # Draw threshold lines
    ax.axhline(z_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Z-score threshold (75th %ile)')
    ax.axvline(dwpc_threshold, color='red', linestyle='--', linewidth=2,
               label=f'DWPC threshold (75th %ile)')

    # Annotate quadrants
    ax.text(dwpc_threshold/2, z_threshold*1.5, 'NOVEL\nDISCOVERIES',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='orange',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='orange', linewidth=3))

    ax.text(dwpc_threshold*1.5, z_threshold*1.5, 'Validated\nEnrichments',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='green', linewidth=2))

    ax.text(dwpc_threshold*1.5, z_threshold/2, 'Degree-driven\nEnrichments',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='blue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='blue', linewidth=2))

    ax.text(dwpc_threshold/2, z_threshold/2, 'Modest\nEnrichments',
            ha='center', va='center', fontsize=12,
            color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='gray', linewidth=1))

    # Highlight example novel discovery
    example_idx = np.where(q3_mask)[0][0]
    ax.scatter(all_dwpc[example_idx], all_z[example_idx],
               c='red', s=300, marker='*', edgecolors='black',
               linewidth=2, zorder=10, label='Example: Metformin')

    ax.annotate('Metformin → Insulin Pathway\nZ=8.5, DWPC=2.3\np=1.2e-17',
                xy=(all_dwpc[example_idx], all_z[example_idx]),
                xytext=(all_dwpc[example_idx]+5, all_z[example_idx]+2),
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                               linewidth=2))

    # Labels and formatting
    ax.set_xlabel('DWPC Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Anomaly Z-score (Enrichment)', fontsize=13, fontweight='bold')
    ax.set_title('Anomaly Detection: Quadrant Analysis\n'
                 'Identifying Novel Compound-Pathway Associations',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add interpretation box
    interpretation = (
        'INTERPRETATION:\n'
        '• Q1 (Green): Both methods agree - strong candidates\n'
        '• Q2 (Blue): High DWPC only - explained by degree\n'
        '• Q3 (Orange): High Z-score only - NOVEL mechanisms\n'
        '• Q4 (Gray): Weak signal - not prioritized'
    )
    ax.text(0.02, 0.98, interpretation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'diagram_d_quadrant_plot.png',
                dpi=300, bbox_inches='tight')
    print("Created: diagram_d_quadrant_plot.png")
    plt.close()


def create_degree_binning_diagram():
    """
    Diagram E: Degree Binning Strategy

    Shows:
    - Continuous degree distribution
    - Binning into quantiles
    - Aggregation within bins
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Original degree distribution
    ax = axes[0, 0]
    np.random.seed(42)

    # Generate power-law-like degree distribution
    degrees = np.random.power(0.5, 1500) * 200

    ax.hist(degrees, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Node Degree', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('A. Original Degree Distribution\n(Power-law-like)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: Quantile binning
    ax = axes[0, 1]

    # Compute bins
    n_bins = 10
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(degrees, percentiles)

    # Plot histogram with bin edges
    ax.hist(degrees, bins=bins, color='coral', alpha=0.7, edgecolor='black')

    # Add vertical lines for bin edges
    for i, b in enumerate(bins[1:-1], 1):
        ax.axvline(b, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(b, ax.get_ylim()[1]*0.9, f'Bin {i}',
                ha='center', fontsize=8, rotation=90,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('Node Degree', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('B. Quantile-Based Binning\n(10 bins, ~equal counts)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel C: Bin characteristics
    ax = axes[1, 0]
    ax.axis('off')

    # Create table of bin characteristics
    bin_data = [['Bin', 'Degree Range', 'Node Count']]
    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i+1]
        count = ((degrees >= lower) & (degrees < upper)).sum()
        bin_data.append([f'{i}', f'{lower:.0f}-{upper:.0f}', f'{count}'])

    table = ax.table(cellText=bin_data, cellLoc='center',
                    loc='center', bbox=[0.1, 0.1, 0.8, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.text(0.5, 0.98, 'C. Bin Characteristics',
            ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Panel D: Aggregation illustration
    ax = axes[1, 1]

    # Show bin pairs (source x target grid)
    n_bins_display = 10
    grid = np.random.randint(0, 100, (n_bins_display, n_bins_display))

    im = ax.imshow(grid, cmap='viridis', aspect='auto')

    ax.set_xticks(range(n_bins_display))
    ax.set_yticks(range(n_bins_display))
    ax.set_xticklabels(range(n_bins_display))
    ax.set_yticklabels(range(n_bins_display))
    ax.set_xlabel('Target Degree Bin', fontsize=11, fontweight='bold')
    ax.set_ylabel('Source Degree Bin', fontsize=11, fontweight='bold')
    ax.set_title('D. Bin Pair Combinations\n(10 x 10 = 100 training samples)',
                 fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Pathway Count', rotation=270, labelpad=20,
                   fontsize=10)

    # Highlight example bin
    rect = Rectangle((6.5, 4.5), 1, 1, fill=False, edgecolor='red',
                     linewidth=4)
    ax.add_patch(rect)
    ax.annotate('Example:\nsource_bin=5\ntarget_bin=7',
                xy=(7, 5), xytext=(8.5, 7),
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', linewidth=2))

    # Overall title
    fig.suptitle('Degree Binning Strategy: From Millions to Hundreds',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_dir / 'diagram_e_degree_binning.png',
                dpi=300, bbox_inches='tight')
    print("Created: diagram_e_degree_binning.png")
    plt.close()


def main():
    """Generate all diagrams."""
    print("Generating Pipeline 18 presentation diagrams...\n")

    create_intermediate_signature_diagram()
    create_pipeline_flow_diagram()
    create_nn_architecture_diagram()
    create_quadrant_plot_diagram()
    create_degree_binning_diagram()

    print(f"\nAll diagrams saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. diagram_a_intermediate_signature.png")
    print("  2. diagram_b_pipeline_flow.png")
    print("  3. diagram_c_nn_architecture.png")
    print("  4. diagram_d_quadrant_plot.png")
    print("  5. diagram_e_degree_binning.png")


if __name__ == "__main__":
    main()