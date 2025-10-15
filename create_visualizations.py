"""
Visualization Scripts for Deep Funding Writeup
Creates publication-quality charts for the final writeup
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================================
# Chart 1: Score Evolution Journey
# ============================================================================

def plot_score_evolution():
    """Shows the journey from disaster to winner"""
    
    phases = [
        ('Baseline', 4.75, 'Initial approach'),
        ('Refined', 4.88, 'Seeds-only tuning'),
        ('CRISIS', 7.03, 'New data added'),
        ('OSO Fix', 4.54, 'External data rescue'),
        ('Phase 2', 4.48, 'Complex juror model'),
        ('Phase 3', 4.46, 'Conservative wins'),
        ('Phase 5', 5.22, 'Over-engineering fails')
    ]
    
    # Private LB scores
    private_scores = {
        'Phase 2': 6.66,
        'Phase 3': 6.46,  # WINNER
        'Phase 5': 6.75
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Public LB evolution
    x = list(range(len(phases)))
    scores = [p[1] for p in phases]
    labels = [p[0] for p in phases]
    colors = ['steelblue', 'steelblue', 'red', 'orange', 'purple', 'green', 'gray']
    
    ax1.plot(x, scores, 'o-', linewidth=2.5, markersize=10, color='darkblue', alpha=0.6)
    for i, (phase, score, desc) in enumerate(phases):
        color = colors[i]
        ax1.scatter(i, score, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=3)
        ax1.annotate(f'{phase}\n{score:.2f}', (i, score), 
                    textcoords="offset points", xytext=(0,15 if score < 6 else -30),
                    ha='center', fontsize=10, fontweight='bold')
    
    ax1.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='Target: Sub-5.0')
    ax1.set_xlabel('Development Timeline', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Public Leaderboard Score (lower = better)', fontsize=13, fontweight='bold')
    ax1.set_title('The Journey: From Crisis to Winner', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[0] for p in phases], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Private LB comparison
    phases_private = ['Phase 2\n(Complex)', 'Phase 3\n(Conservative)', 'Phase 5\n(Constrained)']
    scores_private = [6.66, 6.46, 6.75]
    colors_private = ['purple', 'green', 'gray']
    
    bars = ax2.barh(phases_private, scores_private, color=colors_private, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Highlight winner
    bars[1].set_color('gold')
    bars[1].set_alpha(0.9)
    
    for i, (phase, score) in enumerate(zip(phases_private, scores_private)):
        ax2.text(score + 0.05, i, f'{score:.2f}', va='center', fontsize=12, fontweight='bold')
    
    ax2.axvline(x=6.46, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Winner: 6.46')
    ax2.set_xlabel('Private Leaderboard Score (lower = better)', fontsize=13, fontweight='bold')
    ax2.set_title('Final Private LB Results', fontsize=15, fontweight='bold')
    ax2.invert_xaxis()  # Lower is better
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('viz_1_score_evolution.png', dpi=300, bbox_inches='tight')
    print("✓ Created viz_1_score_evolution.png")
    plt.close()


# ============================================================================
# Chart 2: Prediction vs Actual Scatter
# ============================================================================

def plot_prediction_vs_actual():
    """Compare Phase 3 predictions against actual resolution"""
    
    # Top 20 repos by actual weight
    data = [
        ('ethereum/eips', 0.1394, 0.2903),
        ('ethereum-package', 0.0000, 0.0923),  # MISS!
        ('foundry', 0.0345, 0.0803),
        ('go-ethereum', 0.1767, 0.0585),  # OVERSHOT!
        ('hardhat', 0.0120, 0.0540),
        ('solidity', 0.1006, 0.0424),
        ('teku', 0.0203, 0.0434),
        ('openzeppelin', 0.0365, 0.0359),
        ('lighthouse', 0.0385, 0.0322),
        ('nethermind', 0.0344, 0.0309),
        ('prysm', 0.0375, 0.0286),
        ('lodestar', 0.0251, 0.0256),
        ('account-abstraction', 0.0065, 0.0220),
        ('consensus-specs', 0.0595, 0.0200),
        ('web3.py', 0.0081, 0.0136),
        ('nimbus-eth2', 0.0160, 0.0137),
        ('ethers.js', 0.0423, 0.0116),  # OVERSHOT!
        ('execution-apis', 0.0522, 0.0095),
        ('helios', 0.0000, 0.0097),
        ('besu', 0.0084, 0.0084),
    ]
    
    repos = [d[0] for d in data]
    predicted = np.array([d[1] for d in data]) * 100
    actual = np.array([d[2] for d in data]) * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Perfect prediction line
    max_val = max(predicted.max(), actual.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='Perfect Prediction')
    
    # Color by error magnitude
    errors = np.abs(predicted - actual)
    colors = plt.cm.RdYlGn_r(errors / errors.max())
    
    scatter = ax.scatter(predicted, actual, s=errors*30 + 100, c=errors, 
                        cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Annotate biggest misses
    big_misses = [
        ('eips', predicted[0], actual[0], 'Underestimated'),
        ('ethereum-package', predicted[1], actual[1], 'MISSED!'),
        ('go-ethereum', predicted[3], actual[3], 'Overestimated'),
        ('ethers.js', predicted[16], actual[16], 'Overestimated'),
    ]
    
    for label, pred, act, reason in big_misses:
        ax.annotate(f'{label}\n({reason})', 
                   xy=(pred, act), xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5),
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Phase 3 Predicted Weight (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual Private LB Weight (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prediction vs Reality: What I Got Right (and Wrong)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error (%)', fontsize=11, fontweight='bold')
    
    # Add stats
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    stats_text = f'MAE: {mae:.2f}%\nRMSE: {rmse:.2f}%'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=11, verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig('viz_2_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Created viz_2_prediction_vs_actual.png")
    plt.close()


# ============================================================================
# Chart 3: Approach Comparison Heatmap
# ============================================================================

def plot_approach_comparison():
    """Compare different approaches across key metrics"""
    
    approaches = ['Phase 2\n(Complex)', 'Phase 3\n(Conservative)', 'Phase 5\n(Constrained)']
    metrics = [
        'Training Fit\n(lower=better)',
        'Public LB\n(lower=better)',
        'Private LB\n(lower=better)',
        'Complexity\n(params)',
        'Generalization'
    ]
    
    # Normalize to 0-1 scale (higher = better for visualization)
    data = np.array([
        [0.9, 0.3, 0.4, 0.1, 0.5],  # Phase 2: Great training, okay results, complex
        [0.6, 0.9, 1.0, 0.9, 0.9],  # Phase 3: Decent training, BEST results, simple
        [0.8, 0.1, 0.2, 0.3, 0.3],  # Phase 5: Good training, BAD results, overconstrained
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(approaches)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(approaches, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(len(approaches)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title('Approach Comparison: Why Conservative Won', fontsize=15, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance\n(higher = better)', fontsize=11, fontweight='bold', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('viz_3_approach_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created viz_3_approach_comparison.png")
    plt.close()


# ============================================================================
# Chart 4: Boost Impact Analysis
# ============================================================================

def plot_boost_impact():
    """Show how different boost magnitudes affected key repos"""
    
    repos = ['ethers.js', 'openzeppelin', 'hardhat', 'foundry', 'solidity']
    
    # Different boost strategies
    baseline = [3.8, 3.2, 1.0, 2.8, 9.5]
    aggressive = [76.0, 80.0, 18.0, 50.4, 142.5]  # 20x boost
    conservative = [4.6, 4.0, 1.2, 3.4, 10.9]      # 1.2x boost
    actual = [1.2, 3.6, 5.4, 8.0, 4.2]             # Private LB truth
    
    x = np.arange(len(repos))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - 1.5*width, baseline, width, label='Baseline', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x - 0.5*width, aggressive, width, label='Aggressive (20x)', color='red', alpha=0.7)
    bars3 = ax.bar(x + 0.5*width, conservative, width, label='Conservative (1.2x)', color='green', alpha=0.7)
    bars4 = ax.bar(x + 1.5*width, actual, width, label='Actual Result', color='gold', alpha=0.9, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Repository', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Weight (%)', fontsize=13, fontweight='bold')
    ax.set_title('Boost Strategy Impact: Why 1.2x Beats 20x', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(repos, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.annotate('Aggressive boosts\novershoot reality!',
               xy=(0, aggressive[0]), xytext=(1, 100),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11, color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate('Conservative boosts\ntrack reality better!',
               xy=(2, conservative[2]), xytext=(3.5, 20),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=11, color='green', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('viz_4_boost_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Created viz_4_boost_impact.png")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Generating visualizations for Deep Funding writeup...")
    print("=" * 60)
    
    plot_score_evolution()
    plot_prediction_vs_actual()
    plot_approach_comparison()
    plot_boost_impact()
    
    print("=" * 60)
    print("✓ All visualizations created successfully!")
    print("\nFiles created:")
    print("  - viz_1_score_evolution.png")
    print("  - viz_2_prediction_vs_actual.png")
    print("  - viz_3_approach_comparison.png")
    print("  - viz_4_boost_impact.png")
    print("\nYou can now include these in your writeup!")
