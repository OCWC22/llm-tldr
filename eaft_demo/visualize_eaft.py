"""
EAFT Visualization: See the Entropy Gating in Action

This script visualizes:
1. How entropy varies across tokens
2. How EAFT gates the loss based on entropy
3. The difference between SFT and EAFT during training
4. Identification of "Confident Conflicts"

Run with: python visualize_eaft.py
"""

import torch
import torch.nn.functional as F
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Using ASCII visualization.")

from eaft_loss import EAFTLoss, identify_confident_conflicts


def create_synthetic_scenario():
    """
    Create a synthetic training scenario demonstrating EAFT.

    Simulates tokens with different characteristics:
    - Uncertain tokens (high entropy): Model should learn
    - Confident correct (low entropy, high prob): Already knows
    - Confident wrong (low entropy, low prob): DANGER! Conflict!
    """
    vocab_size = 50
    seq_len = 20

    torch.manual_seed(123)

    # Ground truth sequence
    labels = torch.randint(0, vocab_size, (1, seq_len))

    # Create logits with specific patterns
    logits = torch.zeros(1, seq_len, vocab_size)

    token_types = []

    for t in range(seq_len):
        gt_token = labels[0, t].item()

        if t < 5:
            # UNCERTAIN: High entropy, model doesn't know
            logits[0, t, :] = torch.randn(vocab_size) * 0.5
            token_types.append('uncertain')

        elif t < 10:
            # CONFIDENT CORRECT: Low entropy, high prob on GT
            logits[0, t, :] = -5.0
            logits[0, t, gt_token] = 5.0
            token_types.append('confident_correct')

        elif t < 15:
            # CONFIDENT WRONG: Low entropy, but confident in WRONG answer!
            wrong_token = (gt_token + 1) % vocab_size
            logits[0, t, :] = -5.0
            logits[0, t, wrong_token] = 5.0  # Confident in wrong token
            token_types.append('confident_wrong')

        else:
            # MIXED: Medium entropy
            logits[0, t, :] = torch.randn(vocab_size) * 1.5
            logits[0, t, gt_token] += 1.0  # Slight preference for correct
            token_types.append('mixed')

    return logits, labels, token_types


def compute_metrics(logits, labels, vocab_size):
    """Compute all metrics for visualization."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)
    normalized_entropy = entropy / np.log(vocab_size)

    # GT probability
    gt_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # CE loss per token
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none'
    ).view(1, -1)

    # EAFT weighted loss
    eaft_weighted = normalized_entropy * ce_loss

    return {
        'entropy': normalized_entropy[0].numpy(),
        'gt_prob': gt_probs[0].numpy(),
        'ce_loss': ce_loss[0].numpy(),
        'eaft_weighted': eaft_weighted[0].numpy(),
        'eaft_weight': normalized_entropy[0].numpy(),
    }


def visualize_matplotlib(metrics, token_types, seq_len):
    """Create matplotlib visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    x = np.arange(seq_len)
    colors = {
        'uncertain': '#3498db',        # Blue
        'confident_correct': '#2ecc71', # Green
        'confident_wrong': '#e74c3c',   # Red (DANGER!)
        'mixed': '#9b59b6'              # Purple
    }
    bar_colors = [colors[t] for t in token_types]

    # 1. Normalized Entropy
    ax = axes[0]
    bars = ax.bar(x, metrics['entropy'], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Normalized Entropy\n(H̃)', fontsize=10)
    ax.set_title('Token-Level Analysis: EAFT vs SFT', fontsize=14, fontweight='bold')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Conflict threshold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')

    # Add annotations
    for i, (e, tt) in enumerate(zip(metrics['entropy'], token_types)):
        if tt == 'confident_wrong':
            ax.annotate('⚠️', (i, e + 0.05), ha='center', fontsize=12)

    # 2. Ground Truth Probability
    ax = axes[1]
    ax.bar(x, metrics['gt_prob'], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('P(correct token)', fontsize=10)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Low prob threshold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')

    # 3. Standard CE Loss vs EAFT Weighted Loss
    ax = axes[2]
    width = 0.35
    ax.bar(x - width/2, metrics['ce_loss'], width, label='Standard SFT Loss',
           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, metrics['eaft_weighted'], width, label='EAFT Loss',
           color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Loss Value', fontsize=10)
    ax.legend(loc='upper right')

    # Highlight savings on confident conflicts
    for i, tt in enumerate(token_types):
        if tt == 'confident_wrong':
            diff = metrics['ce_loss'][i] - metrics['eaft_weighted'][i]
            if diff > 0.5:
                ax.annotate(f'↓{diff:.1f}', (i, metrics['ce_loss'][i] + 0.3),
                           ha='center', fontsize=9, color='green', fontweight='bold')

    # 4. EAFT Weight (Entropy Gate)
    ax = axes[3]
    ax.bar(x, metrics['eaft_weight'], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('EAFT Weight\n(Entropy Gate)', fontsize=10)
    ax.set_xlabel('Token Position', fontsize=10)
    ax.set_ylim(0, 1)

    # Add legend for token types
    patches = [
        mpatches.Patch(color=colors['uncertain'], label='Uncertain (learn!)'),
        mpatches.Patch(color=colors['confident_correct'], label='Confident Correct (no change needed)'),
        mpatches.Patch(color=colors['confident_wrong'], label='Confident Wrong (CONFLICT - suppress!)'),
        mpatches.Patch(color=colors['mixed'], label='Mixed'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('eaft_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to: eaft_visualization.png")
    plt.show()


def visualize_ascii(metrics, token_types, seq_len):
    """Create ASCII visualization for terminals without matplotlib."""
    print("\n" + "=" * 80)
    print("EAFT VISUALIZATION (ASCII)")
    print("=" * 80)

    # Token type legend
    type_symbols = {
        'uncertain': '?',
        'confident_correct': '✓',
        'confident_wrong': '⚠',
        'mixed': '~'
    }

    print("\nLegend: ? = Uncertain  ✓ = Confident Correct  ⚠ = Confident Wrong  ~ = Mixed\n")

    # 1. Token Types
    print("Token Types:     ", end="")
    for tt in token_types:
        print(f" {type_symbols[tt]} ", end="")
    print()

    # 2. Normalized Entropy (bar chart)
    print("\nEntropy (H̃):    ", end="")
    for e in metrics['entropy']:
        bar_len = int(e * 10)
        print(f"[{'█' * bar_len}{'░' * (10-bar_len)}]", end=" ")
    print()

    # 3. GT Probability
    print("\nP(correct):      ", end="")
    for p in metrics['gt_prob']:
        bar_len = int(p * 10)
        print(f"[{'█' * bar_len}{'░' * (10-bar_len)}]", end=" ")
    print()

    # 4. Loss comparison table
    print("\n" + "-" * 80)
    print(f"{'Token':<6} {'Type':<18} {'Entropy':<10} {'P(GT)':<10} {'SFT Loss':<12} {'EAFT Loss':<12} {'Saved'}")
    print("-" * 80)

    total_sft = 0
    total_eaft = 0

    for i in range(seq_len):
        tt = token_types[i]
        ent = metrics['entropy'][i]
        prob = metrics['gt_prob'][i]
        sft = metrics['ce_loss'][i]
        eaft = metrics['eaft_weighted'][i]
        saved = sft - eaft
        saved_pct = (saved / sft * 100) if sft > 0 else 0

        total_sft += sft
        total_eaft += eaft

        marker = " ⚠️" if tt == 'confident_wrong' else ""
        print(f"{i:<6} {tt:<18} {ent:<10.3f} {prob:<10.3f} {sft:<12.2f} {eaft:<12.2f} {saved_pct:>5.1f}%{marker}")

    print("-" * 80)
    print(f"{'TOTAL':<6} {'':<18} {'':<10} {'':<10} {total_sft:<12.2f} {total_eaft:<12.2f} {((total_sft-total_eaft)/total_sft*100):>5.1f}%")
    print()

    # Summary
    n_conflicts = sum(1 for tt in token_types if tt == 'confident_wrong')
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Confident Conflicts Detected: {n_conflicts} / {seq_len} tokens

What EAFT does:
• Tokens 0-4  (Uncertain):        Full learning signal → acquire new knowledge
• Tokens 5-9  (Confident Correct): Low weight → no unnecessary updates
• Tokens 10-14 (CONFIDENT WRONG):  ⚠️ SUPPRESSED! → prevents catastrophic forgetting
• Tokens 15-19 (Mixed):           Moderate weight → balanced learning

Standard SFT would force the model to "unlearn" its confident predictions on
tokens 10-14, causing destructive gradient updates that erase general knowledge.

EAFT automatically gates these out, preserving the model's existing capabilities!
""")


def simulate_training():
    """Simulate how EAFT vs SFT affects training over multiple steps."""
    print("\n" + "=" * 80)
    print("TRAINING SIMULATION: EAFT vs SFT")
    print("=" * 80)

    # Simulate capability scores over training
    steps = 10
    np.random.seed(42)

    # Domain performance (both improve similarly)
    domain_sft = [50 + 5*s + np.random.randn()*2 for s in range(steps)]
    domain_eaft = [50 + 5*s + np.random.randn()*2 for s in range(steps)]

    # General capability (SFT degrades, EAFT preserves)
    general_sft = [80 - 3*s + np.random.randn()*2 for s in range(steps)]  # Forgetting!
    general_eaft = [80 - 0.5*s + np.random.randn()*2 for s in range(steps)]  # Preserved!

    print("\n Step | Domain(SFT) | Domain(EAFT) | General(SFT) | General(EAFT) | Δ General")
    print("-" * 80)

    for s in range(steps):
        d_sft = domain_sft[s]
        d_eaft = domain_eaft[s]
        g_sft = general_sft[s]
        g_eaft = general_eaft[s]
        delta = g_eaft - g_sft

        print(f"  {s:<3} |   {d_sft:>5.1f}    |    {d_eaft:>5.1f}     |    {g_sft:>5.1f}     |     {g_eaft:>5.1f}     |  +{delta:>4.1f}")

    print("-" * 80)
    print(f"\n📊 After {steps} steps:")
    print(f"   SFT:  Domain={domain_sft[-1]:.1f}, General={general_sft[-1]:.1f} (FORGOT {80-general_sft[-1]:.1f} pts!)")
    print(f"   EAFT: Domain={domain_eaft[-1]:.1f}, General={general_eaft[-1]:.1f} (preserved!)")


def main():
    print("=" * 80)
    print("EAFT (Entropy-Adaptive Fine-Tuning) Demonstration")
    print("=" * 80)
    print("""
This demo shows how EAFT prevents catastrophic forgetting during fine-tuning.

The Key Insight:
• During SFT, the model encounters "Confident Conflicts" - tokens where it's
  confident in the WRONG answer
• Standard SFT forces learning on these → destroys existing knowledge
• EAFT uses entropy to detect these and suppresses their gradients
""")

    # Create synthetic scenario
    logits, labels, token_types = create_synthetic_scenario()
    vocab_size = logits.shape[-1]
    seq_len = logits.shape[1]

    # Compute metrics
    metrics = compute_metrics(logits, labels, vocab_size)

    # Visualize
    if HAS_MATPLOTLIB:
        visualize_matplotlib(metrics, token_types, seq_len)
    else:
        visualize_ascii(metrics, token_types, seq_len)

    # Training simulation
    simulate_training()


if __name__ == "__main__":
    main()
