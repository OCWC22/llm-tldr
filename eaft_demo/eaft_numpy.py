#!/usr/bin/env python3
"""
EAFT (Entropy-Adaptive Fine-Tuning) - Pure NumPy Implementation

No dependencies required beyond NumPy (usually pre-installed).
This demonstrates the core EAFT algorithm for educational purposes.

Run with: python eaft_numpy.py
"""

import numpy as np
from typing import Tuple, Dict, List

# ============================================================================
# CORE EAFT IMPLEMENTATION
# ============================================================================

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log softmax."""
    return logits - np.max(logits, axis=axis, keepdims=True) - \
           np.log(np.sum(np.exp(logits - np.max(logits, axis=axis, keepdims=True)), axis=axis, keepdims=True))


def compute_entropy(logits: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy: H(P) = -∑ P(v) log P(v)

    Args:
        logits: [seq_len, vocab_size]
    Returns:
        entropy: [seq_len]
    """
    probs = softmax(logits)
    log_probs = log_softmax(logits)
    entropy = -np.sum(probs * log_probs, axis=-1)
    return entropy


def compute_normalized_entropy(logits: np.ndarray) -> np.ndarray:
    """
    Compute normalized entropy H̃ ∈ [0, 1]

    H̃ = H(P) / log(V)

    - H̃ → 0: Model is confident (sharp distribution)
    - H̃ → 1: Model is uncertain (uniform distribution)
    """
    vocab_size = logits.shape[-1]
    entropy = compute_entropy(logits)
    max_entropy = np.log(vocab_size)
    return np.clip(entropy / max_entropy, 0, 1)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Standard cross-entropy loss per token.

    CE(y) = -log P(y)
    """
    log_probs = log_softmax(logits)
    # Gather the log prob of the correct label at each position
    seq_len = len(labels)
    ce_loss = np.array([-log_probs[i, labels[i]] for i in range(seq_len)])
    return ce_loss


def eaft_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    """
    EAFT Loss: Entropy-weighted cross-entropy.

    L_EAFT = -∑ H̃_t · log P(y_t | context)

    Args:
        logits: [seq_len, vocab_size]
        labels: [seq_len]

    Returns:
        loss: scalar
        components: dict with per-token breakdown
    """
    # Normalized entropy (the gating mechanism)
    H_normalized = compute_normalized_entropy(logits)

    # Standard CE loss per token
    ce = cross_entropy_loss(logits, labels)

    # EAFT: weight CE by entropy
    weighted_ce = H_normalized * ce

    # Get probability of ground truth
    probs = softmax(logits)
    gt_probs = np.array([probs[i, labels[i]] for i in range(len(labels))])

    components = {
        'normalized_entropy': H_normalized,
        'ce_loss': ce,
        'eaft_weighted_loss': weighted_ce,
        'gt_probability': gt_probs,
    }

    return weighted_ce.mean(), components


def sft_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Standard SFT cross-entropy loss."""
    return cross_entropy_loss(logits, labels).mean()


def identify_confident_conflicts(
    logits: np.ndarray,
    labels: np.ndarray,
    entropy_threshold: float = 0.15,
    prob_threshold: float = 0.1
) -> np.ndarray:
    """
    Find "Confident Conflicts" - tokens where:
    - Model has LOW entropy (confident)
    - Model has LOW probability on ground truth (wrong)

    These are the tokens that cause catastrophic forgetting!
    """
    H_norm = compute_normalized_entropy(logits)
    probs = softmax(logits)
    gt_probs = np.array([probs[i, labels[i]] for i in range(len(labels))])

    confident = H_norm < entropy_threshold
    wrong = gt_probs < prob_threshold

    return confident & wrong


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_bar(value: float, width: int = 20, fill: str = '█', empty: str = '░') -> str:
    """Create an ASCII bar chart."""
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def visualize_comparison(logits: np.ndarray, labels: np.ndarray, token_types: List[str]):
    """Visualize EAFT vs SFT with ASCII graphics."""

    sft_val = sft_loss(logits, labels)
    eaft_val, components = eaft_loss(logits, labels)
    conflicts = identify_confident_conflicts(logits, labels)

    seq_len = len(labels)

    print("\n" + "=" * 80)
    print("EAFT vs SFT COMPARISON")
    print("=" * 80)

    # Summary stats
    print(f"\n📊 LOSS VALUES:")
    print(f"   Standard SFT Loss: {sft_val:.4f}")
    print(f"   EAFT Loss:         {eaft_val:.4f}")
    print(f"   Reduction:         {((sft_val - eaft_val) / sft_val * 100):.1f}%")
    print(f"   Conflicts found:   {conflicts.sum()} / {seq_len}")

    # Token type legend
    type_colors = {
        'uncertain': '🔵',
        'confident_correct': '🟢',
        'confident_wrong': '🔴',
        'mixed': '🟣'
    }

    print(f"\n📝 LEGEND:")
    print(f"   🔵 Uncertain      - Model doesn't know → LEARN (high weight)")
    print(f"   🟢 Confident OK   - Model already knows → minimal update")
    print(f"   🔴 CONFLICT       - Model confident but WRONG → SUPPRESS!")
    print(f"   🟣 Mixed          - Some uncertainty → moderate learning")

    # Per-token table
    print("\n" + "-" * 80)
    print(f"{'Tok':<4} {'Type':<10} {'Entropy':<22} {'P(GT)':<22} {'SFT':<8} {'EAFT':<8} {'?'}")
    print("-" * 80)

    for i in range(seq_len):
        tt = token_types[i]
        icon = type_colors[tt]
        ent = components['normalized_entropy'][i]
        gt_p = components['gt_probability'][i]
        ce = components['ce_loss'][i]
        weighted = components['eaft_weighted_loss'][i]
        is_conflict = "⚠️" if conflicts[i] else ""

        ent_bar = create_bar(ent, 15)
        prob_bar = create_bar(gt_p, 15)

        print(f"{i:<4} {icon:<10} [{ent_bar}] [{prob_bar}] {ce:<8.2f} {weighted:<8.2f} {is_conflict}")

    print("-" * 80)

    # Totals
    total_sft = components['ce_loss'].sum()
    total_eaft = components['eaft_weighted_loss'].sum()
    saved = total_sft - total_eaft

    print(f"{'SUM':<4} {'':<10} {'':<22} {'':<22} {total_sft:<8.2f} {total_eaft:<8.2f}")
    print(f"\n💡 EAFT saved {saved:.2f} loss units ({saved/total_sft*100:.1f}% reduction)")

    return components


def create_synthetic_scenario(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a synthetic scenario demonstrating different token types.

    Returns logits shaped to show:
    - Uncertain tokens (high entropy)
    - Confident correct tokens (low entropy, high prob on GT)
    - Confident WRONG tokens (low entropy, low prob on GT) ← THE DANGER!
    - Mixed tokens
    """
    np.random.seed(seed)

    vocab_size = 50
    seq_len = 16

    # Ground truth labels
    labels = np.random.randint(0, vocab_size, size=seq_len)

    # Initialize logits
    logits = np.zeros((seq_len, vocab_size))
    token_types = []

    for t in range(seq_len):
        gt = labels[t]

        if t < 4:
            # UNCERTAIN: Nearly uniform distribution
            logits[t, :] = np.random.randn(vocab_size) * 0.3
            token_types.append('uncertain')

        elif t < 8:
            # CONFIDENT CORRECT: Sharp peak on ground truth
            logits[t, :] = -3.0
            logits[t, gt] = 5.0
            token_types.append('confident_correct')

        elif t < 12:
            # CONFIDENT WRONG: Sharp peak on WRONG token!
            wrong_token = (gt + 1) % vocab_size
            logits[t, :] = -3.0
            logits[t, wrong_token] = 5.0  # Confident in wrong answer!
            token_types.append('confident_wrong')

        else:
            # MIXED: Some preference but not too sharp
            logits[t, :] = np.random.randn(vocab_size) * 1.0
            logits[t, gt] += 1.5
            token_types.append('mixed')

    return logits, labels, token_types


def simulate_training():
    """Simulate training curves for EAFT vs SFT."""

    print("\n" + "=" * 80)
    print("TRAINING SIMULATION: General Capability Over Time")
    print("=" * 80)

    np.random.seed(123)
    epochs = 12

    # Simulated metrics
    # Domain: Both methods learn the domain task
    # General: SFT forgets, EAFT preserves

    domain_sft = [45 + 4.5*e + np.random.randn()*2 for e in range(epochs)]
    domain_eaft = [45 + 4.5*e + np.random.randn()*2 for e in range(epochs)]

    general_sft = [82 - 3.0*e + np.random.randn()*2 for e in range(epochs)]  # Forgetting!
    general_eaft = [82 - 0.5*e + np.random.randn()*2 for e in range(epochs)]  # Preserved!

    print("\n   Epoch   Domain(SFT)  Domain(EAFT)  General(SFT)  General(EAFT)")
    print("   " + "-" * 60)

    for e in range(epochs):
        # Visual bars
        d_sft = domain_sft[e]
        d_eaft = domain_eaft[e]
        g_sft = max(0, general_sft[e])
        g_eaft = general_eaft[e]

        print(f"   {e+1:>3}      {d_sft:>5.1f}        {d_eaft:>5.1f}         {g_sft:>5.1f}         {g_eaft:>5.1f}")

    print("   " + "-" * 60)

    # Final comparison
    print(f"\n   📈 FINAL RESULTS (Epoch {epochs}):")
    print(f"   ┌─────────────────────────────────────────────────┐")
    print(f"   │  Method  │  Domain Score  │  General Score      │")
    print(f"   ├─────────────────────────────────────────────────┤")
    print(f"   │  SFT     │     {domain_sft[-1]:>5.1f}       │     {max(0,general_sft[-1]):>5.1f}  ⚠️ FORGOT! │")
    print(f"   │  EAFT    │     {domain_eaft[-1]:>5.1f}       │     {general_eaft[-1]:>5.1f}  ✓ Preserved │")
    print(f"   └─────────────────────────────────────────────────┘")

    forgotten = 82 - max(0, general_sft[-1])
    preserved = 82 - general_eaft[-1]
    print(f"\n   SFT forgot {forgotten:.1f} points of general capability!")
    print(f"   EAFT only lost {preserved:.1f} points (preserved {((82-preserved)/82*100):.0f}%)")


def explain_for_agents():
    """Explain how EAFT applies to computer use agents."""

    print("\n" + "=" * 80)
    print("EAFT FOR COMPUTER USE AGENTS (Android/iOS)")
    print("=" * 80)
    print("""
    🎯 YOUR USE CASE: Training agents to navigate UI
       - Log into Hicksfield.ai
       - Generate videos
       - Select models
       - etc.

    ❌ THE PROBLEM WITH STANDARD SFT:

       Your base LLM knows language well. When you fine-tune it on UI tasks:

       Training data: "Tap login_button"
       Model thinks:  "login_button isn't a word I know well..."
                      → High entropy → Learns fine ✓

       Training data: "Type password: ****"
       Model thinks:  "I know 'password' means something else in text!"
                      → Low entropy, but forced to learn UI meaning
                      → CONFLICT! Destroys text understanding!

    ✅ HOW EAFT HELPS:

       EAFT detects when the model is confident but being forced to learn
       something contradictory. It automatically:

       1. Uncertain tokens  → Full learning (entropy gate ≈ 1)
       2. Conflict tokens   → Suppressed gradient (entropy gate ≈ 0)

       Result: Agent learns UI navigation WITHOUT forgetting how to reason!

    📝 PRACTICAL BENEFITS:

       • Better error recovery (still has general reasoning)
       • Handles novel situations (preserved language understanding)
       • More robust to UI changes (general knowledge helps generalize)

    🔧 HOW TO USE:

       Option 1: LlamaFactory (easiest)
       ```yaml
       use_eaft_loss: true
       ```

       Option 2: Custom training loop
       ```python
       from eaft_loss import EAFTLoss
       loss_fn = EAFTLoss(vocab_size=model.vocab_size)
       loss = loss_fn(logits, labels)
       ```
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  EAFT: Entropy-Adaptive Fine-Tuning")
    print("  Preventing Catastrophic Forgetting in LLM Fine-Tuning")
    print("=" * 80)
    print("""
  The Core Insight:

  During SFT, some tokens are "Confident Conflicts" - the model is SURE
  it knows the answer, but that answer contradicts the training label.

  Standard SFT: Forces the model to unlearn → Destroys existing knowledge!
  EAFT:         Detects conflicts via entropy → Suppresses harmful gradients

  Formula:

    Standard SFT:  L = -∑ log P(y_t)

    EAFT:          L = -∑ H̃_t · log P(y_t)
                       ↑
                 Entropy Gate
                 (0 = suppress, 1 = learn)
""")

    # Create synthetic scenario
    logits, labels, token_types = create_synthetic_scenario()

    # Visualize comparison
    visualize_comparison(logits, labels, token_types)

    # Training simulation
    simulate_training()

    # Agent explanation
    explain_for_agents()

    print("\n" + "=" * 80)
    print("  Run complete! See the visualization above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
