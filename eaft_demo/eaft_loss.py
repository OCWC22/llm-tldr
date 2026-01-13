"""
EAFT (Entropy-Adaptive Fine-Tuning) Loss Implementation

This is a simple, educational implementation of the EAFT loss function
from the paper "Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts
to Mitigate Forgetting" (arXiv:2601.02151)

The key insight: During SFT, "Confident Conflicts" (tokens where the model
is confident but wrong) cause catastrophic forgetting. EAFT uses entropy
to gate the loss - high entropy tokens learn normally, low entropy tokens
(where model is confident) have suppressed gradients.

Usage for Computer Use Agents:
- When fine-tuning an agent on UI navigation tasks (Android/iOS)
- The agent may have strong priors about general text/reasoning
- EAFT prevents forgetting those while learning domain-specific UI actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EAFTLoss(nn.Module):
    """
    Entropy-Adaptive Fine-Tuning Loss

    L_EAFT = -∑ H̃_t · log P(y_t | x, y_{<t})

    where H̃_t = H(P) / log(V) is normalized entropy
    """

    def __init__(self, vocab_size: int, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.log_vocab_size = torch.log(torch.tensor(vocab_size, dtype=torch.float32))

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy for each position.

        H(P) = -∑ P(v) log P(v)

        Args:
            logits: [batch, seq_len, vocab_size]

        Returns:
            entropy: [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # H = -∑ p * log(p)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def compute_normalized_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized entropy H̃ ∈ [0, 1]

        H̃ = H(P) / log(V)

        - H̃ → 0: Model is confident (low entropy)
        - H̃ → 1: Model is uncertain (high entropy / uniform distribution)
        """
        entropy = self.compute_entropy(logits)
        # Normalize by max possible entropy (uniform distribution)
        normalized = entropy / self.log_vocab_size.to(entropy.device)
        return normalized.clamp(0, 1)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Compute EAFT loss.

        Args:
            logits: [batch, seq_len, vocab_size] - model predictions
            labels: [batch, seq_len] - ground truth token IDs
            return_components: if True, return loss breakdown for visualization

        Returns:
            loss: scalar loss value
            components (optional): dict with per-token breakdown
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute normalized entropy for gating
        normalized_entropy = self.compute_normalized_entropy(logits)  # [batch, seq_len]

        # Compute per-token cross-entropy loss
        # Reshape for cross_entropy: [batch*seq, vocab] vs [batch*seq]
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Per-token CE loss (no reduction)
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(batch_size, seq_len)

        # Create mask for valid tokens
        mask = (labels != self.ignore_index).float()

        # EAFT: Weight loss by normalized entropy
        # High entropy (uncertain) → weight ≈ 1 → full learning
        # Low entropy (confident) → weight ≈ 0 → suppressed gradient
        weighted_loss = normalized_entropy * ce_loss * mask

        # Reduce
        if self.reduction == 'mean':
            loss = weighted_loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss

        if return_components:
            # Get model's predicted probability for ground truth tokens
            probs = F.softmax(logits, dim=-1)
            gt_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
            gt_probs = gt_probs * mask  # Mask padding

            components = {
                'normalized_entropy': normalized_entropy,
                'ce_loss': ce_loss,
                'weighted_loss': weighted_loss,
                'gt_probability': gt_probs,
                'mask': mask,
                'entropy_weights': normalized_entropy * mask,
            }
            return loss, components

        return loss


class StandardSFTLoss(nn.Module):
    """Standard SFT cross-entropy loss for comparison."""

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        return F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


def identify_confident_conflicts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    entropy_threshold: float = 0.15,
    prob_threshold: float = 0.1,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Identify "Confident Conflict" tokens.

    These are tokens where:
    - Low entropy (model is confident in SOME prediction)
    - Low probability on ground truth (but not confident in the RIGHT answer)

    These tokens cause catastrophic forgetting in standard SFT.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        entropy_threshold: below this = "confident"
        prob_threshold: below this = "wrong"

    Returns:
        conflict_mask: [batch, seq_len] boolean tensor
    """
    # Compute normalized entropy
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    log_vocab = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32, device=logits.device))
    normalized_entropy = entropy / log_vocab

    # Get probability of ground truth token
    gt_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)

    # Confident conflicts: low entropy AND low probability on correct answer
    valid_mask = labels != ignore_index
    confident = normalized_entropy < entropy_threshold
    wrong = gt_probs < prob_threshold

    conflict_mask = confident & wrong & valid_mask

    return conflict_mask


# Example usage and comparison
if __name__ == "__main__":
    print("=" * 60)
    print("EAFT vs Standard SFT Loss Comparison")
    print("=" * 60)

    # Simulate a small vocabulary and sequence
    vocab_size = 100
    batch_size = 2
    seq_len = 10

    torch.manual_seed(42)

    # Create sample logits with different confidence patterns
    # Token 0-2: High entropy (uncertain) - should learn normally
    # Token 3-5: Low entropy, high prob on GT - already knows, minimal update
    # Token 6-9: Low entropy, low prob on GT - CONFIDENT CONFLICT!

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Make some tokens confident conflicts (model confident but wrong)
    for b in range(batch_size):
        for t in range(6, 10):
            # Low entropy: put most probability mass on wrong token
            wrong_token = (labels[b, t].item() + 1) % vocab_size
            logits[b, t, :] = -10  # Low prob everywhere
            logits[b, t, wrong_token] = 10  # High prob on WRONG token

    # Make some tokens uncertain (uniform-ish)
    for b in range(batch_size):
        for t in range(0, 3):
            logits[b, t, :] = torch.randn(vocab_size)  # More uniform

    # Initialize losses
    eaft_loss = EAFTLoss(vocab_size=vocab_size)
    sft_loss = StandardSFTLoss()

    # Compute losses
    sft_value = sft_loss(logits, labels)
    eaft_value, components = eaft_loss(logits, labels, return_components=True)

    print(f"\nStandard SFT Loss: {sft_value.item():.4f}")
    print(f"EAFT Loss:         {eaft_value.item():.4f}")
    print(f"Reduction:         {((sft_value - eaft_value) / sft_value * 100).item():.1f}%")

    # Identify confident conflicts
    conflicts = identify_confident_conflicts(logits, labels)

    print(f"\n{'='*60}")
    print("Per-Token Analysis (Batch 0)")
    print("=" * 60)
    print(f"{'Token':<6} {'Entropy':<10} {'GT Prob':<10} {'CE Loss':<10} {'EAFT Wt':<10} {'Conflict'}")
    print("-" * 60)

    for t in range(seq_len):
        ent = components['normalized_entropy'][0, t].item()
        gt_p = components['gt_probability'][0, t].item()
        ce = components['ce_loss'][0, t].item()
        wt = components['entropy_weights'][0, t].item()
        conf = "⚠️ YES" if conflicts[0, t].item() else ""

        print(f"{t:<6} {ent:<10.4f} {gt_p:<10.4f} {ce:<10.2f} {wt:<10.4f} {conf}")

    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("""
Tokens 6-9 are CONFIDENT CONFLICTS:
- Model has LOW entropy (very confident in its prediction)
- But LOW probability on the ground truth (confident in WRONG answer)

Standard SFT: Applies full gradient → destroys existing knowledge
EAFT:         Entropy gate ≈ 0 → suppresses harmful gradients

This preserves general capabilities while still learning from uncertain tokens!
""")
