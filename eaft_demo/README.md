# EAFT Demo: Entropy-Adaptive Fine-Tuning

This demo implements and visualizes EAFT from the paper:
**"Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting"**
(arXiv:2601.02151)

## Quick Start

```bash
# Basic loss function demo
python eaft_loss.py

# Visualization (ASCII or matplotlib)
python visualize_eaft.py

# Computer Use Agent example
python eaft_for_agents.py
```

## What is EAFT?

Standard SFT causes **catastrophic forgetting** because it forces the model to learn from
tokens where it's already confident but wrong ("Confident Conflicts").

**EAFT uses entropy to detect and suppress these harmful updates:**

```
L_EAFT = -∑ H̃_t · log P(y_t | x, y_{<t})
         ↑
   Entropy Gate (0-1)

- High entropy → uncertain → learn normally
- Low entropy → confident → suppress gradient
```

## Files

| File | Description |
|------|-------------|
| `eaft_loss.py` | Core EAFT loss implementation |
| `visualize_eaft.py` | Visualization of entropy gating |
| `eaft_for_agents.py` | Application to computer use agents |

## For Computer Use Agents

If you're training agents for Android/iOS that need to:
- Log into websites
- Navigate UI
- Select models, generate content

EAFT helps preserve general reasoning while learning UI-specific actions.

## Resources

- Paper: [arXiv:2601.02151](https://arxiv.org/abs/2601.02151)
- Official repo: [github.com/PRIS-CV/EAFT](https://github.com/PRIS-CV/EAFT)
- LlamaFactory integration: `use_eaft_loss=True`
