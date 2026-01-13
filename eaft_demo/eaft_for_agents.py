"""
EAFT for Computer Use Agents (Android/iOS)

This module shows how EAFT applies to training computer use agents like those
for Android or iPhone that need to:
- Log into websites (e.g., Hicksfield.ai)
- Navigate UI
- Generate videos, select models, etc.

Why EAFT matters for agents:
1. Base LLMs have strong general reasoning/text capabilities
2. Fine-tuning on UI actions often destroys this general knowledge
3. EAFT preserves reasoning while learning domain-specific UI navigation

The key challenge: Agent training data often contains UI-specific tokens
that conflict with the model's text priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Dict
from eaft_loss import EAFTLoss, identify_confident_conflicts


@dataclass
class AgentAction:
    """Represents a single agent action in a trajectory."""
    action_type: str  # 'tap', 'type', 'scroll', 'wait', 'think'
    target: str       # Element ID or text
    value: Optional[str] = None  # For 'type' actions
    reasoning: Optional[str] = None  # Model's reasoning


@dataclass
class AgentTrajectory:
    """A complete trajectory of agent actions."""
    task: str  # e.g., "Log into Hicksfield.ai and generate a video"
    actions: List[AgentAction]
    success: bool


# Example trajectories for computer use agent
EXAMPLE_TRAJECTORIES = [
    AgentTrajectory(
        task="Log into Hicksfield.ai and generate a video",
        actions=[
            AgentAction("think", "", reasoning="I need to open the browser and navigate to Hicksfield.ai"),
            AgentAction("tap", "browser_icon"),
            AgentAction("tap", "url_bar"),
            AgentAction("type", "url_bar", value="hicksfield.ai"),
            AgentAction("tap", "login_button"),
            AgentAction("type", "email_field", value="user@example.com"),
            AgentAction("type", "password_field", value="***"),
            AgentAction("tap", "submit_button"),
            AgentAction("think", "", reasoning="Now I need to find the video generation feature"),
            AgentAction("tap", "generate_video_tab"),
            AgentAction("tap", "model_selector"),
            AgentAction("tap", "model_option_1"),
            AgentAction("tap", "generate_button"),
            AgentAction("wait", "loading_indicator", reasoning="Waiting for video generation"),
        ],
        success=True
    ),
    AgentTrajectory(
        task="Select a different model and adjust settings",
        actions=[
            AgentAction("tap", "settings_icon"),
            AgentAction("scroll", "settings_panel", value="down"),
            AgentAction("tap", "model_dropdown"),
            AgentAction("tap", "advanced_model"),
            AgentAction("tap", "quality_slider"),
            AgentAction("type", "quality_input", value="high"),
            AgentAction("tap", "save_button"),
        ],
        success=True
    )
]


class AgentTokenizer:
    """Simple tokenizer for agent actions (demonstration purposes)."""

    SPECIAL_TOKENS = {
        '<|action|>': 0,
        '<|target|>': 1,
        '<|value|>': 2,
        '<|think|>': 3,
        '<|end|>': 4,
        '<|pad|>': 5,
    }

    ACTION_TOKENS = {
        'tap': 10, 'type': 11, 'scroll': 12, 'wait': 13, 'think': 14
    }

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = self.SPECIAL_TOKENS['<|pad|>']

    def encode_action(self, action: AgentAction) -> List[int]:
        """Encode a single action into tokens."""
        tokens = [self.SPECIAL_TOKENS['<|action|>']]
        tokens.append(self.ACTION_TOKENS.get(action.action_type, 15))
        tokens.append(self.SPECIAL_TOKENS['<|target|>'])
        tokens.append(hash(action.target) % (self.vocab_size - 100) + 100)

        if action.value:
            tokens.append(self.SPECIAL_TOKENS['<|value|>'])
            for char in action.value[:10]:
                tokens.append(ord(char) % (self.vocab_size - 100) + 100)

        if action.reasoning:
            tokens.append(self.SPECIAL_TOKENS['<|think|>'])
            for word in action.reasoning.split()[:5]:
                tokens.append(hash(word) % (self.vocab_size - 100) + 100)

        tokens.append(self.SPECIAL_TOKENS['<|end|>'])
        return tokens

    def encode_trajectory(self, trajectory: AgentTrajectory, max_len: int = 100) -> torch.Tensor:
        """Encode a complete trajectory."""
        tokens = []
        for action in trajectory.actions:
            tokens.extend(self.encode_action(action))

        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return torch.tensor(tokens)


class SimpleAgentModel(nn.Module):
    """
    A simple agent model for demonstration.

    In practice, this would be a full LLM like Qwen or LLaMA.
    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, batch_first=True),
            num_layers=num_layers
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.output_head(x)
        return logits


def compare_training_methods(
    model: nn.Module,
    trajectories: List[AgentTrajectory],
    tokenizer: AgentTokenizer,
    epochs: int = 5,
    lr: float = 1e-3
):
    """
    Compare EAFT vs standard SFT training on agent data.

    This demonstrates how EAFT preserves capabilities while learning.
    """
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON: EAFT vs SFT for Computer Use Agent")
    print("=" * 70)

    # Prepare data
    max_len = 80
    batch_tokens = torch.stack([
        tokenizer.encode_trajectory(traj, max_len) for traj in trajectories
    ])

    # Create labels (next token prediction)
    input_ids = batch_tokens[:, :-1]
    labels = batch_tokens[:, 1:]

    # Initialize losses
    sft_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    eaft_loss_fn = EAFTLoss(
        vocab_size=tokenizer.vocab_size,
        ignore_index=tokenizer.pad_token_id
    )

    # Clone model for fair comparison
    import copy
    model_sft = copy.deepcopy(model)
    model_eaft = copy.deepcopy(model)

    opt_sft = torch.optim.Adam(model_sft.parameters(), lr=lr)
    opt_eaft = torch.optim.Adam(model_eaft.parameters(), lr=lr)

    print("\nTraining Progress:")
    print("-" * 70)
    print(f"{'Epoch':<8} {'SFT Loss':<12} {'EAFT Loss':<12} {'Conflicts':<12} {'Suppressed'}")
    print("-" * 70)

    for epoch in range(epochs):
        # SFT training step
        model_sft.train()
        logits_sft = model_sft(input_ids)
        loss_sft = sft_loss_fn(
            logits_sft.view(-1, tokenizer.vocab_size),
            labels.view(-1)
        )
        opt_sft.zero_grad()
        loss_sft.backward()
        opt_sft.step()

        # EAFT training step
        model_eaft.train()
        logits_eaft = model_eaft(input_ids)
        loss_eaft, components = eaft_loss_fn(logits_eaft, labels, return_components=True)
        opt_eaft.zero_grad()
        loss_eaft.backward()
        opt_eaft.step()

        # Count confident conflicts
        conflicts = identify_confident_conflicts(logits_eaft, labels, ignore_index=tokenizer.pad_token_id)
        n_conflicts = conflicts.sum().item()
        mask = labels != tokenizer.pad_token_id
        total_tokens = mask.sum().item()

        # Calculate suppression (how much gradient was reduced)
        avg_weight = (components['entropy_weights'] * mask).sum() / mask.sum()
        suppressed = (1 - avg_weight.item()) * 100

        print(f"{epoch+1:<8} {loss_sft.item():<12.4f} {loss_eaft.item():<12.4f} {n_conflicts:<12} {suppressed:>6.1f}%")

    print("-" * 70)

    # Analyze token-level differences
    print("\n" + "=" * 70)
    print("TOKEN-LEVEL ANALYSIS: Where EAFT helps")
    print("=" * 70)

    with torch.no_grad():
        logits = model_eaft(input_ids)
        conflicts = identify_confident_conflicts(logits, labels, ignore_index=tokenizer.pad_token_id)

    # Find conflict positions
    conflict_positions = torch.where(conflicts[0])[0].tolist()

    if conflict_positions:
        print(f"\nConfident Conflicts found at positions: {conflict_positions}")
        print("\nThese are tokens where the model was confident but wrong.")
        print("Standard SFT would force unlearning → catastrophic forgetting!")
        print("EAFT suppresses these gradients → preserves general capabilities!")
    else:
        print("\nNo major confident conflicts in this batch.")

    return model_sft, model_eaft


def demonstrate_agent_eaft():
    """Main demonstration function."""
    print("=" * 70)
    print("EAFT FOR COMPUTER USE AGENTS")
    print("=" * 70)
    print("""
Scenario: Training an agent to navigate UI (Android/iOS) for tasks like:
- Logging into Hicksfield.ai
- Generating videos
- Selecting models
- Adjusting settings

The Challenge:
- Base LLM has strong text/reasoning priors
- UI-specific tokens (tap, scroll, element IDs) may conflict with text priors
- Standard SFT destroys general capabilities

The Solution: EAFT
- Detects confident conflicts via entropy
- Suppresses harmful gradients on conflicting tokens
- Preserves general reasoning while learning UI navigation
""")

    # Initialize
    vocab_size = 1000
    tokenizer = AgentTokenizer(vocab_size=vocab_size)
    model = SimpleAgentModel(vocab_size=vocab_size)

    print("\n📱 Example Agent Trajectory:")
    print("-" * 50)
    traj = EXAMPLE_TRAJECTORIES[0]
    print(f"Task: {traj.task}\n")
    for i, action in enumerate(traj.actions[:5]):
        print(f"  {i+1}. {action.action_type}({action.target})", end="")
        if action.value:
            print(f" = '{action.value}'", end="")
        print()
    print("  ...")

    # Compare training
    model_sft, model_eaft = compare_training_methods(
        model=model,
        trajectories=EXAMPLE_TRAJECTORIES,
        tokenizer=tokenizer,
        epochs=5
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR YOUR AGENT")
    print("=" * 70)
    print("""
1. Use EAFT when fine-tuning on UI navigation data
   - Preserves text understanding and reasoning
   - Learns UI-specific actions effectively

2. EAFT is especially valuable for:
   - Login flows (text inputs conflict with general text priors)
   - Element selection (UI tokens may conflict)
   - Multi-step reasoning (preserves chain-of-thought)

3. Integration options:
   - LlamaFactory: use_eaft_loss=True
   - Custom: Use the EAFTLoss class from this demo

4. For your Hicksfield.ai agent:
   - Collect trajectories of successful interactions
   - Fine-tune with EAFT to learn UI navigation
   - General reasoning preserved → better error recovery!
""")


if __name__ == "__main__":
    demonstrate_agent_eaft()
