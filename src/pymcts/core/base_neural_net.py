"""Abstract base class for neural networks in the AlphaZero engine."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pymcts.core.base_game import GameState


def _best_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseNeuralNet(ABC, nn.Module):
    """Base class for all neural networks. Inherits from ABC and nn.Module.

    The developer implements encode() and forward(). predict(), predict_batch(),
    and train_on_examples() have sensible defaults.

    Automatically moves to the best available device (CUDA > MPS > CPU)
    after construction.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def _wrapped_init(self, *args, **kw):
            original_init(self, *args, **kw)
            device = getattr(self, '_force_device', None) or _best_device()
            self.to(device)

        cls.__init__ = _wrapped_init

    @abstractmethod
    def encode(self, state: GameState) -> torch.Tensor:
        """Convert a GameState into the tensor format this architecture expects.
        Should return a CPU tensor — the base class handles device transfer.
        """

    def encode_batch(self, states: list[GameState]) -> torch.Tensor:
        """Encode multiple states into a single batched tensor.
        Override in subclasses for more efficient batch encoding.
        """
        return torch.stack([self.encode(s) for s in states])

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw forward pass. Input: (batch, *encoded_shape).
        Returns: (log_policy (batch, action_space_size), value (batch, 1)).
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def copy(self) -> "BaseNeuralNet": ...

    @property
    def device(self) -> torch.device:
        """Return the device this model's parameters live on."""
        return next(self.parameters()).device

    def to_best_device(self) -> "BaseNeuralNet":
        """Move this model to the best available device."""
        return self.to(_best_device())

    def predict(self, state: GameState) -> tuple[torch.Tensor, float]:
        """Single state -> (policy_1D, value). Returns CPU tensors."""
        self.eval()
        with torch.no_grad():
            tensor = self.encode(state).unsqueeze(0).to(self.device)
            policy, value = self.forward(tensor)
        return policy.squeeze(0).cpu(), value.item()

    def predict_batch(self, states: list[GameState]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch of states -> (policies, values). Returns CPU tensors."""
        self.eval()
        with torch.no_grad():
            tensors = self.encode_batch(states).to(self.device)
            policies, values = self.forward(tensors)
        return policies.cpu(), values.squeeze(-1).cpu()

    def train_on_examples(
        self,
        examples: list[tuple[GameState, torch.Tensor, float]],
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Default training loop: cross-entropy policy loss + MSE value loss.

        Returns metrics dict with final epoch losses.
        """
        device = self.device
        states = self.encode_batch([s for s, _, _ in examples])
        policies = torch.stack([p for _, p, _ in examples])
        values = torch.tensor([v for _, _, v in examples], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

        self.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        epoch_iter = range(num_epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=True)

        for epoch in epoch_iter:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                batch_states = batch_states.to(device)
                batch_policies = batch_policies.to(device)
                batch_values = batch_values.to(device)

                log_policy, value = self.forward(batch_states)
                policy_loss = -torch.sum(batch_policies * log_policy) / batch_states.size(0)
                value_loss = F.mse_loss(value, batch_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

            avg_pi = total_policy_loss / max(num_batches, 1)
            avg_v = total_value_loss / max(num_batches, 1)
            if verbose:
                epoch_iter.set_postfix(
                    pi_loss=f"{avg_pi:.4f}",
                    v_loss=f"{avg_v:.4f}",
                    loss=f"{avg_pi + avg_v:.4f}",
                )

        return {
            "policy_loss": avg_pi,
            "value_loss": avg_v,
            "loss": avg_pi + avg_v,
        }
