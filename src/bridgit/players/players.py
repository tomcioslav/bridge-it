"""Player abstractions for different playing strategies."""

from abc import ABC, abstractmethod
import numpy as np

from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import NetWrapper
from bridgit.game import Bridgit
from bridgit.schema import Move
from bridgit.config import BoardConfig, MCTSConfig


class BasePlayer(ABC):
    """Abstract base class for all players."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def get_action(self, game: Bridgit) -> Move:
        """Select a move (row, col) for the current game state."""
        pass

    def __repr__(self) -> str:
        return self.name


class RandomPlayer(BasePlayer):
    """Player that selects moves uniformly at random."""

    def __init__(self, board: BoardConfig = BoardConfig(), name: str | None = None):
        super().__init__(name)

    def get_action(self, game: Bridgit) -> Move:
        """Select a random legal move."""
        moves = game.get_available_moves()
        if not moves:
            raise ValueError("No valid moves available")
        idx = np.random.randint(len(moves))
        return moves[idx]


class MCTSPlayer(BasePlayer):
    """Player that uses MCTS with neural network guidance."""

    def __init__(
        self,
        net_wrapper: NetWrapper,
        mcts: MCTSConfig,
        board: BoardConfig,
        temperature: float = 1.0,
        name: str | None = None
    ):
        super().__init__(name)
        self.mcts_search = MCTS(net_wrapper, mcts, board)
        self.temperature = temperature

    def get_action(self, game: Bridgit) -> Move:
        """Select move using MCTS."""
        probs = self.mcts_search.get_action_probs(game, temperature=self.temperature)

        # Sample from 2D probability distribution
        flat_probs = probs.flatten()
        if flat_probs.sum() > 0:
            flat_probs = flat_probs / flat_probs.sum()
        else:
            # Fallback to uniform over valid moves
            mask = game.to_mask().numpy()
            flat_probs = mask.flatten()
            flat_probs = flat_probs / flat_probs.sum()

        flat_idx = np.random.choice(len(flat_probs), p=flat_probs)
        row, col = np.unravel_index(flat_idx, probs.shape)
        return Move(row=int(row), col=int(col))


class GreedyMCTSPlayer(MCTSPlayer):
    """MCTS player that always picks the most-visited move (temperature=0)."""

    def __init__(self, net_wrapper: NetWrapper, mcts: MCTSConfig, board: BoardConfig, name: str | None = None):
        super().__init__(net_wrapper, mcts, board, temperature=0.0, name=name)


