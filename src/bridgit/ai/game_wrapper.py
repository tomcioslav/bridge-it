"""Adapter between Bridgit game and the AI pipeline.

Works on the (2n+1)×(2n+1) grid. Actions are flat indices into the g×g board.
Only crossings (interior cells where (r+c)%2==0) are valid actions.
"""

import numpy as np
import torch

from bridgit.game import Bridgit, GameState, Player


class GameWrapper:
    """Bridges the Bridgit game API and the neural network / MCTS."""

    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.g = 2 * board_size + 1
        self.action_size = self.g * self.g  # flat index into g×g grid

    def get_state_tensor(self, game: Bridgit) -> torch.Tensor:
        """3-channel (g, g) tensor from current player's perspective."""
        return game.state.to_tensor(game.current_player)

    def get_valid_moves_mask(self, game: Bridgit) -> np.ndarray:
        """Binary mask of shape (action_size,) — 1.0 at legal crossings."""
        mask = game.state.to_mask().numpy().flatten()
        return mask

    def action_to_move(self, action: int) -> tuple[int, int]:
        """Convert flat action index to (row, col)."""
        return action // self.g, action % self.g

    def move_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) to flat action index."""
        return row * self.g + col

    def get_next_state(self, game: Bridgit, action: int) -> Bridgit:
        """Apply action and return new game state."""
        new_game = game.copy()
        row, col = self.action_to_move(action)
        new_game.make_move(row, col)
        return new_game

    def get_game_result(self, game: Bridgit, player: Player) -> float | None:
        """Return +1 if player won, -1 if lost, None if game ongoing."""
        if not game.game_over:
            return None
        return 1.0 if game.winner == player else -1.0

    def new_game(self) -> Bridgit:
        """Create a fresh game."""
        return Bridgit(self.board_size)
