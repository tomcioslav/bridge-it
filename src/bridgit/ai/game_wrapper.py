"""Adapter between Bridgit game and the AI pipeline."""

import numpy as np
import torch

from bridgit.game import Bridgit, Player


class GameWrapper:
    """Provides canonical board views and state tensors for the neural network."""

    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.action_size = board_size * board_size

    def get_canonical_board(self, game: Bridgit) -> np.ndarray:
        """Return board from current player's perspective.

        If it's VERTICAL's turn, negate and transpose so that
        the current player always appears as -1 connecting left→right.
        """
        board = game.grid.copy()
        if game.current_player == Player.VERTICAL:
            board = -board.T
        return board

    def get_state_tensor(self, game: Bridgit) -> torch.Tensor:
        """Encode game state as a 3-channel tensor for the neural network.

        Channel 0: current player's pieces (1.0 where present)
        Channel 1: opponent's pieces (1.0 where present)
        Channel 2: constant plane (1.0 if HORIZONTAL's turn, 0.0 if VERTICAL's)
        """
        canonical = self.get_canonical_board(game)
        n = self.board_size

        current_pieces = (canonical == -1).astype(np.float32)
        opponent_pieces = (canonical == 1).astype(np.float32)
        turn_plane = np.full((n, n), float(game.current_player == Player.HORIZONTAL), dtype=np.float32)

        state = np.stack([current_pieces, opponent_pieces, turn_plane], axis=0)
        return torch.from_numpy(state)

    def get_valid_moves_mask(self, game: Bridgit) -> np.ndarray:
        """Return binary mask of shape (action_size,) for legal moves.

        If current player is VERTICAL, the canonical board is transposed,
        so we must also transpose the move coordinates.
        """
        mask = np.zeros(self.action_size, dtype=np.float32)
        for r, c in game.get_available_moves():
            if game.current_player == Player.VERTICAL:
                # Canonical board is transposed: (r, c) → (c, r)
                action = c * self.board_size + r
            else:
                action = r * self.board_size + c
            mask[action] = 1.0
        return mask

    def action_to_move(self, action: int, game: Bridgit) -> tuple[int, int]:
        """Convert flat action index to (row, col) in original board coordinates."""
        row = action // self.board_size
        col = action % self.board_size
        if game.current_player == Player.VERTICAL:
            # Undo the transpose: canonical (row, col) → original (col, row)
            row, col = col, row
        return row, col

    def get_next_state(self, game: Bridgit, action: int) -> Bridgit:
        """Apply action and return new game state."""
        new_game = game.copy()
        row, col = self.action_to_move(action, game)
        new_game.make_move(row, col)
        return new_game

    def get_game_result(self, game: Bridgit, player: Player) -> float | None:
        """Return +1 if player won, -1 if lost, None if game ongoing."""
        if not game.game_over:
            return None
        if game.winner == player:
            return 1.0
        return -1.0

    def new_game(self) -> Bridgit:
        """Create a fresh game."""
        return Bridgit(self.board_size)
