"""Tests for GameState: canonical, to_tensor, to_mask."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bridgit import GameState, Player


class TestCanonical:
    """Tests for GameState.canonical()."""

    def test_horizontal_returns_copy(self):
        """Canonical for HORIZONTAL returns an identical copy."""
        state = GameState.empty(3)
        canon = state.canonical(Player.HORIZONTAL)
        np.testing.assert_array_equal(canon.board, state.board)
        # Must be a copy, not the same object
        assert canon.board is not state.board

    def test_vertical_transposes_and_negates(self):
        """Canonical for VERTICAL transposes the board and negates values."""
        state = GameState.empty(3)
        canon = state.canonical(Player.VERTICAL)
        np.testing.assert_array_equal(canon.board, -state.board.T)

    def test_vertical_swaps_boundaries(self):
        """After canonical(VERTICAL), top/bottom become left/right."""
        state = GameState.empty(3)
        g = 2 * 3 + 1
        canon = state.canonical(Player.VERTICAL)
        # Original: VERTICAL (1) owns row 0 and row 6 at odd cols
        # After canonical: those become col 0 and col 6 at odd rows, negated to -1
        for r in range(1, g, 2):
            assert canon.board[r, 0] == Player.HORIZONTAL.value
            assert canon.board[r, g - 1] == Player.HORIZONTAL.value

    def test_double_canonical_is_identity(self):
        """Applying canonical(VERTICAL) twice returns the original board."""
        state = GameState.empty(3)
        state = state.make_move(2, 2, Player.HORIZONTAL)
        state = state.make_move(1, 1, Player.VERTICAL)
        double = state.canonical(Player.VERTICAL).canonical(Player.VERTICAL)
        np.testing.assert_array_equal(double.board, state.board)

    def test_canonical_with_moves(self):
        """Canonical correctly transforms a board with bridges placed."""
        state = GameState.empty(3)
        state = state.make_move(2, 2, Player.HORIZONTAL)
        canon = state.canonical(Player.VERTICAL)
        # HORIZONTAL value at (2,2) becomes -HORIZONTAL = VERTICAL at (2,2)
        assert canon.board[2, 2] == Player.VERTICAL.value
        # Original endpoints at (1,2) and (3,2) become (2,1) and (2,3) negated
        assert canon.board[2, 1] == Player.VERTICAL.value
        assert canon.board[2, 3] == Player.VERTICAL.value


class TestToTensor:
    """Tests for GameState.to_tensor() on canonical boards."""

    def test_shape(self):
        """Tensor has shape (3, g, g)."""
        state = GameState.empty(3)
        tensor = state.to_tensor()
        assert tensor.shape == (3, 7, 7)

    def test_dtype(self):
        """Tensor is float32."""
        state = GameState.empty(3)
        tensor = state.to_tensor()
        assert tensor.dtype == torch.float32

    def test_empty_board_channels(self):
        """On empty board, channel 0 has HORIZONTAL boundaries, channel 1 has VERTICAL."""
        state = GameState.empty(3)
        tensor = state.to_tensor()
        g = 7
        mine = tensor[0]     # HORIZONTAL (-1) cells
        theirs = tensor[1]   # VERTICAL (1) cells
        # HORIZONTAL owns left/right boundaries at odd rows
        for r in range(1, g, 2):
            assert mine[r, 0] == 1.0
            assert mine[r, g - 1] == 1.0
        # VERTICAL owns top/bottom boundaries at odd cols
        for c in range(1, g, 2):
            assert theirs[0, c] == 1.0
            assert theirs[g - 1, c] == 1.0

    def test_playable_channel_empty_board(self):
        """Channel 2 marks all interior crossings as playable on empty board."""
        state = GameState.empty(3)
        tensor = state.to_tensor()
        g = 7
        playable = tensor[2]
        for r in range(g):
            for c in range(g):
                is_crossing = (r + c) % 2 == 0 and 0 < r < g - 1 and 0 < c < g - 1
                assert playable[r, c] == (1.0 if is_crossing else 0.0), \
                    f"Playable mismatch at ({r}, {c})"

    def test_move_updates_channels(self):
        """After a HORIZONTAL move, crossing shows in channel 0, removed from channel 2."""
        state = GameState.empty(3)
        state = state.make_move(2, 2, Player.HORIZONTAL)
        tensor = state.to_tensor()
        # (2,2) should be in mine channel
        assert tensor[0][2, 2] == 1.0
        # (2,2) should not be playable
        assert tensor[2][2, 2] == 0.0
        # (2,2) should not be in opponent channel
        assert tensor[1][2, 2] == 0.0

    def test_canonical_perspective_for_vertical(self):
        """to_tensor on a canonical(VERTICAL) board sees VERTICAL's pieces as mine."""
        state = GameState.empty(3)
        state = state.make_move(1, 1, Player.VERTICAL)
        canon = state.canonical(Player.VERTICAL)
        tensor = canon.to_tensor()
        # After canonical, VERTICAL's pieces become HORIZONTAL value (-1) = channel 0
        # Original (1,1) VERTICAL bridge → canonical transposes to (1,1), negates to -1
        assert tensor[0][1, 1] == 1.0


class TestToMask:
    """Tests for GameState.to_mask() on canonical boards."""

    def test_shape(self):
        """Mask has shape (g, g)."""
        state = GameState.empty(3)
        mask = state.to_mask()
        assert mask.shape == (7, 7)

    def test_dtype(self):
        """Mask is float32."""
        state = GameState.empty(3)
        mask = state.to_mask()
        assert mask.dtype == torch.float32

    def test_empty_board_crossing_count(self):
        """Empty board has correct number of playable crossings."""
        state = GameState.empty(3)
        mask = state.to_mask()
        g = 7
        expected = sum(
            1 for r in range(1, g - 1) for c in range(1, g - 1)
            if (r + c) % 2 == 0
        )
        assert mask.sum().item() == expected

    def test_move_reduces_mask(self):
        """Making a move removes that crossing from the mask."""
        state = GameState.empty(3)
        mask_before = state.to_mask()
        state = state.make_move(2, 2, Player.HORIZONTAL)
        mask_after = state.to_mask()
        assert mask_before[2, 2] == 1.0
        assert mask_after[2, 2] == 0.0
        assert mask_after.sum() == mask_before.sum() - 1

    def test_mask_matches_tensor_channel_2(self):
        """Mask should equal channel 2 of to_tensor."""
        state = GameState.empty(3)
        state = state.make_move(2, 2, Player.HORIZONTAL)
        state = state.make_move(1, 1, Player.VERTICAL)
        tensor = state.to_tensor()
        mask = state.to_mask()
        torch.testing.assert_close(mask, tensor[2])

    def test_canonical_mask_for_vertical(self):
        """Mask on canonical(VERTICAL) board is transposed relative to original."""
        state = GameState.empty(3)
        state = state.make_move(2, 2, Player.HORIZONTAL)
        mask_h = state.to_mask()
        canon = state.canonical(Player.VERTICAL)
        mask_v = canon.to_mask()
        # After canonical(VERTICAL): transpose + negate.
        # Crossing (2,2) was taken → still 0 at (2,2) after transpose (symmetric).
        assert mask_v[2, 2] == 0.0
        # But endpoints stamped at (1,2) and (3,2) by H become (2,1) and (2,3) after T.
        # These are non-crossing cells so they're 0 in both masks anyway.
        # Total playable count should be the same.
        assert mask_v.sum() == mask_h.sum()
