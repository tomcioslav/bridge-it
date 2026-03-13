"""Tests for Bridgit game logic."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bridgit import Bridgit, Player
from bridgit.schema import Move
from bridgit.config import BoardConfig
from test.conftest import load_json


# Load test cases
test_cases = load_json("test_check_winner.json", test_dir="test_game")


@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[tc["description"] for tc in test_cases]
)
def test_check_winner(test_case):
    """Test winner detection for various game states.

    Each test case contains:
    - description: Human-readable test description
    - board_size: Size parameter n for (2n+1)×(2n+1) board
    - moves: List of moves to make [{row, col, player}, ...]
    - expected_winner: Expected winner ("HORIZONTAL", "VERTICAL", or null)
    """
    # Setup
    board_size = test_case["board_size"]
    moves = test_case["moves"]
    expected_winner = test_case["expected_winner"]

    # Create game
    game = Bridgit(BoardConfig(size=board_size))

    # Apply moves
    for move_data in moves:
        move = Move(row=move_data["row"], col=move_data["col"])
        player_name = move_data["player"]

        # Set current player to match the move
        game.current_player = Player[player_name]

        # Make the move
        success = game.make_move(move)
        assert success, f"Move ({move.row}, {move.col}) for {player_name} should be valid"

    # Check winner
    if expected_winner is None:
        assert game.winner is None, f"Expected no winner, but got {game.winner}"
        assert not game.game_over, "Game should not be over"
    else:
        expected_player = Player[expected_winner]
        assert game.winner == expected_player, \
            f"Expected {expected_player.name} to win, but got {game.winner.name if game.winner else 'no winner'}"
        assert game.game_over, "Game should be over when there's a winner"


def test_winner_detection_respects_turn_structure():
    """Test that winner detection works correctly with the 1-2-2 turn structure."""
    game = Bridgit(BoardConfig(size=3))

    # First player (HORIZONTAL) gets 1 move
    assert game.current_player == Player.HORIZONTAL
    assert game.moves_left_in_turn == 1

    # Make a move that doesn't win
    game.make_move(Move(row=2, col=2))

    # Should switch to VERTICAL with 2 moves
    assert game.current_player == Player.VERTICAL
    assert game.moves_left_in_turn == 2
    assert game.winner is None


def test_invalid_moves_dont_trigger_winner_check():
    """Test that invalid moves don't affect game state or winner detection."""
    game = Bridgit(BoardConfig(size=3))

    # Try to make an invalid move (not a crossing)
    success = game.make_move(Move(row=1, col=2))
    assert not success, "Move on non-crossing should fail"
    assert game.winner is None
    assert not game.game_over
    assert game.move_count == 0

    # Valid move should work
    success = game.make_move(Move(row=2, col=2))
    assert success
    assert game.move_count == 1


def test_game_ends_immediately_on_winning_move():
    """Test that the game ends as soon as a winning path is created."""
    game = Bridgit(BoardConfig(size=1))

    # On a 3x3 board (n=1), first move creates a winning path for HORIZONTAL
    game.make_move(Move(row=1, col=1))

    assert game.game_over, "Game should end immediately after winning move"
    assert game.winner == Player.HORIZONTAL

    # Subsequent moves should be rejected
    game.current_player = Player.VERTICAL  # Try to force another move
    success = game.make_move(Move(row=1, col=2))
    assert not success, "No moves should be allowed after game is over"
