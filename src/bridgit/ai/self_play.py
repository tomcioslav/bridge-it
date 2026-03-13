"""Self-play game generation for training data."""

import numpy as np
import torch

from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import NetWrapper
from bridgit.game import Bridgit
from bridgit.schema import Move
from bridgit.config import BoardConfig, MCTSConfig, TrainingConfig


# Training example: (state_tensor, target_policy, target_value)
Example = tuple[torch.Tensor, np.ndarray, float]


def play_game(
    net_wrapper: NetWrapper,
    board: BoardConfig,
    mcts_config: MCTSConfig,
) -> list[Example]:
    """Play a single self-play game and return training examples.

    Each example is stored from the current player's perspective using
    the canonical board representation.
    """
    mcts = MCTS(net_wrapper, mcts_config, board)
    game = Bridgit(board)

    # Collect (state_tensor, policy, current_player) during the game
    history: list[tuple[torch.Tensor, np.ndarray, int]] = []
    move_count = 0

    while not game.game_over:
        # Temperature: exploratory early, greedy later
        temp = 1.0 if move_count < mcts_config.temp_threshold else 0.0

        # Get MCTS policy — shape (g, g)
        pi = mcts.get_action_probs(game, temperature=temp)

        # Store canonical state and policy
        state_tensor = game.to_tensor()
        player_value = game.current_player.value  # -1 or 1
        history.append((state_tensor, pi, player_value))

        # Sample move from policy
        flat_pi = pi.flatten()
        flat_idx = np.random.choice(len(flat_pi), p=flat_pi)
        row, col = np.unravel_index(flat_idx, pi.shape)
        game.make_move(Move(row=int(row), col=int(col)))
        move_count += 1

    # Assign values: +1 for winner, -1 for loser
    winner_value = game.winner.value  # -1 (HORIZONTAL) or 1 (VERTICAL)

    examples: list[Example] = []
    for state_tensor, pi, player_value in history:
        # +1 if this player won, -1 if lost
        value = 1.0 if player_value == winner_value else -1.0
        examples.append((state_tensor, pi, value))

    return examples


def generate_self_play_data(
    net_wrapper: NetWrapper,
    board: BoardConfig,
    mcts_config: MCTSConfig,
    training: TrainingConfig,
) -> list[Example]:
    """Generate training data from multiple self-play games."""
    all_examples: list[Example] = []

    for i in range(training.num_self_play_games):
        examples = play_game(net_wrapper, board, mcts_config)
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play: {i + 1}/{training.num_self_play_games} games completed")

    print(f"  Generated {len(all_examples)} training examples")
    return all_examples
