"""Self-play game generation for training data."""

import numpy as np
import torch

from bridgit.ai.config import Config
from bridgit.ai.game_wrapper import GameWrapper
from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import NeuralNetWrapper


# Training example: (state_tensor, target_policy, target_value)
Example = tuple[torch.Tensor, np.ndarray, float]


def play_game(neural_net: NeuralNetWrapper, config: Config) -> list[Example]:
    """Play a single self-play game and return training examples.

    Each example is stored from the current player's perspective using
    the canonical board representation.
    """
    game_wrapper = GameWrapper(config.board_size)
    mcts = MCTS(neural_net, config)
    game = game_wrapper.new_game()

    # Collect (state_tensor, policy, current_player) during the game
    history: list[tuple[torch.Tensor, np.ndarray, int]] = []
    move_count = 0

    while not game.game_over:
        # Temperature: exploratory early, greedy later
        temp = 1.0 if move_count < config.temp_threshold else 0.0

        # Get MCTS policy
        pi = mcts.get_action_probs(game, temperature=temp)

        # Store canonical state and policy
        state_tensor = game_wrapper.get_state_tensor(game)
        player_value = game.current_player.value  # -1 or 1
        history.append((state_tensor, pi, player_value))

        # Sample action from policy
        action = np.random.choice(len(pi), p=pi)
        game = game_wrapper.get_next_state(game, action)
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
    neural_net: NeuralNetWrapper, config: Config
) -> list[Example]:
    """Generate training data from multiple self-play games."""
    all_examples: list[Example] = []

    for i in range(config.num_self_play_games):
        examples = play_game(neural_net, config)
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play: {i + 1}/{config.num_self_play_games} games completed")

    print(f"  Generated {len(all_examples)} training examples")
    return all_examples
