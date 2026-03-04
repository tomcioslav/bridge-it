"""Main training pipeline: self-play → train → evaluate loop."""

import argparse
import os
from collections import deque

import numpy as np

from bridgit.ai.config import Config
from bridgit.ai.game_wrapper import GameWrapper
from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import NeuralNetWrapper
from bridgit.ai.self_play import generate_self_play_data, Example
from bridgit.ai.arena import Arena


def make_mcts_player(neural_net: NeuralNetWrapper, config: Config):
    """Create a player function that uses MCTS for move selection."""
    game_wrapper = GameWrapper(config.board_size)
    mcts = MCTS(neural_net, config)

    def player_fn(game):
        pi = mcts.get_action_probs(game, temperature=0)
        return int(np.argmax(pi))

    return player_fn


def train(config: Config):
    """Run the full AlphaZero training pipeline."""
    print(f"Training Bridgit AI on {config.board_size}x{config.board_size} board")
    print(f"Config: {config}")
    print()

    neural_net = NeuralNetWrapper(config)
    print(f"Device: {neural_net.device}")
    print(f"Model parameters: {sum(p.numel() for p in neural_net.model.parameters()):,}")
    print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(maxlen=config.replay_buffer_size)

    best_checkpoint = os.path.join(config.checkpoint_dir, "best.pt")

    for iteration in range(1, config.num_iterations + 1):
        print(f"{'=' * 60}")
        print(f"Iteration {iteration}/{config.num_iterations}")
        print(f"{'=' * 60}")

        # 1. Self-play
        print("\n[1/3] Self-play...")
        new_examples = generate_self_play_data(neural_net, config)
        replay_buffer.append(new_examples)

        # Flatten all examples from replay buffer
        all_examples = [ex for batch in replay_buffer for ex in batch]
        print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        print("\n[2/3] Training...")
        # Save current model before training (for comparison)
        temp_checkpoint = os.path.join(config.checkpoint_dir, "temp.pt")
        neural_net.save_checkpoint(temp_checkpoint)

        neural_net.train_on_examples(all_examples)

        # 3. Evaluate
        print("\n[3/3] Arena evaluation...")
        new_player = make_mcts_player(neural_net, config)

        # Load previous best for comparison
        prev_net = NeuralNetWrapper(config)
        if os.path.exists(best_checkpoint):
            prev_net.load_checkpoint(best_checkpoint)
        else:
            prev_net.load_checkpoint(temp_checkpoint)

        prev_player = make_mcts_player(prev_net, config)

        arena = Arena(new_player, prev_player, config)
        new_wins, prev_wins = arena.play_games(config.num_arena_games)
        total = new_wins + prev_wins
        win_rate = new_wins / total if total > 0 else 0

        print(f"  New model: {new_wins} wins | Previous: {prev_wins} wins | "
              f"Win rate: {win_rate:.1%}")

        if win_rate >= config.arena_threshold:
            print("  -> ACCEPTED: new model is better, saving checkpoint")
            neural_net.save_checkpoint(best_checkpoint)
        else:
            print("  -> REJECTED: keeping previous model")
            neural_net.load_checkpoint(temp_checkpoint)

        # Clean up temp checkpoint
        if os.path.exists(temp_checkpoint):
            os.remove(temp_checkpoint)

        # Save iteration checkpoint
        iter_checkpoint = os.path.join(config.checkpoint_dir, f"iter_{iteration:04d}.pt")
        neural_net.save_checkpoint(iter_checkpoint)
        print(f"  Saved iteration checkpoint: {iter_checkpoint}")
        print()

    print("Training complete!")
    print(f"Best model: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train Bridgit AI")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=None,
                        help="Self-play games per iteration")
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS simulations per move")
    parser.add_argument("--arena-games", type=int, default=None,
                        help="Arena games for evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints")
    args = parser.parse_args()

    config = Config()
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.games is not None:
        config.num_self_play_games = args.games
    if args.sims is not None:
        config.num_mcts_sims = args.sims
    if args.arena_games is not None:
        config.num_arena_games = args.arena_games
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    train(config)


if __name__ == "__main__":
    main()
