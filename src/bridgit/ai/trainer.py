"""Main training pipeline: self-play → train → evaluate loop."""

import argparse
import os
from collections import deque

import numpy as np

from bridgit.ai.mcts import MCTS
from bridgit.ai.neural_net import BridgitNet, NetWrapper
from bridgit.ai.self_play import generate_self_play_data, Example
from bridgit.players.arena import Arena
from bridgit.schema import Move
from bridgit.config import Config


def make_mcts_player(net_wrapper: NetWrapper, config: Config):
    """Create a player function that uses MCTS for move selection."""
    mcts = MCTS(net_wrapper, config.mcts, config.board)

    def player_fn(game):
        pi = mcts.get_action_probs(game, temperature=0)
        best = np.unravel_index(np.argmax(pi), pi.shape)
        return Move(row=int(best[0]), col=int(best[1]))

    return player_fn


def train(config: Config):
    """Run the full AlphaZero training pipeline."""
    print(f"Training Bridgit AI on {config.board.size}x{config.board.size} board")
    print(f"Config: {config}")
    print()

    model = BridgitNet(config.board, config.neural_net)
    net_wrapper = NetWrapper(model)
    print(f"Device: {net_wrapper.device}")
    print(f"Model parameters: {sum(p.numel() for p in net_wrapper.model.parameters()):,}")
    print()

    # Replay buffer: stores examples from last N iterations
    replay_buffer: deque[list[Example]] = deque(maxlen=config.training.replay_buffer_size)

    best_checkpoint = config.paths.checkpoints / "best.pt"

    for iteration in range(1, config.training.num_iterations + 1):
        print(f"{'=' * 60}")
        print(f"Iteration {iteration}/{config.training.num_iterations}")
        print(f"{'=' * 60}")

        # 1. Self-play
        print("\n[1/3] Self-play...")
        new_examples = generate_self_play_data(
            net_wrapper, config.board, config.mcts, config.training
        )
        replay_buffer.append(new_examples)

        # Flatten all examples from replay buffer
        all_examples = [ex for batch in replay_buffer for ex in batch]
        print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        print("\n[2/3] Training...")
        # Save current model before training (for comparison)
        temp_checkpoint = config.paths.checkpoints / "temp.pt"
        net_wrapper.save_checkpoint(str(temp_checkpoint))

        net_wrapper.train_on_examples(all_examples)

        # 3. Evaluate
        print("\n[3/3] Arena evaluation...")
        new_player = make_mcts_player(net_wrapper, config)

        # Load previous best for comparison
        prev_model = BridgitNet(config.board, config.neural_net)
        prev_net = NetWrapper(prev_model)
        if best_checkpoint.exists():
            prev_net.load_checkpoint(str(best_checkpoint))
        else:
            prev_net.load_checkpoint(str(temp_checkpoint))

        prev_player = make_mcts_player(prev_net, config)

        arena = Arena(new_player, prev_player, config.board)
        new_wins, prev_wins = arena.play_games(config.arena.num_games)
        total = new_wins + prev_wins
        win_rate = new_wins / total if total > 0 else 0

        print(f"  New model: {new_wins} wins | Previous: {prev_wins} wins | "
              f"Win rate: {win_rate:.1%}")

        if win_rate >= config.arena.threshold:
            print("  -> ACCEPTED: new model is better, saving checkpoint")
            net_wrapper.save_checkpoint(str(best_checkpoint))
        else:
            print("  -> REJECTED: keeping previous model")
            net_wrapper.load_checkpoint(str(temp_checkpoint))

        # Clean up temp checkpoint
        if temp_checkpoint.exists():
            temp_checkpoint.unlink()

        # Save iteration checkpoint
        iter_checkpoint = config.paths.checkpoints / f"iter_{iteration:04d}.pt"
        net_wrapper.save_checkpoint(str(iter_checkpoint))
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
    args = parser.parse_args()

    config = Config()
    if args.iterations is not None:
        config.training.num_iterations = args.iterations
    if args.games is not None:
        config.training.num_self_play_games = args.games
    if args.sims is not None:
        config.mcts.num_simulations = args.sims
    if args.arena_games is not None:
        config.arena.num_games = args.arena_games

    train(config)


if __name__ == "__main__":
    main()
