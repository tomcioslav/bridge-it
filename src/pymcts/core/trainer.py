"""Main training pipeline: self-play -> train -> evaluate loop."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig, PathsConfig, TrainingConfig
from pymcts.core.data import examples_from_records
from pymcts.core.players import GreedyMCTSPlayer

logger = logging.getLogger("pymcts.core.trainer")


def _create_run_dir(paths: PathsConfig) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = paths.trainings / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_config(
    run_dir: Path,
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
) -> None:
    net_class = type(net)
    config = {
        "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
        "mcts_config": mcts_config.model_dump(),
        "training_config": training_config.model_dump(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def train(
    game_factory: Callable[[], BaseGame],
    net: BaseNeuralNet,
    mcts_config: MCTSConfig,
    training_config: TrainingConfig,
    self_play_arena: Arena,
    eval_arena: Arena,
    paths_config: PathsConfig | None = None,
    verbose: bool = True,
):
    """Run the full AlphaZero training pipeline.

    Each iteration: self-play -> train -> evaluate (accept/reject).
    """
    paths = paths_config or PathsConfig()
    run_dir = _create_run_dir(paths)

    _save_run_config(run_dir, net, mcts_config, training_config)

    if verbose:
        print(f"Training run directory: {run_dir}")
        print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}\n")

    replay_buffer: deque[list] = deque(maxlen=training_config.replay_buffer_size)

    for iteration in range(1, training_config.num_iterations + 1):
        if verbose:
            print(f"{'=' * 60}\nIteration {iteration}/{training_config.num_iterations}\n{'=' * 60}")

        iter_dir = run_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        pre_checkpoint = str(iter_dir / "pre_training.pt")
        net.save_checkpoint(pre_checkpoint)

        # 1. Self-play
        if verbose:
            print("\n[1/3] Self-play...")
        player = GreedyMCTSPlayer(net, mcts_config, name="self_play")
        records = self_play_arena.play_games(player, num_games=training_config.num_self_play_games)
        (iter_dir / "self_play_games.json").write_text(records.model_dump_json(indent=2))

        new_examples = examples_from_records(records, lambda cfg: game_factory())
        replay_buffer.append(new_examples)
        all_examples = [ex for batch in replay_buffer for ex in batch]

        if verbose:
            print(f"  {len(records)} games, {len(new_examples)} examples")
            print(f"  Replay buffer: {len(replay_buffer)} iterations, {len(all_examples)} examples")

        # 2. Train
        if verbose:
            print("\n[2/3] Training...")
        net.train_on_examples(
            all_examples,
            num_epochs=training_config.num_epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            verbose=verbose,
        )
        net.save_checkpoint(str(iter_dir / "post_training.pt"))

        # 3. Evaluate
        if verbose:
            print("\n[3/3] Arena evaluation...")
        candidate = GreedyMCTSPlayer(net, mcts_config, name="candidate")
        result = eval_arena.is_candidate_better(candidate)

        if not result.accepted:
            net.load_checkpoint(pre_checkpoint)

        # Save iteration data
        iteration_data = {
            "iteration": iteration,
            "training": {"num_examples": len(all_examples)},
            "evaluation": result.details,
            "accepted": result.accepted,
        }
        (iter_dir / "iteration_data.json").write_text(json.dumps(iteration_data, indent=2))

        if verbose:
            print()

    if verbose:
        print(f"Training complete!\nRun directory: {run_dir}")
