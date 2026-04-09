import json
from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.core.config import ArenaConfig, EloArenaConfig, MCTSConfig, TrainingConfig, PathsConfig
from pymcts.core.players import MCTSPlayer, GreedyMCTSPlayer
from pymcts.core.trainer import train


class TestArenaConfigPlayerSaving:
    def test_accepted_players_saved_to_arena_dir(self, tmp_path):
        """When using ArenaConfig, accepted players are saved to arena/."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=2,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        arena_config = ArenaConfig(num_games=4, threshold=0.0)  # threshold=0 to always accept

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=arena_config,
            paths_config=paths,
            verbose=False,
        )

        # Find the run directory
        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Arena directory should exist with saved players
        arena_dir = run_dir / "arena"
        assert arena_dir.exists()
        player_dirs = sorted(arena_dir.glob("iteration_*"))
        assert len(player_dirs) >= 1  # At least one accepted player

        # Each should have player.json
        for player_dir in player_dirs:
            assert (player_dir / "player.json").exists()


class TestEloArenaTraining:
    def test_elo_arena_runs_to_completion(self, tmp_path):
        """Training with EloArenaConfig should complete without errors."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=3,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=2,
            max_pool_size=5,
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        arena_dir = run_dir / "arena"
        assert arena_dir.exists()
        assert (arena_dir / "random" / "player.json").exists()

    def test_elo_arena_pool_grows(self, tmp_path):
        """Pool should grow at pool_growth_interval."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=4,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=2,
        )

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        arena_dir = run_dir / "arena"

        player_dirs = list(arena_dir.iterdir())
        assert len(player_dirs) >= 2  # random + at least 1 grown player

    def test_elo_arena_with_initial_pool(self, tmp_path):
        """Training with initial_pool should seed the pool from saved players."""
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        seed_player = MCTSPlayer(net, config, name="seed_player")
        seed_player.elo = 1050.0
        seed_dir = tmp_path / "seed_player"
        seed_player.save(seed_dir)

        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        fresh_net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=2,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_arena = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            initial_pool=[str(seed_dir)],
        )

        train(
            game_factory=TicTacToe,
            net=fresh_net,
            mcts_config=config,
            training_config=training_config,
            arena=elo_arena,
            paths_config=paths,
            verbose=False,
        )

        run_dirs = list((tmp_path / "trainings").glob("run_*"))
        run_dir = run_dirs[0]
        arena_dir = run_dir / "arena"

        assert (arena_dir / "random" / "player.json").exists()
        assert (arena_dir / "seed_player" / "player.json").exists()
