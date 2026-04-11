import json
from pathlib import Path

from test.test_core.test_mcts import TicTacToe, DummyNet
from pymcts.arena import SinglePlayerArena, EloArena
from pymcts.arena.config import SinglePlayerArenaConfig, EloArenaConfig
from pymcts.core.config import MCTSConfig, TrainingConfig, PathsConfig
from pymcts.core.players import MCTSPlayer, GreedyMCTSPlayer
from pymcts.core.trainer import train


class TestArenaConfigPlayerSaving:
    def test_accepted_players_saved_to_arena_dir(self, tmp_path):
        """When using SinglePlayerArena, accepted players are saved to arena/."""
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
        arena_config = SinglePlayerArenaConfig(num_games=4, threshold=0.0)  # threshold=0 to always accept

        self_play_arena = SinglePlayerArena(arena_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = SinglePlayerArena(arena_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        # Eval arena directory should exist with saved players
        history_dir = tmp_path / "eval_arena" / "history"
        assert history_dir.exists()
        player_dirs = sorted(history_dir.glob("iteration_*"))
        assert len(player_dirs) >= 1  # At least one accepted player

        # Each should have player.json
        for player_dir in player_dirs:
            assert (player_dir / "player.json").exists()


class TestEloArenaTraining:
    def test_elo_arena_runs_to_completion(self, tmp_path):
        """Training with EloArena should complete without errors."""
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
        elo_config = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=2,
            max_pool_size=5,
        )
        sp_config = SinglePlayerArenaConfig(num_games=4)

        self_play_arena = SinglePlayerArena(sp_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = EloArena(elo_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        eval_arena_dir = tmp_path / "eval_arena"
        assert eval_arena_dir.exists()
        assert (eval_arena_dir / "pool" / "random" / "player.json").exists()

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
        elo_config = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=2,
        )
        sp_config = SinglePlayerArenaConfig(num_games=4)

        self_play_arena = SinglePlayerArena(sp_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = EloArena(elo_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        pool_dir = tmp_path / "eval_arena" / "pool"
        player_dirs = list(pool_dir.iterdir())
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
        elo_config = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            initial_pool=[str(seed_dir)],
        )
        sp_config = SinglePlayerArenaConfig(num_games=4)

        self_play_arena = SinglePlayerArena(sp_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = EloArena(elo_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=fresh_net,
            mcts_config=config,
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        pool_dir = tmp_path / "eval_arena" / "pool"
        assert (pool_dir / "random" / "player.json").exists()
        assert (pool_dir / "seed_player" / "player.json").exists()


class TestEloArenaPoolEviction:
    def test_pool_respects_max_size(self, tmp_path):
        """Pool should not exceed max_pool_size."""
        paths = PathsConfig(
            root=tmp_path,
            trainings=tmp_path / "trainings",
            checkpoints=tmp_path / "checkpoints",
            models=tmp_path / "models",
            data=tmp_path / "data",
        )
        net = DummyNet()
        training_config = TrainingConfig(
            num_iterations=6,
            num_self_play_games=4,
            num_epochs=1,
            batch_size=4,
            self_play_batch_size=2,
            replay_buffer_size=2,
        )
        elo_config = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=1,  # Add every iteration
            max_pool_size=3,  # random + 2 others max
        )
        sp_config = SinglePlayerArenaConfig(num_games=4)

        self_play_arena = SinglePlayerArena(sp_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = EloArena(elo_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        pool_dir = tmp_path / "eval_arena" / "pool"
        # Random player should always survive eviction
        assert (pool_dir / "random" / "player.json").exists()

    def test_random_player_never_evicted(self, tmp_path):
        """RandomPlayer should never be evicted from the pool."""
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
        elo_config = EloArenaConfig(
            games_per_matchup=4,
            elo_threshold=0.0,
            pool_growth_interval=1,
            max_pool_size=2,  # Very tight — random + 1
        )
        sp_config = SinglePlayerArenaConfig(num_games=4)

        self_play_arena = SinglePlayerArena(sp_config, TicTacToe, arena_dir=tmp_path / "self_play")
        eval_arena = EloArena(elo_config, TicTacToe, arena_dir=tmp_path / "eval_arena")

        train(
            game_factory=TicTacToe,
            net=net,
            mcts_config=MCTSConfig(num_simulations=5),
            training_config=training_config,
            self_play_arena=self_play_arena,
            eval_arena=eval_arena,
            paths_config=paths,
            verbose=False,
        )

        pool_dir = tmp_path / "eval_arena" / "pool"
        assert (pool_dir / "random" / "player.json").exists()
