from pymcts.arena.config import EloArenaConfig


class TestEloArenaConfig:
    def test_defaults(self):
        config = EloArenaConfig()
        assert config.games_per_matchup == 40
        assert config.elo_threshold == 20.0
        assert config.pool_growth_interval == 5
        assert config.max_pool_size is None
        assert config.swap_players is True
        assert config.initial_pool is None

    def test_custom_values(self):
        config = EloArenaConfig(
            games_per_matchup=20,
            elo_threshold=30.0,
            pool_growth_interval=3,
            max_pool_size=10,
            initial_pool=["/path/to/player1", "/path/to/player2"],
        )
        assert config.games_per_matchup == 20
        assert config.elo_threshold == 30.0
        assert config.pool_growth_interval == 3
        assert config.max_pool_size == 10
        assert len(config.initial_pool) == 2

    def test_serialization_roundtrip(self):
        config = EloArenaConfig(max_pool_size=15, initial_pool=["/some/path"])
        data = config.model_dump()
        restored = EloArenaConfig(**data)
        assert restored == config
