from pymcts.core.config import TrainingConfig


class TestTrainingConfigElo:
    def test_elo_fields_exist(self):
        config = TrainingConfig()
        assert config.elo_tracking is False
        assert config.elo_reference_interval == 5
        assert config.elo_games_per_matchup == 40

    def test_elo_tracking_enabled(self):
        config = TrainingConfig(elo_tracking=True, elo_reference_interval=3)
        assert config.elo_tracking is True
        assert config.elo_reference_interval == 3
