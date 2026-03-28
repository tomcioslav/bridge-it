from pymcts.elo.config import MatchResult, EloRating, TournamentConfig, TournamentResult


class TestMatchResult:
    def test_create(self):
        m = MatchResult(player_a="alice", player_b="bob", wins_a=7, wins_b=3, draws=0)
        assert m.player_a == "alice"
        assert m.wins_a == 7
        assert m.total_games == 10

    def test_json_roundtrip(self):
        m = MatchResult(player_a="alice", player_b="bob", wins_a=7, wins_b=3, draws=0)
        data = m.model_dump_json()
        m2 = MatchResult.model_validate_json(data)
        assert m == m2


class TestEloRating:
    def test_create(self):
        r = EloRating(name="alice", rating=1500.0, games_played=10)
        assert r.name == "alice"
        assert r.rating == 1500.0


class TestTournamentConfig:
    def test_defaults(self):
        c = TournamentConfig()
        assert c.games_per_matchup == 40
        assert c.swap_players is True
        assert c.num_rounds is None
        assert c.convergence_threshold == 10.0
        assert c.batch_size == 8


class TestTournamentResult:
    def test_json_roundtrip(self):
        result = TournamentResult(
            ratings=[EloRating(name="random", rating=1000.0, games_played=10)],
            match_results=[
                MatchResult(player_a="random", player_b="alice", wins_a=3, wins_b=7, draws=0)
            ],
            anchor_player="random",
            anchor_rating=1000.0,
            timestamp="2026-03-28T12:00:00",
            metadata={"game": "bridgit"},
        )
        data = result.model_dump_json()
        result2 = TournamentResult.model_validate_json(data)
        assert result == result2
        assert result2.ratings[0].rating == 1000.0
