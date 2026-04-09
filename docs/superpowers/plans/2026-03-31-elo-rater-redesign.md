# Elo Rater Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the elo module around an `EloRater` class with benchmark player brigades, add save/load to all players, and add `elo_rating` field to `BasePlayer`.

**Architecture:** Three areas of change: (1) `BasePlayer` gets `elo_rating` field + abstract save/load, with `RandomPlayer` and `MCTSPlayer` implementing them; (2) `PathsConfig` gets a `players` path; (3) New `EloRater` class replaces the tournament system, keeping `compute_elo_ratings` and data models intact.

**Tech Stack:** Python, Pydantic, pytest, torch (for MCTSPlayer save/load)

---

### Task 1: Add `elo_rating` field to `BasePlayer` and abstract save/load methods

**Files:**
- Modify: `src/pymcts/core/players.py:24-37`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing tests for new BasePlayer features**

In `test/test_core/test_players.py`, add:

```python
class TestBasePlayerEloRating:
    def test_default_elo_rating_is_none(self):
        player = RandomPlayer()
        assert player.elo_rating is None

    def test_elo_rating_can_be_set(self):
        player = RandomPlayer(elo_rating=1500.0)
        assert player.elo_rating == 1500.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_players.py::TestBasePlayerEloRating -v`
Expected: FAIL — `elo_rating` not accepted by `__init__`

- [ ] **Step 3: Implement changes to BasePlayer**

In `src/pymcts/core/players.py`, modify `BasePlayer`:

```python
class BasePlayer(ABC):
    def __init__(self, name: str | None = None, elo_rating: float | None = None):
        self.name = name or self.__class__.__name__
        self.elo_rating = elo_rating
        self._last_policy: torch.Tensor | None = None

    @abstractmethod
    def get_action(self, game: BaseGame) -> int: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BasePlayer": ...

    @property
    def last_policy(self) -> torch.Tensor | None:
        return self._last_policy

    def __repr__(self) -> str:
        return self.name
```

Note: Add `Path` to the existing `from pathlib import Path` import in `players.py` (MCTSPlayer.save already uses it so it should already be there).

- [ ] **Step 4: Update RandomPlayer.__init__ to pass elo_rating through**

```python
class RandomPlayer(BasePlayer):
    def __init__(self, name: str | None = None, elo_rating: float | None = None):
        super().__init__(name=name, elo_rating=elo_rating)
```

Add stub save/load so the class isn't abstract (will be implemented properly in Task 2):

```python
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "RandomPlayer":
        raise NotImplementedError
```

- [ ] **Step 5: Update MCTSPlayer.__init__ to pass elo_rating through**

```python
class MCTSPlayer(BasePlayer):
    def __init__(
        self,
        net: BaseNeuralNet,
        mcts_config: MCTSConfig,
        temperature: float = 1.0,
        temp_threshold: int = 0,
        name: str | None = None,
        elo_rating: float | None = None,
    ):
        super().__init__(name, elo_rating=elo_rating)
        self.mcts = MCTS(net, mcts_config)
        self.temperature = temperature
        self.temp_threshold = temp_threshold
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: ALL PASS (including existing tests)

- [ ] **Step 7: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: add elo_rating field and abstract save/load to BasePlayer"
```

---

### Task 2: Implement RandomPlayer save/load

**Files:**
- Modify: `src/pymcts/core/players.py:40-43`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing tests**

In `test/test_core/test_players.py`, add:

```python
class TestRandomPlayerSaveLoad:
    def test_save_creates_player_json(self, tmp_path):
        player = RandomPlayer(name="rng1")
        player.save(tmp_path / "rng1")
        assert (tmp_path / "rng1" / "player.json").exists()

    def test_load_restores_player(self, tmp_path):
        player = RandomPlayer(name="rng1")
        player.save(tmp_path / "rng1")
        loaded = RandomPlayer.load(tmp_path / "rng1")
        assert loaded.name == "rng1"
        assert isinstance(loaded, RandomPlayer)

    def test_save_load_preserves_elo_rating(self, tmp_path):
        player = RandomPlayer(name="rng1", elo_rating=1200.0)
        player.save(tmp_path / "rng1")
        loaded = RandomPlayer.load(tmp_path / "rng1")
        assert loaded.elo_rating == 1200.0

    def test_save_load_preserves_none_elo(self, tmp_path):
        player = RandomPlayer(name="rng1")
        player.save(tmp_path / "rng1")
        loaded = RandomPlayer.load(tmp_path / "rng1")
        assert loaded.elo_rating is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_players.py::TestRandomPlayerSaveLoad -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement RandomPlayer save/load**

In `src/pymcts/core/players.py`, replace the RandomPlayer stubs:

```python
class RandomPlayer(BasePlayer):
    def __init__(self, name: str | None = None, elo_rating: float | None = None):
        super().__init__(name=name, elo_rating=elo_rating)

    def get_action(self, game: BaseGame) -> int:
        self._last_policy = None
        return random.choice(game.valid_actions())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "type": "RandomPlayer",
            "name": self.name,
            "elo_rating": self.elo_rating,
        }
        (path / "player.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RandomPlayer":
        path = Path(path)
        config = json.loads((path / "player.json").read_text())
        return cls(name=config["name"], elo_rating=config.get("elo_rating"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: implement RandomPlayer save/load"
```

---

### Task 3: Update MCTSPlayer save/load to persist elo_rating

**Files:**
- Modify: `src/pymcts/core/players.py:75-109`
- Test: `test/test_core/test_players.py`

- [ ] **Step 1: Write failing tests**

In `test/test_core/test_players.py`, add:

```python
class TestMCTSPlayerSaveLoadElo:
    def test_save_load_preserves_elo_rating(self, tmp_path):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config, name="mcts1", elo_rating=1350.0)
        player.save(tmp_path / "mcts1")
        loaded = MCTSPlayer.load(tmp_path / "mcts1")
        assert loaded.elo_rating == 1350.0

    def test_save_load_preserves_none_elo(self, tmp_path):
        net = DummyNet()
        config = MCTSConfig(num_simulations=5)
        player = MCTSPlayer(net, config, name="mcts1")
        player.save(tmp_path / "mcts1")
        loaded = MCTSPlayer.load(tmp_path / "mcts1")
        assert loaded.elo_rating is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_core/test_players.py::TestMCTSPlayerSaveLoadElo -v`
Expected: FAIL — `elo_rating` not persisted/restored

- [ ] **Step 3: Update MCTSPlayer.save to include elo_rating**

In `src/pymcts/core/players.py`, in `MCTSPlayer.save()`, add `"elo_rating"` to the config dict:

```python
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        net = self.mcts.net
        net.save_checkpoint(str(path / "model.pt"))

        net_class = type(net)
        config = {
            "net_class": f"{net_class.__module__}.{net_class.__qualname__}",
            "mcts_config": self.mcts.mcts_config.model_dump(),
            "temperature": self.temperature,
            "temp_threshold": self.temp_threshold,
            "name": self.name,
            "elo_rating": self.elo_rating,
        }
        (path / "player.json").write_text(json.dumps(config, indent=2))
```

- [ ] **Step 4: Update MCTSPlayer.load to restore elo_rating**

In `src/pymcts/core/players.py`, in `MCTSPlayer.load()`:

```python
    @classmethod
    def load(cls, path: str | Path) -> "MCTSPlayer":
        path = Path(path)
        config = json.loads((path / "player.json").read_text())

        net_class = _import_class(config["net_class"])
        net = net_class.from_checkpoint(str(path / "model.pt"))
        mcts_config = MCTSConfig(**config["mcts_config"])

        return cls(
            net=net,
            mcts_config=mcts_config,
            temperature=config["temperature"],
            temp_threshold=config["temp_threshold"],
            name=config["name"],
            elo_rating=config.get("elo_rating"),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_core/test_players.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/core/players.py test/test_core/test_players.py
git commit -m "feat: persist elo_rating in MCTSPlayer save/load"
```

---

### Task 4: Add `players` path to PathsConfig

**Files:**
- Modify: `src/pymcts/core/config.py:12-18`

- [ ] **Step 1: Add `players` field to PathsConfig**

In `src/pymcts/core/config.py`:

```python
class PathsConfig(BaseModel):
    """File system paths for the project."""
    root: Path = _PROJECT_ROOT
    checkpoints: Path = _PROJECT_ROOT / "checkpoints"
    models: Path = _PROJECT_ROOT / "models"
    data: Path = _PROJECT_ROOT / "data"
    trainings: Path = _PROJECT_ROOT / "trainings"
    players: Path = _PROJECT_ROOT / "players"
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `python -m pytest test/test_core/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/pymcts/core/config.py
git commit -m "feat: add players path to PathsConfig"
```

---

### Task 5: Add EloRaterConfig and clean up elo config.py

**Files:**
- Modify: `src/pymcts/elo/config.py`
- Modify: `test/test_elo/test_config.py`

- [ ] **Step 1: Update test_config.py**

Replace `test/test_elo/test_config.py` with:

```python
from pymcts.elo.config import MatchResult, EloRating, EloRaterConfig, TournamentResult


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


class TestEloRaterConfig:
    def test_defaults(self):
        c = EloRaterConfig()
        assert c.games_per_matchup == 40
        assert c.swap_players is True
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_elo/test_config.py -v`
Expected: FAIL — `EloRaterConfig` not found

- [ ] **Step 3: Update elo config.py**

Replace `src/pymcts/elo/config.py` — remove `TournamentConfig`, add `EloRaterConfig`, keep `TournamentResult` (used by trainer.py):

```python
"""Pydantic models for the Elo rating system."""

from pydantic import BaseModel


class MatchResult(BaseModel):
    """Outcome of a matchup between two players (multiple games)."""
    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.wins_a + self.wins_b + self.draws


class EloRating(BaseModel):
    """Elo rating for a single player."""
    name: str
    rating: float
    games_played: int


class EloRaterConfig(BaseModel):
    """Configuration for the EloRater."""
    games_per_matchup: int = 40
    swap_players: bool = True
    batch_size: int = 8


class TournamentResult(BaseModel):
    """Complete result of a tournament: ratings + match history."""
    ratings: list[EloRating]
    match_results: list[MatchResult]
    anchor_player: str
    anchor_rating: float
    timestamp: str
    metadata: dict
```

- [ ] **Step 4: Update trainer.py import**

In `src/pymcts/core/trainer.py`, line 16, change:

```python
from pymcts.elo.config import EloRating, MatchResult, TournamentResult
```

No change needed — `TournamentResult` is still in config.py.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test/test_elo/test_config.py test/test_elo/test_rating.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/pymcts/elo/config.py test/test_elo/test_config.py
git commit -m "feat: replace TournamentConfig with EloRaterConfig"
```

---

### Task 6: Implement EloRater class

**Files:**
- Create: `src/pymcts/elo/elo_rater.py`
- Create: `test/test_elo/test_elo_rater.py`

- [ ] **Step 1: Write failing tests for EloRater**

Create `test/test_elo/test_elo_rater.py`:

```python
import pytest

from test.test_core.test_mcts import TicTacToe

from pymcts.core.players import RandomPlayer
from pymcts.elo.config import EloRaterConfig
from pymcts.elo.elo_rater import EloRater


class TestEloRaterInit:
    def test_benchmark_players_get_elo_ratings(self):
        players = [RandomPlayer(name=f"p{i}") for i in range(3)]
        config = EloRaterConfig(games_per_matchup=10, batch_size=1)
        rater = EloRater(players, game_factory=TicTacToe, config=config)
        for p in players:
            assert p.elo_rating is not None

    def test_benchmark_ratings_are_finite(self):
        players = [RandomPlayer(name=f"p{i}") for i in range(3)]
        config = EloRaterConfig(games_per_matchup=10, batch_size=1)
        rater = EloRater(players, game_factory=TicTacToe, config=config)
        for p in players:
            assert -5000 < p.elo_rating < 5000


class TestEloRaterFromRatedPlayers:
    def test_does_not_recalculate(self):
        players = [
            RandomPlayer(name="p0", elo_rating=1000.0),
            RandomPlayer(name="p1", elo_rating=1200.0),
        ]
        config = EloRaterConfig(games_per_matchup=10, batch_size=1)
        rater = EloRater.from_rated_players(players, game_factory=TicTacToe, config=config)
        assert players[0].elo_rating == 1000.0
        assert players[1].elo_rating == 1200.0


class TestEloRaterCalculateElo:
    def test_calculates_elo_for_new_player(self):
        benchmark = [RandomPlayer(name=f"p{i}") for i in range(3)]
        config = EloRaterConfig(games_per_matchup=10, batch_size=1)
        rater = EloRater(benchmark, game_factory=TicTacToe, config=config)

        new_player = RandomPlayer(name="challenger")
        rating = rater.calculate_elo(new_player)
        assert new_player.elo_rating is not None
        assert new_player.elo_rating == rating
        assert -5000 < rating < 5000

    def test_does_not_change_benchmark_ratings(self):
        benchmark = [RandomPlayer(name=f"p{i}") for i in range(3)]
        config = EloRaterConfig(games_per_matchup=10, batch_size=1)
        rater = EloRater(benchmark, game_factory=TicTacToe, config=config)

        original_ratings = [p.elo_rating for p in benchmark]

        new_player = RandomPlayer(name="challenger")
        rater.calculate_elo(new_player)

        for p, orig in zip(benchmark, original_ratings):
            assert p.elo_rating == orig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/test_elo/test_elo_rater.py -v`
Expected: FAIL — `elo_rater` module not found

- [ ] **Step 3: Implement EloRater**

Create `src/pymcts/elo/elo_rater.py`:

```python
"""EloRater: rate players against a benchmark brigade."""

import itertools
from typing import Callable

from pymcts.core.arena import batched_arena
from pymcts.core.base_game import BaseGame
from pymcts.core.players import BasePlayer, RandomPlayer
from pymcts.elo.config import EloRaterConfig, MatchResult
from pymcts.elo.rating import compute_elo_ratings

_ANCHOR_NAME = "__elo_anchor__"


class EloRater:
    """Maintains a brigade of benchmark players and rates new players against them."""

    def __init__(
        self,
        benchmark_players: list[BasePlayer],
        game_factory: Callable[[], BaseGame],
        config: EloRaterConfig,
    ):
        self.benchmark_players = benchmark_players
        self.game_factory = game_factory
        self.config = config
        self._benchmark_match_results: list[MatchResult] = []
        self.recalculate_elo_for_benchmark_players()

    @classmethod
    def from_rated_players(
        cls,
        players: list[BasePlayer],
        game_factory: Callable[[], BaseGame],
        config: EloRaterConfig,
    ) -> "EloRater":
        instance = cls.__new__(cls)
        instance.benchmark_players = players
        instance.game_factory = game_factory
        instance.config = config
        instance._benchmark_match_results = []
        return instance

    def recalculate_elo_for_benchmark_players(self) -> None:
        anchor = RandomPlayer(name=_ANCHOR_NAME)
        all_players = [anchor] + list(self.benchmark_players)
        player_map = {p.name: p for p in all_players}

        match_results: list[MatchResult] = []
        for pa, pb in itertools.combinations(all_players, 2):
            records = batched_arena(
                player_a=pa,
                player_b=pb,
                game_factory=self.game_factory,
                num_games=self.config.games_per_matchup,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=False,
            )
            scores = records.scores
            wins_a = scores.get(pa.name, 0)
            wins_b = scores.get(pb.name, 0)
            draws = len(records) - wins_a - wins_b
            match_results.append(MatchResult(
                player_a=pa.name,
                player_b=pb.name,
                wins_a=wins_a,
                wins_b=wins_b,
                draws=draws,
            ))

        elo_ratings = compute_elo_ratings(
            match_results, anchor_player=_ANCHOR_NAME, anchor_rating=1000.0,
        )
        rating_map = {r.name: r.rating for r in elo_ratings}

        for p in self.benchmark_players:
            p.elo_rating = rating_map.get(p.name, 1000.0)

        self._benchmark_match_results = match_results

    def calculate_elo(self, player: BasePlayer) -> float:
        anchor = RandomPlayer(name=_ANCHOR_NAME)
        opponents = [anchor] + list(self.benchmark_players)

        new_match_results: list[MatchResult] = []
        for opponent in opponents:
            records = batched_arena(
                player_a=player,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=self.config.games_per_matchup,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=False,
            )
            scores = records.scores
            wins_a = scores.get(player.name, 0)
            wins_b = scores.get(opponent.name, 0)
            draws = len(records) - wins_a - wins_b
            new_match_results.append(MatchResult(
                player_a=player.name,
                player_b=opponent.name,
                wins_a=wins_a,
                wins_b=wins_b,
                draws=draws,
            ))

        all_results = self._benchmark_match_results + new_match_results
        elo_ratings = compute_elo_ratings(
            all_results, anchor_player=_ANCHOR_NAME, anchor_rating=1000.0,
        )
        rating_map = {r.name: r.rating for r in elo_ratings}

        rating = rating_map.get(player.name, 1000.0)
        player.elo_rating = rating

        # Restore benchmark ratings (don't let recomputation shift them)
        for p in self.benchmark_players:
            p.elo_rating = p.elo_rating  # no-op, kept for clarity

        return rating
```

Note on `calculate_elo`: Since we feed all match results (benchmark + new) into `compute_elo_ratings`, the optimizer may shift benchmark ratings slightly. To truly keep them fixed, we save and restore them. The benchmark `elo_rating` attributes are not modified because we re-assign from the saved values. However, a cleaner approach: after computing, just read the new player's rating and ignore the rest.

Actually, the simpler correct implementation: we just read `player`'s rating from the result and don't touch benchmark ratings at all. The `p.elo_rating = p.elo_rating` line is a no-op and can be removed. The benchmark players' `elo_rating` attributes are never overwritten in `calculate_elo` — only `player.elo_rating` is set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_elo/test_elo_rater.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/pymcts/elo/elo_rater.py test/test_elo/test_elo_rater.py
git commit -m "feat: implement EloRater class"
```

---

### Task 7: Clean up elo module — remove old tournament code, update exports

**Files:**
- Delete: `src/pymcts/elo/tournament.py` (contents replaced by elo_rater.py)
- Modify: `src/pymcts/elo/__init__.py`
- Delete: `test/test_elo/test_tournament.py`
- Delete: `test/test_elo/test_integration.py`

- [ ] **Step 1: Delete tournament.py**

```bash
rm src/pymcts/elo/tournament.py
```

- [ ] **Step 2: Delete old tournament tests**

```bash
rm test/test_elo/test_tournament.py
rm test/test_elo/test_integration.py
```

- [ ] **Step 3: Update elo __init__.py**

Replace `src/pymcts/elo/__init__.py`:

```python
from pymcts.elo.config import EloRaterConfig, EloRating, MatchResult, TournamentResult
from pymcts.elo.elo_rater import EloRater
from pymcts.elo.rating import compute_elo_ratings
```

- [ ] **Step 4: Run all tests to verify nothing is broken**

Run: `python -m pytest test/ -v`
Expected: ALL PASS (trainer tests and elo tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove old tournament code, update elo module exports"
```

---

### Task 8: Update core __init__.py exports

**Files:**
- Modify: `src/pymcts/core/__init__.py`

- [ ] **Step 1: Verify core __init__.py doesn't export anything from elo**

The current `src/pymcts/core/__init__.py` does not export elo types — no changes needed. Verify by reading the file.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest test/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit (only if changes were needed)**

No commit needed if no changes were made.
