# Elo Rater Redesign

## Overview

Redesign the elo module around an `EloRater` class that maintains a brigade of benchmark players with known Elo ratings, and can rate new players against them. Add save/load to all player types, `elo_rating` field to `BasePlayer`, and a `PLAYERS` path to project config.

## Changes to BasePlayer

- Add `elo_rating: float | None = None` parameter to `BasePlayer.__init__`
- Add abstract methods:
  - `save(path: str | Path) -> None`
  - `load(path: str | Path) -> BasePlayer` (classmethod)
- Both `MCTSPlayer` and `RandomPlayer` persist `elo_rating` in `player.json`

## RandomPlayer save/load

- `save(path)`: writes `player.json` with `{"type": "RandomPlayer", "name": "...", "elo_rating": ...}`
- `load(path)`: reads `player.json`, returns `RandomPlayer(name=..., elo_rating=...)`

## MCTSPlayer save/load updates

- Add `elo_rating` to the JSON config written by `save()`
- Restore `elo_rating` in `load()`

## PathsConfig

- Add `players: Path = _PROJECT_ROOT / "players"` to `PathsConfig`
- Convention: each player saved at `paths.players / player.name /`

## Elo module cleanup

Remove from the elo module:
- `TournamentConfig`, `TournamentResult` from `config.py`
- `RatedPlayer`, `run_tournament`, `_swiss_pair` from `tournament.py`

Keep:
- `MatchResult`, `EloRating` in `config.py`
- `compute_elo_ratings` in `rating.py`

## EloRaterConfig

```python
class EloRaterConfig(BaseModel):
    games_per_matchup: int = 40
    swap_players: bool = True
    batch_size: int = 8
```

Replaces `TournamentConfig`. Lives in `config.py`.

## EloRater class

Lives in a new `elo_rater.py` (replaces `tournament.py`).

### `__init__(self, benchmark_players: list[BasePlayer], game_factory: Callable[[], BaseGame], config: EloRaterConfig)`

Stores players, game_factory, config. Calls `recalculate_elo_for_benchmark_players()`.

### `from_rated_players(cls, players, game_factory, config) -> EloRater`

Class method. Creates instance without recalculating. Players must already have `elo_rating` set.

### `recalculate_elo_for_benchmark_players(self)`

1. Creates a `RandomPlayer(name="__elo_anchor__")` as anchor at 1000
2. Plays all pairs among benchmark_players + anchor (all-vs-all) using `batched_arena`
3. Calls `compute_elo_ratings(match_results, anchor_player="__elo_anchor__", anchor_rating=1000.0)`
4. Sets `elo_rating` on each benchmark player from the results

### `calculate_elo(self, player: BasePlayer) -> float`

1. Creates a `RandomPlayer(name="__elo_anchor__")` as anchor at 1000
2. Plays `player` vs each benchmark player AND vs the anchor
3. Computes Elo with benchmark ratings fixed + anchor fixed at 1000, only optimizes `player`'s rating
4. Sets `player.elo_rating` and returns it

For step 3: use `compute_elo_ratings` with all match results (benchmark mutual results from recalculation + new player's matches), anchor at 1000. The benchmark players' ratings should come out unchanged since their mutual match data is the same. Alternatively, we can compute a single-player ML estimate directly — but reusing `compute_elo_ratings` with all data is simpler and consistent.

## `__init__.py` exports

Update to export: `EloRating`, `MatchResult`, `EloRaterConfig`, `compute_elo_ratings`, `EloRater`.
