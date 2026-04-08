# Elo-Based Arena Evaluation

## Overview

Replace the single arena evaluation strategy with a pluggable arena system. The `train()` function accepts an `arena` parameter — either `ArenaConfig` (head-to-head, current behavior) or `EloArenaConfig` (Elo pool-based, new behavior). Both strategies manage an `arena/` directory inside the training run for persisting players.

## Motivation

The current head-to-head approach only compares the post-training model against the pre-training model. This is a narrow signal — a model might beat its immediate predecessor but still be weaker than models from earlier iterations. An Elo-based evaluation against a diverse pool of players gives a richer, more stable measure of model strength.

## Configuration

### `ArenaConfig` (existing, adapted)

No changes to fields. Behavior change: accepted players are now saved to `arena/iteration_NNN/` using `MCTSPlayer.save()`.

```python
class ArenaConfig(BaseModel):
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
```

### `EloArenaConfig` (new)

```python
class EloArenaConfig(BaseModel):
    games_per_matchup: int = 40
    elo_threshold: float = 20.0
    pool_growth_interval: int = 5
    max_pool_size: int | None = None
    swap_players: bool = True
    initial_pool: list[str] | None = None  # paths to player dirs
```

- `games_per_matchup`: number of games played against each pool player per evaluation
- `elo_threshold`: minimum Elo improvement over current model's Elo to accept
- `pool_growth_interval`: add current player to pool every N iterations
- `max_pool_size`: cap on pool size; `None` means unlimited
- `swap_players`: play both sides (first/second player) for fairness
- `initial_pool`: optional list of paths to player directories (saved with `MCTSPlayer.save()`) to seed the pool. If `None`, pool starts with just a `RandomPlayer`.

### `train()` signature change

Replace `arena_config: ArenaConfig` with `arena: ArenaConfig | EloArenaConfig`. The training loop branches based on which type it receives.

## MCTSPlayer changes

Add an `elo: float | None` field to `MCTSPlayer`:

- Stored in `player.json` alongside existing fields
- Defaults to `None` (no impact on existing saves/loads)
- Set when a player is evaluated against the pool
- Persisted on `save()`, restored on `load()`

## `arena/` directory

Located at `trainings/run_xxx/arena/`. Owned by the active arena strategy.

### With `ArenaConfig` (head-to-head)

```
arena/
  iteration_001/
    model.pt
    player.json
  iteration_003/     # iteration_002 was rejected
    model.pt
    player.json
```

Only accepted players are saved here.

### With `EloArenaConfig` (pool-based)

```
arena/
  random/            # RandomPlayer (always present)
    player.json      # no model.pt needed for RandomPlayer
  iteration_001/
    model.pt
    player.json      # includes frozen elo
  iteration_005/     # added at pool_growth_interval
    model.pt
    player.json
  ...
```

Contains all pool players (initial + grown). Each player's `player.json` includes their frozen Elo rating.

## Elo Arena per-iteration flow

1. Train the model, save post-training checkpoint
2. Create `GreedyMCTSPlayer` from post-training weights
3. Play post-training player against every player in the pool
4. Compute post-training player's Elo using `compute_elo_ratings()` with pool players' Elo ratings frozen
5. **Accept if** post-training Elo >= current model's known Elo + `elo_threshold`
6. On accept: update current Elo to post-training Elo; save player to `arena/iteration_NNN/`
7. On reject: revert to pre-training weights; current Elo stays unchanged
8. Every `pool_growth_interval` iterations: add current player to pool with its frozen Elo
9. When pool exceeds `max_pool_size`: evict the player with the lowest Elo (never evict `RandomPlayer`)

### First iteration handling

No known Elo exists for the current model. Evaluate both pre-training and post-training players against the pool to establish a baseline. The pre-training Elo becomes the initial "current Elo" and the normal accept/reject logic applies.

### Frozen Elo computation

Pool players have fixed Elo ratings assigned when they entered the pool. When evaluating a new player:

1. Play the candidate against each pool player
2. Compute the candidate's Elo from the match results, treating pool players' Elo ratings as fixed constants. The candidate's rating is the one that maximizes the likelihood of the observed results given the known pool ratings. `RandomPlayer` at 1000.0 serves as the anchor.
3. Extract the candidate's computed Elo

Note: this may require a small wrapper around `compute_elo_ratings()` or a new function that optimizes a single player's rating against fixed opponents, rather than recomputing all ratings jointly.

## RandomPlayer serialization

`RandomPlayer` needs a `save()` method to be stored in the `arena/` directory. It saves a `player.json` with:

```json
{
  "type": "random",
  "name": "random",
  "elo": 1000.0
}
```

No `model.pt` needed. A corresponding `load()` classmethod reconstructs it.

## Backwards compatibility

- Existing `ArenaConfig` usage continues to work with the added benefit of player persistence in `arena/`
- The `elo` field on `MCTSPlayer` defaults to `None`, so existing saved players load without issues
- The `initial_pool` field on `EloArenaConfig` is optional — omitting it gives fully organic growth from `RandomPlayer`
- A previous run's `arena/` directory can be passed as `initial_pool` paths for a new run

## Reusability

The `arena/` directory from any training run can be reused:

- Point `initial_pool` at paths like `["trainings/run_xxx/arena/iteration_001", "trainings/run_xxx/arena/iteration_010"]`
- Or load individual players with `MCTSPlayer.load("trainings/run_xxx/arena/iteration_005")`

## Interaction with existing Elo tracking

The existing `elo_tracking` feature in `TrainingConfig` is a separate reporting mechanism. The new Elo arena is a decision-making mechanism. They can coexist independently — `elo_tracking` reports ratings for monitoring, `EloArenaConfig` uses ratings for accept/reject decisions.
