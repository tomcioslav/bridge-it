# Engine Components

The game-agnostic engine that powers MCTS, self-play, training, and evaluation.

## MCTS

::: pymcts.core.mcts.MCTS

::: pymcts.core.mcts.MCTSNode

## Self-Play

::: pymcts.core.self_play.batched_self_play

## Trainer

::: pymcts.core.trainer.train

## Arena

::: pymcts.arena.base.Arena

::: pymcts.arena.arena_types.single_player.SinglePlayerArena

::: pymcts.arena.arena_types.multi_player.MultiPlayerArena

::: pymcts.arena.arena_types.elo.EloArena

### Game Engine

::: pymcts.arena.engine.batched_arena

## Players

::: pymcts.core.players.BasePlayer

::: pymcts.core.players.RandomPlayer

::: pymcts.core.players.MCTSPlayer

::: pymcts.core.players.GreedyMCTSPlayer

## Game Records

::: pymcts.core.game_record.MoveRecord

::: pymcts.core.game_record.GameRecord

::: pymcts.core.game_record.GameRecordCollection

::: pymcts.core.game_record.EvalResult

## Data

::: pymcts.core.data.examples_from_records
