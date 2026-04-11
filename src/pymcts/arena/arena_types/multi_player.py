"""MultiPlayerArena: evaluate against top N historical players."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import MultiPlayerArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecord, GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer

logger = logging.getLogger("pymcts.arena.multi_player")


class MultiPlayerArena(Arena):
    """Arena that evaluates candidates against top N historical players.

    - play_games: player plays against itself AND top N players
    - is_candidate_better: candidate plays against top N, aggregate accept logic
    """

    def __init__(
        self,
        config: MultiPlayerArenaConfig,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        super().__init__(config, game_factory, arena_dir, verbose)
        self._history_dir = self.arena_dir / "history"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._best_so_far_dir = self.arena_dir / "best_so_far"
        self._iteration = 0

    def _has_best(self) -> bool:
        return (self._best_so_far_dir / "player.json").exists()

    def _load_best(self) -> BasePlayer:
        return MCTSPlayer.load(self._best_so_far_dir)

    def _save_as_best(self, player: BasePlayer) -> None:
        player.save(self._best_so_far_dir)

    def _save_to_history(self, player: BasePlayer) -> None:
        self._iteration += 1
        player.save(self._history_dir / f"iteration_{self._iteration:03d}")

    def _load_top_n(self) -> list[BasePlayer]:
        """Load the most recent top_n players from history."""
        player_dirs = sorted(self._history_dir.glob("iteration_*"))
        top_dirs = player_dirs[-self.config.top_n:]
        players = []
        for d in top_dirs:
            if (d / "player.json").exists():
                players.append(MCTSPlayer.load(d))
        return players

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against itself AND top N historical players."""
        all_records: list[GameRecord] = []

        # Self-play
        self_play_records = batched_arena(
            player_a=player,
            player_b=player,
            game_factory=self.game_factory,
            num_games=num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )
        all_records.extend(self_play_records.game_records)

        # Play against historical players
        pool = self._load_top_n()
        games_per_opponent = max(1, num_games // max(len(pool), 1))
        for opponent in pool:
            records = batched_arena(
                player_a=player,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=games_per_opponent,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)

        return GameRecordCollection(game_records=all_records)

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate against top N players (aggregate results)."""
        if not self._has_best():
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  No previous best — accepting first candidate.")
            return EvaluationResult(accepted=True, details={"first_candidate": True})

        pool = self._load_top_n()
        all_records: list[GameRecord] = []

        for opponent in pool:
            records = batched_arena(
                player_a=candidate,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=self.config.num_games,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)

        combined = GameRecordCollection(game_records=all_records)
        accepted = combined.is_better(candidate.name, self.config.threshold)
        result = combined.evaluate(candidate.name)

        if self.verbose:
            print(f"  {candidate.name} vs {len(pool)} opponents: "
                  f"{result.wins}W/{result.losses}L ({result.win_rate:.1%})")

        if accepted:
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  -> ACCEPTED")
        else:
            if self.verbose:
                print("  -> REJECTED")

        details = {
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "total": result.total,
            "win_rate": result.win_rate,
            "pool_size": len(pool),
        }

        eval_dir = self._history_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(combined.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
