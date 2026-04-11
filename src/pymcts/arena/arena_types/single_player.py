"""SinglePlayerArena: AlphaZero-style head-to-head evaluation."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import SinglePlayerArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer

logger = logging.getLogger("pymcts.arena.single_player")


class SinglePlayerArena(Arena):
    """Arena that evaluates candidates against the current best player.

    - play_games: player plays against itself (self-play)
    - is_candidate_better: candidate plays against best_so_far
    """

    def __init__(
        self,
        config: SinglePlayerArenaConfig,
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

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against itself (self-play)."""
        return batched_arena(
            player_a=player,
            player_b=player,
            game_factory=self.game_factory,
            num_games=num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate against best_so_far."""
        if not self._has_best():
            self._save_as_best(candidate)
            self._save_to_history(candidate)
            if self.verbose:
                print("  No previous best — accepting first candidate.")
            return EvaluationResult(accepted=True, details={"first_candidate": True})

        best = self._load_best()
        records = batched_arena(
            player_a=candidate,
            player_b=best,
            game_factory=self.game_factory,
            num_games=self.config.num_games,
            batch_size=self.config.batch_size,
            swap_players=self.config.swap_players,
            verbose=self.verbose,
        )

        accepted = records.is_better(candidate.name, self.config.threshold)
        result = records.evaluate(candidate.name)

        if self.verbose:
            print(f"  {candidate.name}: {result.wins}W/{result.losses}L "
                  f"({result.win_rate:.1%})")

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
        }

        # Save evaluation games
        eval_dir = self._history_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(records.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
