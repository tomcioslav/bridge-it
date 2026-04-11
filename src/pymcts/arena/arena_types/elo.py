"""EloArena: Elo pool-based evaluation."""

import json
import logging
from pathlib import Path
from typing import Callable

from pymcts.arena.base import Arena
from pymcts.arena.config import EloArenaConfig
from pymcts.arena.engine import batched_arena
from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecord, GameRecordCollection
from pymcts.core.players import BasePlayer, MCTSPlayer, RandomPlayer
from pymcts.elo.config import MatchResult
from pymcts.elo.rating import compute_elo_against_pool

logger = logging.getLogger("pymcts.arena.elo")


class EloArena(Arena):
    """Arena that evaluates candidates using Elo ratings against a player pool.

    - play_games: player plays against pool players
    - is_candidate_better: candidate plays pool, Elo must improve by threshold
    """

    def __init__(
        self,
        config: EloArenaConfig,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        super().__init__(config, game_factory, arena_dir, verbose)
        self._pool_dir = self.arena_dir / "pool"
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._iteration = 0
        self._current_elo: float | None = None

        # Pool: list of (name, player, elo)
        self._pool: list[tuple[str, BasePlayer, float]] = []
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize pool with a RandomPlayer and optional seed players."""
        random_player = RandomPlayer(name="random")
        random_player.elo = 1000.0
        random_player.save(self._pool_dir / "random")
        self._pool.append(("random", random_player, 1000.0))

        if self.config.initial_pool:
            for path in self.config.initial_pool:
                loaded = MCTSPlayer.load(path)
                elo = loaded.elo if loaded.elo is not None else 1000.0
                loaded.save(self._pool_dir / loaded.name)
                self._pool.append((loaded.name, loaded, elo))

    def _pool_ratings(self) -> dict[str, float]:
        return {name: elo for name, _, elo in self._pool}

    def _play_vs_pool(self, player: BasePlayer) -> tuple[list[MatchResult], GameRecordCollection]:
        """Play player against every pool member. Returns match results and all game records."""
        match_results: list[MatchResult] = []
        all_records: list[GameRecord] = []

        for name, opponent, _ in self._pool:
            records = batched_arena(
                player_a=player,
                player_b=opponent,
                game_factory=self.game_factory,
                num_games=self.config.games_per_matchup,
                batch_size=self.config.batch_size,
                swap_players=self.config.swap_players,
                verbose=self.verbose,
            )
            all_records.extend(records.game_records)
            scores = records.scores
            match_results.append(MatchResult(
                player_a=player.name,
                player_b=name,
                wins_a=scores.get(player.name, 0),
                wins_b=scores.get(name, 0),
                draws=len(records) - scores.get(player.name, 0) - scores.get(name, 0),
            ))

        return match_results, GameRecordCollection(game_records=all_records)

    def _evict_weakest(self) -> None:
        weakest_idx, weakest_elo = None, float("inf")
        for idx, (name, _, elo) in enumerate(self._pool):
            if name != "random" and elo < weakest_elo:
                weakest_elo = elo
                weakest_idx = idx
        if weakest_idx is not None:
            evicted = self._pool.pop(weakest_idx)[0]
            if self.verbose:
                print(f"  Pool: evicted {evicted} (Elo {weakest_elo:.0f})")

    def _grow_pool(self, player: BasePlayer, elo: float) -> None:
        self._iteration += 1
        name = f"pool_iteration_{self._iteration:03d}"
        player.save(self._pool_dir / name)
        self._pool.append((name, player, elo))
        if self.config.max_pool_size is not None and len(self._pool) > self.config.max_pool_size:
            self._evict_weakest()

    def _save_elo_ratings(self) -> None:
        ratings = {name: elo for name, _, elo in self._pool}
        if self._current_elo is not None:
            ratings["_current_best"] = self._current_elo
        (self.arena_dir / "elo_ratings.json").write_text(json.dumps(ratings, indent=2))

    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Player plays against pool players."""
        _, records = self._play_vs_pool(player)
        return records

    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate candidate Elo against the pool."""
        self._iteration += 1
        pool_ratings = self._pool_ratings()
        match_results, records = self._play_vs_pool(candidate)
        post_elo = compute_elo_against_pool(candidate.name, pool_ratings, match_results)

        if self._current_elo is None:
            self._current_elo = post_elo
            accepted = True
        else:
            accepted = post_elo >= self._current_elo + self.config.elo_threshold

        if self.verbose:
            current_str = f"{self._current_elo:.0f}" if self._current_elo is not None else "N/A"
            print(f"  Elo: {post_elo:.0f} | Current: {current_str} | "
                  f"Threshold: +{self.config.elo_threshold:.0f}")

        if accepted:
            self._current_elo = post_elo
            if self.verbose:
                print("  -> ACCEPTED")
        else:
            if self.verbose:
                print("  -> REJECTED")

        if self._iteration % self.config.pool_growth_interval == 0:
            self._grow_pool(candidate, post_elo)

        self._save_elo_ratings()

        details = {
            "post_elo": post_elo,
            "current_elo": self._current_elo,
            "threshold": self.config.elo_threshold,
            "pool_size": len(self._pool),
        }

        # Save evaluation games
        eval_dir = self.arena_dir / f"eval_{self._iteration:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_games.json").write_text(records.model_dump_json(indent=2))
        (eval_dir / "eval_result.json").write_text(json.dumps(details, indent=2))

        return EvaluationResult(accepted=accepted, details=details)
