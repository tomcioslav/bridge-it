"""Arena base class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from pymcts.arena.models import EvaluationResult
from pymcts.core.base_game import BaseGame
from pymcts.core.game_record import GameRecordCollection
from pymcts.core.players import BasePlayer


class Arena(ABC):
    """Base class for all arena types.

    An Arena is a stateful evaluator that:
    - Plays games between players (for training data or evaluation)
    - Manages its own directory (persisting players, game records, history)
    - Decides whether a candidate player is better than the current best
    """

    def __init__(
        self,
        config,
        game_factory: Callable[[], BaseGame],
        arena_dir: Path,
        verbose: bool = True,
    ):
        self.config = config
        self.game_factory = game_factory
        self.arena_dir = arena_dir
        self.verbose = verbose
        self.arena_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def play_games(self, player: BasePlayer, num_games: int) -> GameRecordCollection:
        """Play games for training data generation."""
        ...

    @abstractmethod
    def is_candidate_better(self, candidate: BasePlayer) -> EvaluationResult:
        """Evaluate whether candidate should replace the current best."""
        ...
