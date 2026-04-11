"""All public Pydantic models from the arena package."""

from pydantic import BaseModel

from pymcts.arena.config import (
    EloArenaConfig,
    MultiPlayerArenaConfig,
    SinglePlayerArenaConfig,
)


class EvaluationResult(BaseModel):
    """Returned by Arena.is_candidate_better()."""
    accepted: bool
    details: dict


__all__ = [
    "EloArenaConfig",
    "EvaluationResult",
    "MultiPlayerArenaConfig",
    "SinglePlayerArenaConfig",
]
