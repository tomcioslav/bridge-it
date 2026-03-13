"""Move type for Bridgit."""

from pydantic import BaseModel


class Move(BaseModel, frozen=True):
    """A move on the Bridgit board."""
    row: int
    col: int
