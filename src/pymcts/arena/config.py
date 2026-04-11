"""Arena configuration models."""

from pydantic import BaseModel


class SinglePlayerArenaConfig(BaseModel):
    """Config for SinglePlayerArena (AlphaZero-style head-to-head)."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
    batch_size: int = 8


class MultiPlayerArenaConfig(BaseModel):
    """Config for MultiPlayerArena (play against top N historical players)."""
    num_games: int = 40
    threshold: float = 0.55
    swap_players: bool = True
    batch_size: int = 8
    top_n: int = 5


class EloArenaConfig(BaseModel):
    """Config for EloArena (Elo pool-based evaluation)."""
    games_per_matchup: int = 40
    elo_threshold: float = 20.0
    pool_growth_interval: int = 5
    max_pool_size: int | None = None
    swap_players: bool = True
    batch_size: int = 8
    initial_pool: list[str] | None = None
