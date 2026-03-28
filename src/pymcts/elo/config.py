"""Pydantic models for the Elo rating system."""

from pydantic import BaseModel


class MatchResult(BaseModel):
    """Outcome of a matchup between two players (multiple games)."""
    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.wins_a + self.wins_b + self.draws


class EloRating(BaseModel):
    """Elo rating for a single player."""
    name: str
    rating: float
    games_played: int


class TournamentConfig(BaseModel):
    """Configuration for a Swiss-style tournament."""
    games_per_matchup: int = 40
    swap_players: bool = True
    num_rounds: int | None = None
    convergence_threshold: float = 10.0
    batch_size: int = 8


class TournamentResult(BaseModel):
    """Complete result of a tournament: ratings + match history."""
    ratings: list[EloRating]
    match_results: list[MatchResult]
    anchor_player: str
    anchor_rating: float
    timestamp: str
    metadata: dict
