"""Arena for pitting two agents against each other."""

from typing import Callable

from bridgit.players.players import BasePlayer
from bridgit.schema import Move
from bridgit.game import Bridgit, Player as GamePlayer
from bridgit.config import BoardConfig


# Legacy: A player function takes a game state and returns a move (row, col)
PlayerFn = Callable[[Bridgit], Move]


class Arena:
    """Play games between two agents and track results."""

    def __init__(self, player1: BasePlayer | PlayerFn, player2: BasePlayer | PlayerFn, board: BoardConfig):
        self.player1 = player1
        self.player2 = player2
        self.board_config = board

    def _get_action(self, player: BasePlayer | PlayerFn, game: Bridgit) -> Move:
        """Get move from player, handling both BasePlayer objects and legacy functions."""
        if isinstance(player, BasePlayer):
            return player.get_action(game)
        else:
            return player(game)

    def play_game(self, player1_starts: bool = True, verbose: bool = False) -> int:
        """Play a single game.

        Args:
            player1_starts: If True, player1 goes first as HORIZONTAL
            verbose: If True, print game progress

        Returns:
            1 if player1 wins, -1 if player2 wins
        """
        game = Bridgit(self.board_config)

        # Map game players to arena players
        if player1_starts:
            players = {GamePlayer.HORIZONTAL: self.player1, GamePlayer.VERTICAL: self.player2}
        else:
            players = {GamePlayer.HORIZONTAL: self.player2, GamePlayer.VERTICAL: self.player1}

        if verbose:
            p1_name = self.player1.name if isinstance(self.player1, BasePlayer) else "Player1"
            p2_name = self.player2.name if isinstance(self.player2, BasePlayer) else "Player2"
            if player1_starts:
                print(f"{p1_name} (HORIZONTAL) vs {p2_name} (VERTICAL)")
            else:
                print(f"{p2_name} (HORIZONTAL) vs {p1_name} (VERTICAL)")

        while not game.game_over:
            player = players[game.current_player]
            move = self._get_action(player, game)
            game.make_move(move)

            if verbose:
                player_name = self.player1.name if player == self.player1 else self.player2.name
                if not isinstance(player, BasePlayer):
                    player_name = "Player1" if player == self.player1 else "Player2"
                print(f"Move {game.move_count}: ({move.row},{move.col}) by {player_name}")

        # Determine who won in arena terms
        if player1_starts:
            result = 1 if game.winner == GamePlayer.HORIZONTAL else -1
        else:
            result = 1 if game.winner == GamePlayer.VERTICAL else -1

        if verbose:
            winner_name = self.player1.name if result == 1 else self.player2.name
            if not isinstance(self.player1, BasePlayer):
                winner_name = "Player1" if result == 1 else "Player2"
            print(f"Winner: {winner_name}\n")

        return result

    def play_games(self, num_games: int, verbose: bool = False) -> tuple[int, int, int]:
        """Play multiple games, alternating who goes first.

        Args:
            num_games: Number of games to play
            verbose: If True, print progress

        Returns:
            (player1_wins, player2_wins, draws) - draws should always be 0 for Bridgit
        """
        p1_wins = 0
        p2_wins = 0

        for i in range(num_games):
            player1_starts = (i % 2 == 0)
            result = self.play_game(player1_starts=player1_starts, verbose=verbose)
            if result == 1:
                p1_wins += 1
            else:
                p2_wins += 1

            if verbose:
                print(f"Games: {i+1}/{num_games}, Score: {p1_wins}-{p2_wins}")

        return p1_wins, p2_wins, 0
