"""Arena for pitting two agents against each other."""

from typing import Callable

from bridgit.game import Bridgit, Player
from bridgit.ai.config import Config
from bridgit.ai.game_wrapper import GameWrapper


# A player function takes a game state and returns an action (int)
PlayerFn = Callable[[Bridgit], int]


class Arena:
    """Play games between two agents and track results."""

    def __init__(self, player1: PlayerFn, player2: PlayerFn, config: Config):
        self.player1 = player1
        self.player2 = player2
        self.config = config
        self.game_wrapper = GameWrapper(config.board_size)

    def play_game(self, player1_starts: bool = True) -> int:
        """Play a single game.

        Returns:
            1 if player1 wins, -1 if player2 wins
        """
        game = self.game_wrapper.new_game()

        # Map game players to arena players
        if player1_starts:
            players = {Player.HORIZONTAL: self.player1, Player.VERTICAL: self.player2}
        else:
            players = {Player.HORIZONTAL: self.player2, Player.VERTICAL: self.player1}

        while not game.game_over:
            player_fn = players[game.current_player]
            action = player_fn(game)
            row, col = self.game_wrapper.action_to_move(action, game)
            game.make_move(row, col)

        # Determine who won in arena terms
        if player1_starts:
            return 1 if game.winner == Player.HORIZONTAL else -1
        else:
            return 1 if game.winner == Player.VERTICAL else -1

    def play_games(self, num_games: int) -> tuple[int, int]:
        """Play multiple games, alternating who goes first.

        Returns:
            (player1_wins, player2_wins)
        """
        p1_wins = 0
        p2_wins = 0

        for i in range(num_games):
            player1_starts = (i % 2 == 0)
            result = self.play_game(player1_starts=player1_starts)
            if result == 1:
                p1_wins += 1
            else:
                p2_wins += 1

        return p1_wins, p2_wins
