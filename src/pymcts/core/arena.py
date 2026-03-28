"""Arena: run evaluation games between two players."""

import logging
from typing import Callable

import torch
from tqdm.auto import tqdm

from pymcts.core.base_game import BaseGame
from pymcts.core.mcts import MCTS
from pymcts.core.players import BasePlayer, MCTSPlayer
from pymcts.core.game_record import GameRecord, GameRecordCollection, MoveRecord

logger = logging.getLogger("core.arena")


def batched_arena(
    player_a: BasePlayer,
    player_b: BasePlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    batch_size: int = 8,
    swap_players: bool = False,
    temperature: float = 0.0,
    game_type: str = "arena",
    verbose: bool = True,
) -> GameRecordCollection:
    """Play arena games between two players.

    If both players are MCTSPlayer, uses batched MCTS inference for speed.
    Otherwise, falls back to sequential get_action() calls.
    """
    both_mcts = isinstance(player_a, MCTSPlayer) and isinstance(player_b, MCTSPlayer)

    if both_mcts:
        return _batched_mcts_arena(
            player_a, player_b, game_factory, num_games,
            batch_size, swap_players, temperature, game_type, verbose,
        )
    else:
        return _sequential_arena(
            player_a, player_b, game_factory, num_games,
            swap_players, game_type, verbose,
        )


def _sequential_arena(
    player_a: BasePlayer,
    player_b: BasePlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    swap_players: bool,
    game_type: str,
    verbose: bool,
) -> GameRecordCollection:
    """Play games sequentially using player.get_action()."""
    name_a = player_a.name
    name_b = player_b.name
    half = num_games // 2 if swap_players else num_games
    completed: list[GameRecord] = []

    pbar = tqdm(total=num_games, desc=f"{name_a} vs {name_b}", leave=False) if verbose else None

    for game_idx in range(num_games):
        swapped = swap_players and game_idx >= half
        game = game_factory()

        if swapped:
            players = [player_b, player_a]
            names = [name_b, name_a]
        else:
            players = [player_a, player_b]
            names = [name_a, name_b]

        moves: list[MoveRecord] = []
        while not game.is_over:
            current = game.current_player
            action = players[current].get_action(game)
            moves.append(MoveRecord(
                action=action,
                player=current,
                policy=players[current].last_policy,
            ))
            game.make_action(action)

        completed.append(GameRecord(
            game_type=game_type,
            game_config=game.get_config(),
            moves=moves,
            winner=game.winner,
            player_names=names,
        ))

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return GameRecordCollection(game_records=completed)


def _batched_mcts_arena(
    player_a: MCTSPlayer,
    player_b: MCTSPlayer,
    game_factory: Callable[[], BaseGame],
    num_games: int,
    batch_size: int,
    swap_players: bool,
    temperature: float,
    game_type: str,
    verbose: bool,
) -> GameRecordCollection:
    """Play arena games with batched MCTS inference (both players must be MCTSPlayer)."""
    name_a = player_a.name
    name_b = player_b.name
    mcts_a = player_a.mcts
    mcts_b = player_b.mcts

    half = num_games // 2 if swap_players else num_games
    active_size = min(batch_size, num_games)

    games: list[BaseGame] = []
    move_histories: list[list[MoveRecord]] = []
    slot_names: list[list[str]] = []
    slot_mcts: list[dict[int, MCTS]] = []

    def _make_slot(game_idx: int) -> tuple[BaseGame, list, list[str], dict[int, MCTS]]:
        swapped = swap_players and game_idx >= half
        game = game_factory()
        if swapped:
            names = [name_b, name_a]
            mcts_map = {0: mcts_b, 1: mcts_a}
        else:
            names = [name_a, name_b]
            mcts_map = {0: mcts_a, 1: mcts_b}
        return game, [], names, mcts_map

    for i in range(active_size):
        game, hist, names, mcts_map = _make_slot(i)
        games.append(game)
        move_histories.append(hist)
        slot_names.append(names)
        slot_mcts.append(mcts_map)

    completed: list[GameRecord] = []
    games_started = active_size
    recorded: set[int] = set()

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc=f"{name_a} vs {name_b}", leave=False)

    wins_a = 0
    wins_b = 0
    first_player_wins = 0
    second_player_wins = 0

    while len(completed) < num_games:
        active_idx = [
            i for i in range(len(games))
            if not games[i].is_over and i not in recorded
        ]
        if not active_idx:
            break

        groups: dict[int, list[int]] = {}
        for i in active_idx:
            current = games[i].current_player
            mcts_obj = slot_mcts[i][current]
            key = id(mcts_obj)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        mcts_by_id = {}
        for i in active_idx:
            current = games[i].current_player
            mcts_obj = slot_mcts[i][current]
            mcts_by_id[id(mcts_obj)] = mcts_obj

        for mcts_id, slot_indices in groups.items():
            mcts_obj = mcts_by_id[mcts_id]
            batch_games = [games[i] for i in slot_indices]
            roots = mcts_obj.search_batch(batch_games)

            for j, i in enumerate(slot_indices):
                root = roots[j]
                action_space = games[i].action_space_size
                visit_counts = root.visit_counts(action_space)
                probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

                if probs.sum() == 0:
                    probs = games[i].to_mask().float()

                if temperature > 0:
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = torch.argmax(probs).item()

                current_player = games[i].current_player
                games[i].make_action(action)

                move_histories[i].append(MoveRecord(
                    action=action,
                    player=current_player,
                    policy=probs,
                ))

        for i in range(len(games)):
            if games[i].is_over and i not in recorded:
                record = GameRecord(
                    game_type=game_type,
                    game_config=games[i].get_config(),
                    moves=move_histories[i],
                    winner=games[i].winner,
                    player_names=slot_names[i],
                )
                completed.append(record)

                if record.winner is not None:
                    winner_name = record.player_names[record.winner]
                    if winner_name == name_a:
                        wins_a += 1
                    else:
                        wins_b += 1
                    if record.winner == 0:
                        first_player_wins += 1
                    else:
                        second_player_wins += 1

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"{name_a}={wins_a} | {name_b}={wins_b} | "
                        f"1st={first_player_wins} 2nd={second_player_wins}"
                    )

                if games_started < num_games:
                    game, hist, names, mcts_map = _make_slot(games_started)
                    games[i] = game
                    move_histories[i] = hist
                    slot_names[i] = names
                    slot_mcts[i] = mcts_map
                    games_started += 1
                else:
                    recorded.add(i)

    if pbar is not None:
        pbar.close()

    return GameRecordCollection(game_records=completed)
