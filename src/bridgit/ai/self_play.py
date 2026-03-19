"""Batched self-play: run N games concurrently with batched neural net inference."""

import logging

import numpy as np
import torch
from tqdm.auto import tqdm

from bridgit.ai.mcts import MCTS, MCTSNode
from bridgit.ai.neural_net import NetWrapper
from bridgit.config import BoardConfig, MCTSConfig
from bridgit.game import Bridgit
from bridgit.schema import GameRecord, GameRecordCollection, Move, MoveRecord
from bridgit.schema.player import Player

logger = logging.getLogger("bridgit.self_play")


class BatchedMCTS:
    """MCTS that batches neural net calls across multiple trees."""

    def __init__(self, net_wrapper: NetWrapper, mcts_config: MCTSConfig):
        self.net_wrapper = net_wrapper
        self.mcts_config = mcts_config

    def _predict_batch(self, games: list[Bridgit]) -> list[tuple[torch.Tensor, float]]:
        """Run neural net on a batch of game states.

        Returns list of (policy, value) tuples, one per game.
        """
        model = self.net_wrapper.model
        device = self.net_wrapper.device

        model.eval()
        tensors = torch.stack([g.to_tensor() for g in games]).to(device)

        with torch.no_grad():
            log_policies, values = model(tensors)

        policies = torch.exp(log_policies).cpu()  # (N, g, g)
        vals = values.cpu()  # (N, 1)

        return [
            (policies[i], vals[i].item())
            for i in range(len(games))
        ]

    def _expand_with_policy(self, node: MCTSNode, policy: torch.Tensor, value: float) -> float:
        """Expand a node using a pre-computed policy and value."""
        if node.game.game_over:
            node.is_expanded = True
            return 1.0

        valid_mask = node.game.to_mask()

        # Mask and renormalize
        policy = policy * valid_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            total = valid_mask.sum()
            if total > 0:
                policy = valid_mask / total
            else:
                node.is_expanded = True
                return value

        player = node.game.current_player
        for idx, (r, c) in enumerate(torch.nonzero(valid_mask, as_tuple=False)):
            r, c = r.item(), c.item()
            canonical_move = Move(row=r, col=c)
            actual_move = canonical_move.decanonicalize(player)
            child_game = node.game.copy()
            child_game.make_move(actual_move)
            child = MCTSNode(child_game, parent=node, action=canonical_move,
                             prior=policy[r, c].item(), child_index=idx)
            node.children[canonical_move] = child

        node.is_expanded = True
        return value

    @staticmethod
    def _select_leaf(root: MCTSNode, c_puct: float) -> MCTSNode:
        """Select a leaf node from the tree."""
        node = root
        while node.is_expanded and node.children:
            node = node.best_child(c_puct)
        return node

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    @staticmethod
    def _add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
        """Add Dirichlet noise to root priors."""
        if not node.children:
            return
        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        for i, move in enumerate(moves):
            node.children[move].prior = (
                (1 - epsilon) * node.children[move].prior + epsilon * noise[i]
            )

    def search_batch(self, games: list[Bridgit]) -> list[MCTSNode]:
        """Run MCTS for multiple games with batched neural net inference.

        Returns a list of root nodes, one per game.
        """
        n = len(games)
        roots = [MCTSNode(g.copy()) for g in games]

        # Initial expansion — batch all roots
        predictions = self._predict_batch([r.game for r in roots])
        for i in range(n):
            policy, value = predictions[i]
            self._expand_with_policy(roots[i], policy, value)
            self._add_dirichlet_noise(
                roots[i],
                self.mcts_config.dirichlet_alpha,
                self.mcts_config.dirichlet_epsilon,
            )

        # Run simulations
        c_puct = self.mcts_config.c_puct
        for _ in range(self.mcts_config.num_simulations):
            # Select leaf nodes
            leaves = [self._select_leaf(root, c_puct) for root in roots]

            # Separate terminal vs non-terminal leaves
            to_predict_idx = []
            to_predict_games = []
            for i, leaf in enumerate(leaves):
                if leaf.game.game_over:
                    self._backpropagate(leaf, 1.0)
                elif leaf.is_expanded:
                    # Already expanded (e.g. all children visited) — just backprop
                    self._backpropagate(leaf, leaf.q_value if leaf.visit_count > 0 else 0.0)
                else:
                    to_predict_idx.append(i)
                    to_predict_games.append(leaf.game)

            if not to_predict_games:
                continue

            # Batch predict
            predictions = self._predict_batch(to_predict_games)

            # Expand and backpropagate
            for j, idx in enumerate(to_predict_idx):
                policy, value = predictions[j]
                value = self._expand_with_policy(leaves[idx], policy, value)
                self._backpropagate(leaves[idx], value)

        return roots


def batched_self_play(
    net_wrapper: NetWrapper,
    board_config: BoardConfig,
    mcts_config: MCTSConfig,
    num_games: int,
    batch_size: int = 8,
    temperature: float = 1.0,
    verbose: bool = True,
) -> GameRecordCollection:
    """Play self-play games with batched MCTS inference.

    Runs `batch_size` games concurrently. When a game finishes,
    a new one starts in its slot until all games are complete.

    Args:
        net_wrapper: Neural net wrapper.
        board_config: Board configuration.
        mcts_config: MCTS configuration.
        num_games: Total games to play.
        batch_size: Number of concurrent games.
        temperature: MCTS temperature for move selection.
        verbose: Show progress bar.

    Returns:
        GameRecordCollection with all completed games.
    """
    batched_mcts = BatchedMCTS(net_wrapper, mcts_config)
    g = board_config.grid_size

    # Active game slots
    active_size = min(batch_size, num_games)
    games = [Bridgit(board_config) for _ in range(active_size)]
    move_histories: list[list[MoveRecord]] = [[] for _ in range(active_size)]

    completed: list[GameRecord] = []
    games_started = active_size

    pbar = None
    if verbose:
        pbar = tqdm(total=num_games, desc="Self-play")

    while len(completed) < num_games:
        # Run batched MCTS for all active games that aren't over
        active_idx = [i for i in range(len(games)) if not games[i].game_over]
        if not active_idx:
            break

        active_games = [games[i] for i in active_idx]
        roots = batched_mcts.search_batch(active_games)

        # Pick moves and advance games
        for j, i in enumerate(active_idx):
            root = roots[j]
            visit_counts = root.visit_counts()
            probs = MCTS.visit_counts_to_probs(visit_counts, temperature)

            if probs.sum() == 0:
                probs = games[i].to_mask()

            # Sample move
            flat_idx = torch.multinomial(probs.flatten(), 1).item()
            row, col = divmod(flat_idx, g)
            canonical_move = Move(row=row, col=col)
            actual_move = canonical_move.decanonicalize(games[i].current_player)

            current_player = games[i].current_player
            if not games[i].make_move(actual_move):
                logger.error("Invalid move (%d,%d) in batched self-play", row, col)
                continue

            move_histories[i].append(MoveRecord(
                move=actual_move,
                player=current_player,
                moves_left_after=games[i].moves_left_in_turn,
                policy=probs,
            ))

        # Check for completed games and replace them
        for i in range(len(games)):
            if games[i].game_over:
                record = GameRecord(
                    board_size=board_config.size,
                    moves=move_histories[i],
                    winner=games[i].winner,
                    horizontal_player="self-play",
                    vertical_player="self-play",
                )
                completed.append(record)

                if pbar is not None:
                    pbar.update(1)

                logger.debug("Game %d/%d done: %s wins (%d moves)",
                             len(completed), num_games,
                             games[i].winner.name, len(move_histories[i]))

                # Start a new game in this slot if needed
                if games_started < num_games:
                    games[i] = Bridgit(board_config)
                    move_histories[i] = []
                    games_started += 1
                else:
                    # Mark as done — replace with a dummy finished game
                    # so it gets skipped in the active_idx filter
                    pass  # game_over is already True, will be skipped

    if pbar is not None:
        pbar.close()

    logger.info("Batched self-play: %d games completed", len(completed))
    return GameRecordCollection(game_records=completed)
