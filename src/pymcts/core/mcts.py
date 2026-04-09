"""Game-agnostic Monte Carlo Tree Search with neural network guidance."""

import logging
import math

import numpy as np
import torch
from tqdm.auto import tqdm

from pymcts.core.base_game import BaseGame
from pymcts.core.base_neural_net import BaseNeuralNet
from pymcts.core.config import MCTSConfig

logger = logging.getLogger("bridgit.core.mcts")

_VIRTUAL_LOSS = 1.0


class MCTSNode:
    """A node in the MCTS tree.

    Uses lazy expansion: after neural net evaluation, only priors are stored
    (in `unexpanded_moves`). Child nodes are created on demand when selected.
    """

    __slots__ = ["game", "parent", "children", "visit_count",
                 "value_sum", "prior", "is_expanded", "unexpanded_moves"]

    def __init__(self, game: BaseGame, parent: "MCTSNode | None" = None,
                 prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.unexpanded_moves: dict[int, float] = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def fully_expanded(self) -> bool:
        """True when all moves have been turned into child nodes."""
        return self.is_expanded and not self.unexpanded_moves

    def best_child_or_expand(self, c_puct: float) -> "MCTSNode":
        """Select the best child, possibly creating one from unexpanded moves.

        Compares UCB scores of existing children with potential scores of
        unexpanded moves. If an unexpanded move wins, creates the child node.
        """
        best_score = -math.inf
        best_child = None
        best_unexpanded_action = -1
        best_unexpanded_prior = 0.0

        parent_visits = self.visit_count
        sqrt_parent = math.sqrt(parent_visits)
        my_player = self.game.current_player

        # Score existing children — inline UCB calculation
        for child in self.children.values():
            vc = child.visit_count
            if vc == 0:
                q = 0.0
            else:
                q = child.value_sum / vc
                if child.game.current_player != my_player:
                    q = -q
            score = q + c_puct * child.prior * sqrt_parent / (1 + vc)
            if score > best_score:
                best_score = score
                best_child = child
                best_unexpanded_action = -1

        # Score unexpanded moves (visit_count=0, q=0)
        explore_base = c_puct * sqrt_parent
        for action, prior in self.unexpanded_moves.items():
            score = explore_base * prior
            if score > best_score:
                best_score = score
                best_child = None
                best_unexpanded_action = action
                best_unexpanded_prior = prior

        if best_unexpanded_action >= 0:
            child_game = self.game.copy()
            child_game.make_action(best_unexpanded_action)
            child = MCTSNode(child_game, parent=self, prior=best_unexpanded_prior)
            self.children[best_unexpanded_action] = child
            del self.unexpanded_moves[best_unexpanded_action]
            return child

        return best_child

    def best_move(self) -> int:
        """Return the most-visited child's action."""
        return max(self.children, key=lambda a: self.children[a].visit_count)

    def visit_counts(self, action_space_size: int) -> torch.Tensor:
        """Return visit counts as a 1D tensor of length action_space_size."""
        visits = torch.zeros(action_space_size, dtype=torch.float32)
        for action, child in self.children.items():
            visits[action] = child.visit_count
        return visits

    @property
    def has_candidates(self) -> bool:
        """True if there are children or unexpanded moves to select from."""
        return bool(self.children) or bool(self.unexpanded_moves)


class MCTS:
    """Monte Carlo Tree Search guided by a neural network.

    Supports both single-game and batched multi-game search.
    When searching multiple games, uses batched neural net inference
    and virtual loss for parallel leaf selection.
    """

    def __init__(self, net: BaseNeuralNet, mcts_config: MCTSConfig):
        self.net = net
        self.mcts_config = mcts_config

    def _predict_batch(self, states: list) -> list[tuple[np.ndarray, float]]:
        """Run neural net on a batch of game states."""
        log_policies, values = self.net.predict_batch(states)
        policies = torch.exp(log_policies).numpy()
        value_list = values.tolist()
        return list(zip(policies, value_list))

    @staticmethod
    def _set_priors(node: MCTSNode, policy: np.ndarray, value: float) -> float:
        """Store masked/renormalized priors in node.unexpanded_moves."""
        if node.game.is_over:
            node.is_expanded = True
            return 1.0

        mask = node.game.to_mask().numpy()

        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            total = mask.sum()
            if total > 0:
                policy = mask.astype(np.float32) / total
            else:
                node.is_expanded = True
                return value

        indices = np.nonzero(mask)[0]
        priors = policy[indices]
        node.unexpanded_moves = dict(zip(indices.tolist(), priors.tolist()))

        node.is_expanded = True
        return value

    @staticmethod
    def _select_leaf(root: MCTSNode, c_puct: float) -> MCTSNode:
        """Select a leaf node using lazy expansion."""
        node = root
        while node.is_expanded and node.has_candidates:
            node = node.best_child_or_expand(c_puct)
            if not node.is_expanded:
                break
        return node

    @staticmethod
    def _apply_virtual_loss(node: MCTSNode):
        """Apply virtual loss along the path from node to root."""
        n = node
        while n is not None:
            n.visit_count += 1
            n.value_sum -= _VIRTUAL_LOSS
            n = n.parent

    @staticmethod
    def _undo_virtual_loss(node: MCTSNode):
        """Undo virtual loss along the path from node to root."""
        n = node
        while n is not None:
            n.visit_count -= 1
            n.value_sum += _VIRTUAL_LOSS
            n = n.parent

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float):
        """Backpropagate value, flipping sign on player boundaries."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    @staticmethod
    def _add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
        """Add Dirichlet noise to root priors for exploration."""
        all_actions = list(node.children.keys()) + list(node.unexpanded_moves.keys())
        if not all_actions:
            return
        noise = np.random.dirichlet([alpha] * len(all_actions))
        idx = 0
        for action in list(node.children.keys()):
            node.children[action].prior = (
                (1 - epsilon) * node.children[action].prior + epsilon * noise[idx]
            )
            idx += 1
        for action in list(node.unexpanded_moves.keys()):
            node.unexpanded_moves[action] = (
                (1 - epsilon) * node.unexpanded_moves[action] + epsilon * noise[idx]
            )
            idx += 1

    def search(self, game: BaseGame) -> MCTSNode:
        """Run MCTS for a single game. Returns the root node."""
        return self.search_batch([game])[0]

    def search_batch(self, games: list[BaseGame]) -> list[MCTSNode]:
        """Run MCTS for one or more games with batched inference.

        Uses virtual loss when num_parallel_leaves > 1.
        """
        n = len(games)
        k = self.mcts_config.num_parallel_leaves
        roots = [MCTSNode(g.copy()) for g in games]

        # Initial expansion — batch all roots
        predictions = self._predict_batch([r.game.get_state() for r in roots])
        for i in range(n):
            policy, value = predictions[i]
            self._set_priors(roots[i], policy, value)
            self._add_dirichlet_noise(
                roots[i],
                self.mcts_config.dirichlet_alpha,
                self.mcts_config.dirichlet_epsilon,
            )

        c_puct = self.mcts_config.c_puct
        num_sims = self.mcts_config.num_simulations
        sim = 0

        while sim < num_sims:
            chunk = min(k, num_sims - sim)

            to_predict_leaves: list[MCTSNode] = []
            to_predict_states: list = []

            for i in range(n):
                for _ in range(chunk):
                    leaf = self._select_leaf(roots[i], c_puct)

                    if leaf.game.is_over:
                        self._backpropagate(leaf, 1.0)
                    elif leaf.is_expanded and not leaf.has_candidates:
                        self._backpropagate(
                            leaf, leaf.q_value if leaf.visit_count > 0 else 0.0,
                        )
                    else:
                        self._apply_virtual_loss(leaf)
                        to_predict_leaves.append(leaf)
                        to_predict_states.append(leaf.game.get_state())

            if to_predict_states:
                predictions = self._predict_batch(to_predict_states)

                for j, leaf in enumerate(to_predict_leaves):
                    policy, value = predictions[j]
                    self._undo_virtual_loss(leaf)
                    value = self._set_priors(leaf, policy, value)
                    self._backpropagate(leaf, value)

            sim += chunk

        return roots

    @staticmethod
    def visit_counts_to_probs(
        visit_counts: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert 1D visit counts to a probability distribution."""
        if temperature == 0:
            best = torch.argmax(visit_counts)
            probs = torch.zeros_like(visit_counts)
            probs[best] = 1.0
            return probs

        counts = visit_counts ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            return visit_counts
        return counts / total.item()

    def get_action_probs(
        self, game: BaseGame, temperature: float = 1.0, verbose: bool = False,
    ) -> torch.Tensor:
        """Run MCTS and return 1D action probabilities."""
        root = self.search(game)
        logger.info("MCTS search done: %d visits, best_action=%d",
                     root.visit_count, root.best_move())
        return self.visit_counts_to_probs(
            root.visit_counts(game.action_space_size), temperature
        )
