"""Monte Carlo Tree Search with neural network guidance."""

import math

import numpy as np

from bridgit.game import Bridgit
from bridgit.ai.config import Config
from bridgit.ai.game_wrapper import GameWrapper
from bridgit.ai.neural_net import NeuralNetWrapper


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ["game", "parent", "action", "children", "visit_count",
                 "value_sum", "prior", "is_expanded"]

    def __init__(self, game: Bridgit, parent: "MCTSNode | None" = None,
                 action: int | None = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.action = action  # action that led to this node
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """PUCT selection score."""
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))


class MCTS:
    """Monte Carlo Tree Search guided by a neural network."""

    def __init__(self, neural_net: NeuralNetWrapper, config: Config):
        self.neural_net = neural_net
        self.config = config
        self.game_wrapper = GameWrapper(config.board_size)

    def search(self, game: Bridgit) -> np.ndarray:
        """Run MCTS simulations and return move probabilities.

        Returns:
            policy: np.ndarray of shape (action_size,) — visit count distribution
        """
        root = MCTSNode(game.copy())

        # Expand root
        self._expand(root)

        # Add Dirichlet noise to root priors
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.num_mcts_sims):
            node = root

            # SELECT: walk tree until reaching an unexpanded or terminal node
            while node.is_expanded and node.children:
                node = node.best_child(self.config.c_puct)

            # If terminal, backprop the game result
            if node.game.game_over:
                # Value from the perspective of the node's parent's player
                # (the player who made the move leading to this terminal state)
                result = self.game_wrapper.get_game_result(node.game, node.game.current_player)
                # game_over means the *previous* player won, so current player lost
                value = -1.0
                self._backpropagate(node, value)
                continue

            # EXPAND & EVALUATE
            value = self._expand(node)

            # BACKPROPAGATE
            self._backpropagate(node, value)

        # Build policy from visit counts
        action_size = self.config.action_size
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        return visits

    def _expand(self, node: MCTSNode) -> float:
        """Expand a node using the neural network.

        Returns:
            value: float — neural network's value estimate for this position
                   from the perspective of the current player
        """
        # Terminal nodes: don't expand, return game result
        if node.game.game_over:
            node.is_expanded = True
            # The previous player won, so the current player (whose turn it would be) lost
            return -1.0

        state_tensor = self.game_wrapper.get_state_tensor(node.game)
        policy, value = self.neural_net.predict(state_tensor)

        valid_mask = self.game_wrapper.get_valid_moves_mask(node.game)
        policy = policy * valid_mask

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # If no valid moves have probability, use uniform over valid moves
            total = valid_mask.sum()
            if total > 0:
                policy = valid_mask / total
            else:
                # No valid moves at all (shouldn't happen if game_over check above works)
                node.is_expanded = True
                return value

        # Create child nodes for each valid action
        for action in range(self.config.action_size):
            if valid_mask[action] > 0:
                child_game = self.game_wrapper.get_next_state(node.game, action)
                child = MCTSNode(child_game, parent=node, action=action, prior=policy[action])
                node.children[action] = child

        node.is_expanded = True
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree, flipping sign at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip for opponent
            node = node.parent

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node priors for exploration."""
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            node.children[action].prior = (
                (1 - eps) * node.children[action].prior + eps * noise[i]
            )

    def get_action_probs(self, game: Bridgit, temperature: float = 1.0) -> np.ndarray:
        """Run MCTS and return action probabilities.

        Args:
            game: current game state
            temperature: controls exploration.
                1.0 = proportional to visit counts (exploratory)
                0.0 = argmax (greedy)

        Returns:
            probs: np.ndarray of shape (action_size,) summing to 1
        """
        visit_counts = self.search(game)

        if temperature == 0:
            # Greedy: pick the most visited action
            best = np.argmax(visit_counts)
            probs = np.zeros_like(visit_counts)
            probs[best] = 1.0
            return probs

        # Temperature-scaled distribution
        counts = visit_counts ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            return visit_counts  # fallback
        return counts / total
