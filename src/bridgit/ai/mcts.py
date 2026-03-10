"""Monte Carlo Tree Search with neural network guidance."""

import math

import numpy as np
import torch

from bridgit.game import Bridgit
from bridgit.ai.config import Config
from bridgit.ai.game_wrapper import GameWrapper
from bridgit.ai.neural_net import NetWrapper


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ["game", "parent", "action", "children", "visit_count",
                 "value_sum", "prior", "is_expanded"]

    def __init__(self, game: Bridgit, parent: "MCTSNode | None" = None,
                 action: int | None = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.action = action
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

    def __init__(self, net_wrapper: NetWrapper, config: Config):
        self.net_wrapper = net_wrapper
        self.config = config
        self.game_wrapper = GameWrapper(config.board_size)

    def _predict(self, game: Bridgit) -> tuple[np.ndarray, float]:
        """Run neural net on game state.

        Returns:
            policy: np.ndarray of shape (action_size,) — probabilities
            value: float — position evaluation for current player
        """
        model = self.net_wrapper.model
        device = self.net_wrapper.device

        model.eval()
        tensor = self.game_wrapper.get_state_tensor(game).unsqueeze(0).to(device)

        with torch.no_grad():
            log_policy, value = model(tensor)

        # log_policy: (1, g, g) → flatten to (g*g,)
        policy = torch.exp(log_policy[0]).cpu().numpy().flatten()
        val = value[0].item()

        return policy, val

    def search(self, game: Bridgit) -> np.ndarray:
        """Run MCTS simulations and return visit counts.

        Returns:
            visits: np.ndarray of shape (action_size,)
        """
        root = MCTSNode(game.copy())
        self._expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.config.num_mcts_sims):
            node = root

            # SELECT
            while node.is_expanded and node.children:
                node = node.best_child(self.config.c_puct)

            # TERMINAL — current_player is the winner (turn doesn't switch on win)
            if node.game.game_over:
                self._backpropagate(node, 1.0)
                continue

            # EXPAND & EVALUATE
            value = self._expand(node)

            # BACKPROPAGATE
            self._backpropagate(node, value)

        # Collect visit counts
        visits = np.zeros(self.config.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        return visits

    def _expand(self, node: MCTSNode) -> float:
        """Expand node using neural network. Returns value estimate."""
        if node.game.game_over:
            node.is_expanded = True
            return 1.0  # current_player is the winner

        policy, value = self._predict(node.game)
        valid_mask = self.game_wrapper.get_valid_moves_mask(node.game)

        # Mask and renormalize policy
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

        # Create children for valid actions
        for action in np.nonzero(valid_mask)[0]:
            child_game = self.game_wrapper.get_next_state(node.game, int(action))
            child = MCTSNode(child_game, parent=node, action=int(action), prior=policy[action])
            node.children[int(action)] = child

        node.is_expanded = True
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value, flipping sign only when the player changes.

        With the 1-2-2 turn structure, consecutive nodes may belong to the
        same player. We only negate the value when crossing a player boundary.
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if (node.parent is not None
                    and node.game.current_player != node.parent.game.current_player):
                value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
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
            temperature: 1.0 = proportional to visits, 0.0 = greedy

        Returns:
            probs: np.ndarray of shape (action_size,) summing to 1
        """
        visit_counts = self.search(game)

        if temperature == 0:
            best = np.argmax(visit_counts)
            probs = np.zeros_like(visit_counts)
            probs[best] = 1.0
            return probs

        counts = visit_counts ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            return visit_counts
        return counts / total
