"""Configuration for AlphaZero-style Bridgit AI training."""

from dataclasses import dataclass


@dataclass
class Config:
    # Game
    board_size: int = 5

    # Neural network
    num_channels: int = 64
    num_res_blocks: int = 4

    # MCTS
    num_mcts_sims: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    temp_threshold: int = 8  # moves before switching to greedy

    # Training
    num_iterations: int = 50
    num_self_play_games: int = 50
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Arena
    num_arena_games: int = 40
    arena_threshold: float = 0.55

    # Replay buffer
    replay_buffer_size: int = 5  # number of iterations to keep

    # Paths
    checkpoint_dir: str = "checkpoints"

    @property
    def grid_size(self) -> int:
        return 2 * self.board_size + 1

    @property
    def action_size(self) -> int:
        return self.grid_size * self.grid_size
