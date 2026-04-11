# Training

This guide covers how to configure and run the AlphaZero training pipeline.

## The train() function

```python
from pymcts.core.trainer import train
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig
from pathlib import Path

game_factory = lambda: MyGame()
arena_config = SinglePlayerArenaConfig(num_games=40)
self_play_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/self_play"))
eval_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/eval"))

train(
    game_factory=game_factory,           # creates fresh game instances
    net=my_net,                          # neural network to train
    mcts_config=mcts_config,             # MCTS settings
    training_config=training_config,     # training loop settings
    self_play_arena=self_play_arena,     # arena for self-play games
    eval_arena=eval_arena,               # arena for model evaluation
)
```

`game_factory` is a callable that returns a new game instance. This is called for every self-play game and every arena game.

The `self_play_arena` parameter controls how self-play games are collected. The `eval_arena` parameter controls how the new model is compared against the previous version. Both accept `SinglePlayerArena`, `MultiPlayerArena`, or `EloArena` instances. See below for details on arena configurations.

## Configuration

### MCTSConfig

Controls the MCTS search at self-play time.

```python
from pymcts.core.config import MCTSConfig

mcts_config = MCTSConfig(
    num_simulations=200,       # simulations per move (more = stronger)
    c_puct=1.5,                # exploration constant
    dirichlet_alpha=0.3,       # root noise for exploration
    dirichlet_epsilon=0.25,    # noise mixing ratio
    num_parallel_leaves=8,     # leaves per batch (GPU efficiency)
)
```

!!! tip "Simulations vs quality"
    - **25-50**: fast but weak, good for initial debugging
    - **200**: reasonable quality, good default
    - **800+**: strong play, slow training

### TrainingConfig

Controls the training loop.

```python
from pymcts.core.config import TrainingConfig

training_config = TrainingConfig(
    num_iterations=50,           # self-play → train → evaluate cycles
    num_self_play_games=100,     # games per iteration
    self_play_batch_size=16,     # concurrent self-play games
    batch_size=64,               # training batch size
    num_epochs=10,               # epochs per training step
    learning_rate=0.001,
    weight_decay=1e-4,
    replay_buffer_size=5,        # keep examples from last N iterations
)
```

### SinglePlayerArenaConfig

Controls head-to-head model comparison. The new model plays against the previous version and must exceed the win rate threshold to be accepted.

```python
from pymcts.arena.config import SinglePlayerArenaConfig

arena_config = SinglePlayerArenaConfig(
    num_games=40,       # games to play per evaluation
    threshold=0.55,     # win rate needed to accept new model
    swap_players=True,  # play both sides for fairness
)
```

### MultiPlayerArenaConfig

Controls multi-player arena evaluation with more than two competitors.

```python
from pymcts.arena.config import MultiPlayerArenaConfig

arena_config = MultiPlayerArenaConfig(
    num_games=40,       # games to play per evaluation
    threshold=0.55,     # win rate needed to accept new model
    swap_players=True,  # play both sides for fairness
)
```

### EloArenaConfig

Controls Elo pool-based evaluation. Instead of head-to-head comparison, the new model plays against a pool of reference players and must achieve a higher Elo rating than the current model.

```python
from pymcts.arena.config import EloArenaConfig

elo_arena = EloArenaConfig(
    games_per_matchup=40,       # games against each pool player
    elo_threshold=20.0,         # minimum Elo improvement to accept
    pool_growth_interval=5,     # add current model to pool every N iterations
    max_pool_size=None,         # cap on pool size (None = unlimited)
    swap_players=True,          # play both sides for fairness
    initial_pool=None,          # paths to seed players (None = start with RandomPlayer)
)
```

The pool starts with a `RandomPlayer` (Elo 1000) and grows organically. You can seed it with players from a previous training run:

```python
elo_arena = EloArenaConfig(
    initial_pool=[
        "trainings/run_previous/arena/iteration_010",
        "trainings/run_previous/arena/iteration_020",
    ],
)
```

!!! tip "When to use EloArena"
    Use `EloArena` with `EloArenaConfig` when head-to-head comparison is too noisy or when you want
    to measure improvement against a diverse field rather than just the previous model.

## Checkpoints

Training saves to `trainings/run_<timestamp>/` by default:

```
trainings/run_2026-03-24_143000/
├── run_config.json                # all configs (net, mcts, training, arena)
├── arena/                         # accepted/pool players (loadable with MCTSPlayer.load())
│   ├── iteration_001/             # accepted player from iteration 1
│   │   ├── model.pt
│   │   └── player.json
│   └── iteration_003/             # iteration 2 was rejected
│       ├── model.pt
│       └── player.json
├── iteration_001/
│   ├── pre_training.pt            # weights before training
│   ├── post_training.pt           # weights after training
│   ├── self_play_games.json       # games played
│   ├── eval_games.json            # arena games
│   ├── arena_results.json         # arena evaluation summary
│   └── iteration_data.json        # losses, win rates, etc.
├── iteration_002/
│   └── ...
└── elo_results.json               # Elo ratings (if elo_tracking=True)
```

When using `EloArenaConfig`, the `arena/` directory also contains pool players:

```
arena/
├── random/                        # always present
│   └── player.json
├── iteration_001/                 # accepted model
│   ├── model.pt
│   └── player.json
└── pool_iteration_005/            # pool snapshot
    ├── model.pt
    └── player.json
```

### Resuming training

Load the checkpoint and call `train()` again:

```python
from pymcts.arena import SinglePlayerArena
from pymcts.arena.config import SinglePlayerArenaConfig
from pathlib import Path

net = MyNet()
net.load_checkpoint("trainings/run_.../iteration_010/post_training.pt")

game_factory = lambda: MyGame()
arena_config = SinglePlayerArenaConfig(num_games=40)
self_play_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/self_play"))
eval_arena = SinglePlayerArena(arena_config, game_factory, arena_dir=Path("trainings/eval"))

train(
    game_factory=game_factory,
    net=net,
    mcts_config=mcts_config,
    training_config=TrainingConfig(num_iterations=20),  # 20 more
    self_play_arena=self_play_arena,
    eval_arena=eval_arena,
)
```

## Monitoring

Each iteration logs:

- **Self-play**: number of games, average game length
- **Training**: policy loss, value loss, total loss per epoch
- **Arena**: win rate, games played, accepted/rejected

The `iteration_data.json` files contain this data for analysis. See `notebooks/analysis.ipynb` for a plotting example.

## Hyperparameter guidance

### Starting point

For a new game, start with conservative settings and increase:

```python
MCTSConfig(num_simulations=50)
TrainingConfig(num_iterations=10, num_self_play_games=20, num_epochs=5)
SinglePlayerArenaConfig(num_games=20, threshold=0.55)
```

This runs fast and lets you verify the pipeline works.

### Scaling up

Once you see improvement:

- Increase `num_simulations` (200-800)
- Increase `num_self_play_games` (100-500)
- Increase `num_iterations` (50-200)
- Use `num_parallel_leaves` for GPU efficiency (4-16)

### Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Training loss doesn't decrease | Learning rate too low, or too few examples | Increase `num_self_play_games` or `learning_rate` |
| Arena never accepts (SinglePlayerArenaConfig) | Threshold too high, or not enough training | Lower `threshold` to 0.52, increase `num_epochs` |
| Arena always accepts (SinglePlayerArenaConfig) | Threshold too low | Raise `threshold` to 0.55-0.60 |
| Elo never improves (EloArenaConfig) | Threshold too high, or pool too strong | Lower `elo_threshold`, or start with a fresh pool |
| Training is slow | Too many MCTS simulations | Reduce `num_simulations`, increase `num_parallel_leaves` |
| Model doesn't improve after many iterations | Network too small, or too few simulations | Increase network size or `num_simulations` |

## Next step

Learn how to [evaluate and compare models](evaluation.md).
