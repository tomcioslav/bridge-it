# Bridgit Game

A Python implementation of the Bridgit connection game with support for player vs player gameplay and future AI training capabilities.

## About Bridgit

Bridgit is a two-player connection game played on an n×n grid. Players take turns placing their lines in empty grid cells:

- **Player 1 (Horizontal - Green)**: Places horizontal lines and tries to connect the left edge to the right edge
- **Player 2 (Vertical - Red)**: Places vertical lines and tries to connect the top edge to the bottom edge

Players must create a continuous path through adjacent cells (up, down, left, right) to win. The first player to complete their connection wins!

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.10+
- uv (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd bridge-it

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install numpy pygame

# Or install the package with all dependencies
uv pip install -e ".[all]"
```

## Usage

Play with default board size (5×5):
```bash
python play.py
```

Play with custom board size:
```bash
python play.py <n>

# Example: 7×7 board
python play.py 7
```

### How to Play

The game features a beautiful graphical interface with:
- **Square grid**: An n×n grid where each cell can hold one line
- **Click on cells** to place your line (green horizontal or red vertical)
- **Hover highlight** shows which cell you're about to select
- **Visual player indicators** show whose turn it is
- **Win screen** celebrates when someone completes their connection from side to side

### Controls

- **Mouse**: Click on grid cells to place your line
- **R key**: Restart the game
- **Q key**: Quit

## Project Structure

```
bridge-it/
├── bridgit/           # Core game package
│   ├── __init__.py    # Package initialization
│   └── game.py        # Game logic and board representation
├── ai/                # AI components (future)
├── tests/             # Unit tests (future)
├── play.py            # GUI interface for player vs player
├── README.md          # This file
├── LICENSE            # MIT License
└── pyproject.toml     # Project configuration
```

## Game Rules

### Board

- The board is an n×n grid (default: 5×5)
- Each grid cell can contain one line (horizontal or vertical)
- Grid representation:
  - 0: empty cell
  - -1: Player 1's horizontal line (green)
  - 1: Player 2's vertical line (red)

### Gameplay

1. **Player 1 (HORIZONTAL - Green)** goes first
   - Places horizontal lines in empty cells
   - Wins by creating a connected path from the left edge to the right edge

2. **Player 2 (VERTICAL - Red)** goes second
   - Places vertical lines in empty cells
   - Wins by creating a connected path from the top edge to the bottom edge

3. **Adjacency**: Cells are connected if they share an edge (up, down, left, right)
4. **Turns**: Players alternate placing one line per turn
5. **Winning**: First player to create an unbroken path across the board wins

### Strategy Tips

- Block your opponent by cutting off their potential paths
- Create multiple paths to increase your chances of connecting
- Control the center for maximum flexibility
- Think ahead - ensure your path can actually reach the opposite side
- Adjacent cells (orthogonal connections only) form your path

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows standard Python conventions:
- PEP 8 style guide
- Type hints where applicable
- Docstrings for all public functions

## Future Features

- [ ] GUI interface with pygame
- [ ] AI opponent (random, minimax, MCTS)
- [ ] Neural network training with PyTorch
- [ ] Reinforcement learning agents
- [ ] Game state serialization
- [ ] Move history and replay
- [ ] Multiple board size presets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Bridgit was invented by David Gale in the 1960s and is part of the family of connection games.
