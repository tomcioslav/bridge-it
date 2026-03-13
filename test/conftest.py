"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path
from typing import Any


def load_json(filename: str, test_dir: str | None = None) -> list[dict[str, Any]]:
    """Load test cases from a JSON file.

    Args:
        filename: Name of the JSON file (e.g., 'test_check_winner.json')
        test_dir: Optional test directory name (e.g., 'test_game').
                  If None, looks in the same directory as the calling test file.

    Returns:
        List of test case dictionaries
    """
    test_root = Path(__file__).parent

    if test_dir:
        data_path = test_root / test_dir / "data" / filename
    else:
        data_path = test_root / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    with open(data_path, "r") as f:
        return json.load(f)
