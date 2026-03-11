"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def unitest_path():
    return TEST_DATA_DIR / "unitest.png"


@pytest.fixture
def unitest2_path():
    return TEST_DATA_DIR / "unitest2.png"


@pytest.fixture
def bitest_path():
    return TEST_DATA_DIR / "bitest.png"


@pytest.fixture
def random_kymograph():
    """A small random kymograph for unit testing."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64), dtype=np.float32)


@pytest.fixture
def simple_skeleton():
    """A simple binary skeleton with known structure for testing morphology."""
    skeleton = np.zeros((32, 32), dtype=bool)
    # Vertical line at col 16
    skeleton[4:28, 16] = True
    # Branch at row 15
    skeleton[15, 16:22] = True
    return skeleton
