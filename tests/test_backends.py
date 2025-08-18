import numpy as np
import pytest

from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi


@pytest.fixture
def c():
    rng = np.random.default_rng(0)
    shape = (100, 100)
    return rng.random(shape) + 1j * rng.random(shape)


def test_numba(c) -> None:
    mandelbrot_numba(c)


def test_taichi(c) -> None:
    mandebrot_taichi(c)
