from typing import Any

import numpy as np
import pytest

from mandelbrot_benchmark.backends.numba import mandelbrot_numba, mandelbrot_numba_cuda
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi


@pytest.fixture
def c() -> Any:
    rng = np.random.default_rng(0)
    shape = (100, 100)
    return rng.random(shape) + 1j * rng.random(shape)


def test_numba(c: Any) -> None:
    mandelbrot_numba(c)


def test_numba_cuda(c: Any) -> None:
    mandelbrot_numba_cuda(c)


def test_taichi(c: Any) -> None:
    mandebrot_taichi(c)
