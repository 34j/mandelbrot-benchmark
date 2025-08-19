from typing import Any

import numpy as np
import pytest
import taichi as ti
import torch
import warp as wp
from numpy.testing import assert_array_equal

from mandelbrot_benchmark.backends.array_api import mandelbrot_vectorized
from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi
from mandelbrot_benchmark.backends.warp import mandelbrot_warp

wp.init()


@pytest.fixture(params=["cpu", "cuda"])
def c(request: pytest.FixtureRequest) -> Any:
    device = request.param
    if device == "cuda":
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)
    rng = np.random.default_rng(0)
    shape = (100, 100)
    return torch.asarray(
        rng.random(shape) + 1j * rng.random(shape),
        dtype=torch.complex128,
        device=device,
    )


def test_numba(c: Any) -> None:
    mandelbrot_numba(c)


def test_taichi(c: Any) -> None:
    mandebrot_taichi(c)


def test_warp(c: Any) -> None:
    mandelbrot_warp(c)


def test_vectorized(c: Any) -> None:
    mandelbrot_vectorized(c)


def test_all_same(c: Any) -> None:
    """Test that all backends return the same result."""
    if c.device.type == "cuda":
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)
    wp.init()

    numba_result = mandelbrot_numba(c)
    taichi_result = mandebrot_taichi(c)
    warp_result = mandelbrot_warp(c)
    vectorized_result = mandelbrot_vectorized(c)

    assert_array_equal(numba_result, taichi_result)
    assert_array_equal(numba_result, warp_result)
    assert_array_equal(numba_result, vectorized_result)
