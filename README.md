# mandelbrot-benchmark

<p align="center">
  <a href="https://github.com/34j/mandelbrot-benchmark/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/mandelbrot-benchmark/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://mandelbrot-benchmark.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/mandelbrot-benchmark.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
</p>
<p align="center">
  <a href="https://codecov.io/gh/34j/mandelbrot-benchmark">
    <img src="https://img.shields.io/codecov/c/github/34j/mandelbrot-benchmark.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/mandelbrot-benchmark/">
    <img src="https://img.shields.io/pypi/v/mandelbrot-benchmark.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/mandelbrot-benchmark.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/mandelbrot-benchmark.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://mandelbrot-benchmark.readthedocs.io" target="_blank">https://mandelbrot-benchmark.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/mandelbrot-benchmark" target="_blank">https://github.com/34j/mandelbrot-benchmark </a>

---

Benchmark Numba, Taichi, Warp, CuPy Kernel, Triton using Mandelbrot set

## Results

- 200 iterations max
- $c \in [-2, 1] \times [-1.5, 1.5]$
- JAX version is quite hacky ("vectorized")
- `JAX` version is omitted because it makes the benchmark unstable (see previous release for the comparison)
- AMD Ryzen 9 3950X + NVIDIA GeForce RTX 4070 SUPER

![Results](https://raw.githubusercontent.com/34j/mandelbrot-benchmark/main/results.png)

- The results are almost identical when `device = cuda`
- The computation time for `Numba` is quite unstable

![Mandelbrot Set](https://raw.githubusercontent.com/34j/mandelbrot-benchmark/main/test_plot.png)

## Some notes

- In `Numba`, the type of variables needs to be explicitly specified to use `float32` instead of `float64` (default), while is not the case for `Taichi`, `Warp`.

## Installation

Install this via pip (or your favourite package manager):

```shell
pipx install mandelbrot-benchmark
```

## Usage

Run the benchmark and plot the results:

```shell
mandelbrot-benchmark benchmark
mandelbrot-benchmark plot
```

## Alternatives

- [YanagiAyame/python-mandelbrot-benchmark: Which is better, Numba, Taichi, Warp or JAX?](https://github.com/YanagiAyame/python-mandelbrot-benchmark)
- [mandelbrot-on-all-accelerators.ipynb](https://gist.github.com/jpivarski/da343abd8024834ee8c5aaba691aafc7)
