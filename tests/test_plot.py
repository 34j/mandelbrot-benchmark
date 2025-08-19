import matplotlib.pyplot as plt
import numpy as np

from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.plot import plot_mandelbrot


def test_plot():
    extent = (-2, 1, -1.5, 1.5)
    x, y = np.meshgrid(
        np.linspace(*extent[:2], 1000),
        np.linspace(*extent[2:], 1000),
    )
    c = x + 1j * y
    z = mandelbrot_numba(c)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_mandelbrot(z, ax=ax, extent=extent)
    fig.savefig("tests/test_plot.png")
