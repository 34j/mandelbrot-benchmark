from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_mandelbrot(
    z: Any,
    /,
    *,
    ax: Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """Plot the Mandelbrot set."""
    ax = ax or plt.gca()
    im = ax.imshow(
        z, cmap="jet", interpolation="bilinear", extent=extent, origin="lower"
    )
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, orientation="vertical", label="Iterations")
