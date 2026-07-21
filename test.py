"""Compare SHRINK and Sim-Piece on random points using the TerseTS Python bindings."""

import sys
import time
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the in-repository Python bindings importable without installation. The
# bindings locate the compiled library in zig-out/lib automatically.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "bindings" / "python"))

from tersets import compress, decompress, Method

POINT_COUNTS = [10_000, 100_000, 1_000_000]
ERROR_BOUND = 0.1

METHODS = [
    ("SHRINK", Method.Shrink, {"abs_error_bound": ERROR_BOUND, "lambda": 0.1}),
    ("Sim-Piece", Method.SimPiece, {"abs_error_bound": ERROR_BOUND}),
]

# Colors for the plot: muted gray for the raw signal, one fixed hue per method.
RAW_COLOR = "#8a8a8a"
METHOD_COLORS = {"SHRINK": "#2a78d6", "Sim-Piece": "#eb6834"}

PLOT_WINDOW = 500  # Number of points shown in the plot so segments stay visible.
PLOT_PATH = pathlib.Path(__file__).parent / "compression_comparison.png"


def generate_points(rng, count):
    """Generate a random piecewise-linear trend with small Gaussian noise.

    SHRINK and Sim-Piece both extract piecewise-linear segments, so a signal
    with linear trends and noise below the error bound suits them well.
    """
    knot_count = max(count // 500, 2)
    knots = np.linspace(0, count - 1, knot_count)
    knot_values = np.cumsum(rng.normal(0, 5.0, knot_count))
    trend = np.interp(np.arange(count), knots, knot_values)
    noise = rng.normal(0, 0.08, count)
    return trend + noise


def run_method(values, method, config):
    """Compress and decompress, returning (compressed_size, seconds, max_error, decompressed)."""
    start = time.perf_counter()
    compressed = compress(values, method, config)
    elapsed = time.perf_counter() - start

    decompressed = decompress(compressed)
    assert len(decompressed) == len(values), (
        f"expected {len(values)} values, got {len(decompressed)}"
    )
    max_error = np.max(np.abs(values - decompressed))
    assert max_error <= ERROR_BOUND, (
        f"max error {max_error} exceeds bound {ERROR_BOUND}"
    )
    return len(compressed), elapsed, max_error, decompressed


def plot_comparison(values, reconstructions):
    """Plot the raw points and each method's reconstruction in stacked panels."""
    window = slice(0, PLOT_WINDOW)
    x = np.arange(len(values))[window]

    fig, axes = plt.subplots(
        len(reconstructions), 1, figsize=(10, 6), sharex=True, sharey=True
    )
    fig.suptitle(
        f"Raw signal vs. reconstruction (first {PLOT_WINDOW:,} of {len(values):,} points, "
        f"error bound {ERROR_BOUND})",
        fontsize=11,
    )

    for axis, (name, decompressed) in zip(axes, reconstructions.items()):
        axis.plot(
            x,
            values[window],
            linestyle="none",
            marker=".",
            markersize=3,
            color=RAW_COLOR,
            alpha=0.6,
            label="Raw",
        )
        axis.plot(
            x,
            np.asarray(decompressed)[window],
            color=METHOD_COLORS[name],
            linewidth=2,
            label=name,
        )
        axis.legend(loc="upper right", frameon=False, fontsize=9)
        axis.grid(True, color="#e6e6e6", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8, color="#bbbbbb")

    axes[-1].set_xlabel("index", fontsize=9)
    fig.supylabel("value", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {PLOT_PATH}")


def main():
    rng = np.random.default_rng(42)

    print(f"Error bound: {ERROR_BOUND}\n")
    header = (
        f"{'points':>10} | {'method':<9} | {'raw bytes':>12} | {'compressed':>12} |"
        f" {'ratio':>7} | {'time (s)':>8} | {'max error':>10}"
    )
    print(header)
    print("-" * len(header))

    plot_data = None
    for count in POINT_COUNTS:
        values = generate_points(rng, count)
        reconstructions = {}

        for name, method, config in METHODS:
            size, elapsed, max_error, decompressed = run_method(values, method, config)
            reconstructions[name] = decompressed
            ratio = values.nbytes / size
            print(
                f"{count:>10,} | {name:<9} | {values.nbytes:>12,} | {size:>12,} |"
                f" {ratio:>6.2f}x | {elapsed:>8.4f} | {max_error:>10.6f}"
            )

        # Plot the smallest series, where individual segments are visible.
        if plot_data is None:
            plot_data = (values, reconstructions)

    print("\nAll sizes compressed and verified within the error bound.")
    plot_comparison(*plot_data)


if __name__ == "__main__":
    main()
