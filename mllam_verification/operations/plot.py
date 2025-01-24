"""Module with various plotting functions."""

import math
from typing import Generator, Hashable, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_error_timeline(ds_error: xr.Dataset) -> plt.Figure:
    """Plot the error vs time from the error dataset.

    Parameters:
    -----------
    ds_error: xr.Dataset
        The dataset containing the error with coordinates [time]
    """
    # Prepare figure and axes
    fig, ax = plt.subplots()

    # Create a cycler for plot styles in order to have same color for each feature
    # but different line styles for each variable
    nfeatures = len(ds_error["state_feature"])
    nvars = len(ds_error.data_vars)
    colors = plt.cm.tab10.colors
    repeated_colors = np.tile(colors[:nfeatures], nvars).reshape((nvars * nfeatures, 3))
    repeated_lines = ["-"] * nfeatures + ["--"] * nfeatures
    plot_style_cycler = plt.cycler(color=repeated_colors, linestyle=repeated_lines)
    ax.set_prop_cycle(plot_style_cycler)

    # Plot each variable in the dataset for every feature
    for var in ds_error.data_vars:
        ds_error[var].plot.line(
            ax=ax,
            x="elapsed_forecast_duration",
            hue="state_feature",
        )

    # Create custom legends for features and variables
    feature_labels = list(ds_error["state_feature"].values)
    feature_handles = [
        plt.Line2D([0], [0], color=colors[i], marker="s", linestyle="")
        for i in range(len(feature_labels))
    ]
    variable_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-"),
        plt.Line2D([0], [0], color="black", linestyle="--"),
    ]
    variable_labels = [
        f"{var} [cell_methods: {ds_error[var].attrs['cell_methods']}]"
        for var in ds_error.data_vars
    ]
    legend_features = ax.legend(
        feature_handles,
        feature_labels,
        prop={"size": 8},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(feature_labels),
        frameon=False,
    )
    ax.add_artist(legend_features)
    ax.legend(
        variable_handles,
        variable_labels,
        prop={"size": 8},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
        frameon=False,
    )

    plt.tight_layout()
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Error vs Time")
    plt.grid(True)

    return fig


def plot_error_map(
    ds_error: xr.Dataset,
) -> Generator[Tuple[plt.Figure, xr.DataArray], None, None]:
    """Plot the error map from the error dataset.

    Parameters:
    -----------
    ds_error: xr.Dataset
        The dataset containing the error with coordinates [x, y]
    """
    # Calculate the number of rows and columns for the grid
    num_vars = len(ds_error.data_vars)
    nrows = math.floor(math.sqrt(num_vars))  # Number of columns for the grid
    ncols = math.ceil(num_vars / nrows)  # Calculate the number of rows needed

    # Get max and min values for the colorbars
    ds_max = ds_error.max()
    ds_min = ds_error.min()

    # Loop over elapsed_forecast_duration and produce one plot per elapsed_forecast_duration
    var: Hashable
    elapsed_forecast_duration: xr.DataArray
    for elapsed_forecast_duration in ds_error["elapsed_forecast_duration"]:
        # Prepare the figure
        fig = plt.figure(figsize=(10, 5 * nrows), constrained_layout=True)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        plt.suptitle("Error Map")

        # Plot each variable in the dataset
        for i, var in enumerate(ds_error.data_vars):
            row = i // ncols
            col = i % ncols
            axes = fig.add_subplot(gs[row, col])
            ds_error.sel(
                elapsed_forecast_duration=elapsed_forecast_duration,
            )[var].plot.pcolormesh(
                x="x",
                y="y",
                cmap="bwr",
                ax=axes,
                vmin=ds_min[var],
                vmax=ds_max[var],
            )
            axes.set_title(
                f"{var} at elapsed_forecast_duration {elapsed_forecast_duration.values}"
            )

        yield fig, elapsed_forecast_duration
