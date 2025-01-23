"""Module with main plotting functions."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import xarray as xr
from loguru import logger

from .config import Config
from .operations.loading import load_xarray_dataset
from .operations.plot import plot_error_map, plot_error_timeline


def plot_verification_results(
    config: Config, datasets: List[Path], plottype: str, saveplots: bool = False
):
    """Plot verification results for specified variables.

    If no variables are specified, all variables will be plotted.

    Parameters:
    -----------
    config: Config
        The configuration object
    datasets: List[Path]
        List of paths to the datasets to plot
    plottype: str
        The type of plot to create. Either 'line' or 'map'
    saveplots: bool, optional (default: False)
        Whether to save the plots to disk
    """
    # Type hint variables
    state_feature: xr.DataArray

    # Produce plots for each dataset
    for dataset in datasets:
        logger.info(f"Plotting {plottype} plot of dataset {dataset}")
        ds = load_xarray_dataset(dataset)

        if plottype == "line":
            fig = plot_error_timeline(ds)
            if saveplots:
                logger.info(f"Saving plot to {config.output.path}")
                fig.savefig(config.output.path / f"{dataset.name}_error_timeline.png")

        elif plottype == "map":
            # Produce plots for each state feature
            for state_feature in ds["state_feature"]:
                logger.info(f"Plotting of feature {state_feature.values}")

                plotter = plot_error_map(ds.sel(state_feature=state_feature))
                # Iterate over the plotter to produce the plots for each
                # elapsed_forecast_duration
                for fig, elapsed_forecast_duration in plotter:
                    if saveplots:
                        output_path: Path = (
                            config.output.path
                            / state_feature.values
                            / (
                                f"{dataset.stem}_map_"
                                f"{elapsed_forecast_duration.values}.png"
                            )
                        )
                        logger.info(
                            "Saving elapsed_forecast_duration "
                            f"{elapsed_forecast_duration.values} to {output_path}"
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(output_path)
                    plt.close()
