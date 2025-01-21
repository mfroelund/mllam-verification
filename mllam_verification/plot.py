"""Module with main plotting functions."""

from pathlib import Path
from typing import List

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
    for dataset in datasets:
        logger.info(f"Plotting {plottype} plot of dataset {dataset}")
        ds = load_xarray_dataset(dataset)

        if plottype == "line":
            fig = plot_error_timeline(ds)
            if saveplots:
                logger.info(f"Saving plot to {config.output.path}")
                fig.savefig(config.output.path / f"{dataset.name}_error_timeline.png")

        elif plottype == "map":
            plotter = plot_error_map(ds)
            for fig, time in plotter:
                if saveplots:
                    logger.info(f"Saving plot {time.values} to {config.output.path}")
                    fig.savefig(
                        config.output.path / f"{dataset.stem}_map_{time.values}.png"
                    )
                continue
