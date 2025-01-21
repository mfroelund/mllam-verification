"""Module for specifying the CLI interface for the mllam-verification package."""

from pathlib import Path
from typing import List

import click
import yaml
from loguru import logger

from .config import Config
from .plot import plot_verification_results
from .verify import verify_prediction


@click.group()
@click.option(
    "-c",
    "--config",
    type=click.File(),
    default=Path(__file__).parents[1] / "example.yaml",
    help="Path to the config file",
)
@click.pass_context
def cli(ctx: click.Context, config: click.File):
    """Main entry point for the CLI"""
    ctx.ensure_object(dict)
    logger.info(f"Running mllam-verification on config file {config.name}")
    ctx.obj["config"] = Config(**yaml.safe_load(config))


@cli.command()
@click.pass_context
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output datasets",
)
def verify(ctx: click.Context, overwrite: bool):
    """Execute verification of the prediction against the reference dataset."""
    config: Config = ctx.obj["config"]
    logger.info("Verifying the prediction against the reference dataset.")

    verify_prediction(config, overwrite)


@cli.command()
@click.pass_context
@click.argument(
    "datasets",
    nargs=-1,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--plottype",
    default="line",
    type=click.Choice(["line", "map"]),
    help="Type of plot to create",
)
@click.option(
    "--saveplots",
    is_flag=True,
    help="Whether to save plots or not",
)
def plot(ctx: click.Context, datasets: List[Path], plottype: str, saveplots: bool):
    """Execute plotting of error dataset."""
    config: Config = ctx.obj["config"]

    plot_verification_results(config, datasets, plottype, saveplots=saveplots)
