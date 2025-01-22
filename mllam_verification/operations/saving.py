"""Module to save xarray datasets to disk."""

from pathlib import Path
from typing import Literal, Optional

import xarray as xr
from loguru import logger


def save_xarray_dataset(
    dataset: xr.Dataset,
    path: Path,
    format_: Optional[Literal["netcdf", "zarr"]] = "netcdf",
    overwrite: bool = False,
) -> None:
    """
    Save an xarray dataset to a file at the given path in the specified format.

    Parameters:
        dataset (xr.Dataset): The xarray dataset to save.
        path (Path): The file path where the dataset will be saved.
        format_ (str), Optional (default: 'netcdf'): The format in which to save
            the dataset. Supported formats are 'netcdf' and 'zarr'.

    Raises:
        ValueError: If the specified format is not supported.
    """
    # Get path with suffix
    path = path.with_suffix(".nc" if format_ == "netcdf" else ".zarr")

    # If not explicitly requested, don't overwrite existing files
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File {path} already exists. Set `overwrite=True` to overwrite."
        )
    logger.info(f"Saving dataset to {path}")
    if format_ == "netcdf":
        dataset.to_netcdf(path)
    else:
        dataset.to_zarr(path)
