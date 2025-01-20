from pathlib import Path
from typing import Literal, Optional

import xarray as xr


def save_xarray_dataset(
    dataset: xr.Dataset,
    path: Path,
    format_: Optional[Literal["netcdf", "zarr"]] = "netcdf",
) -> None:
    """
    Save an xarray dataset to a file at the given path in the specified format_.

    Parameters:
        dataset (xr.Dataset): The xarray dataset to save.
        path (Path): The file path where the dataset will be saved.
        format_ (str), Optional (default: 'netcdf'): The format in which to save
            the dataset. Supported formats are 'netcdf' and 'zarr'.

    Raises:
        ValueError: If the specified format is not supported.
    """
    if format_ == "netcdf":
        path = path.with_suffix(".nc")
        dataset.to_netcdf(path)
    elif format_ == "zarr":
        path = path.with_suffix(".zarr")
        dataset.to_zarr(path)
    else:
        raise ValueError(
            f"Unsupported format: {format_}. Supported formats are 'netcdf' "
            "and 'zarr'."
        )
