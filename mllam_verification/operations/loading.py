from pathlib import Path

import xarray as xr


def load_xarray_dataset(path: Path) -> xr.Dataset:
    """Load an xarray dataset from a file at the given path.

    Supported formats are 'netcdf', 'hdf5', and 'zarr'.

    Parameters:
        path (Path): The file path from where the dataset will be loaded.
        format_ (str), Optional: The format in which the dataset was saved.

    Returns:
        xr.Dataset: The loaded xarray dataset.

    Raises:
        ValueError: If the specified or detected format is not supported.
    """

    if path.suffix in [".nc", ".netcdf"]:
        return xr.open_dataset(path)
    if path.suffix in [".h5", ".hdf5"]:
        return xr.open_dataset(path, engine="h5netcdf")
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    raise ValueError(
        "Unsupported file format. Supported formats are 'netcdf', "
        "'hdf5', and 'zarr'."
    )
