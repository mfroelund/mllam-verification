"""Module with the main verification functions."""

from loguru import logger

from .config import Config
from .operations.loading import load_xarray_dataset
from .operations.saving import save_xarray_dataset


def verify_prediction(config: Config, overwrite: bool = False):
    """Verify the prediction against the reference dataset.

    Parameters:
    -----------
    config: Config
        The configuration object
    """
    logger.info(config)
    ds_reference = load_xarray_dataset(config.inputs.datasets.reference.path)

    for ds_prediction in config.inputs.datasets.predictions:
        ds_prediction = load_xarray_dataset(ds_prediction.path)
        for method in config.methods:
            error = method.object(
                ds_reference, ds_prediction, method.include_persistence
            )

            save_xarray_dataset(
                error,
                config.output.path / method.name,
                format_="zarr",
                overwrite=overwrite,
            )
