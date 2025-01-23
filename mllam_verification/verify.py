"""Module with the main verification functions."""

from loguru import logger

from .config import Config
from .operations.dataset_manipulation import align_shapes, get_relevant_reference_data
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
    ds_reference_orig = load_xarray_dataset(config.inputs.datasets.reference.path)

    for prediction in config.inputs.datasets.predictions:
        ds_prediction = load_xarray_dataset(prediction.path)
        ds_reference = get_relevant_reference_data(ds_reference_orig, ds_prediction)
        ds_reference = align_shapes(ds_reference, ds_prediction)

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
