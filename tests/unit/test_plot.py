from mllam_verification.operations.plot import plot_error_map, plot_error_timeline
from mllam_verification.verify import (
    calculate_error_per_gridpoint,
    calculate_global_error,
)


def test_plot_error_timeline(ds_reference_1D, ds_prediction_1D):
    ds_error = calculate_global_error(
        ds_reference_1D, ds_prediction_1D, include_persistence=True
    )
    plot_error_timeline(ds_error)


def test_plot_error_per_gridpoint(ds_reference_2D, ds_prediction_2D):
    ds_error = calculate_error_per_gridpoint(
        ds_reference_2D, ds_prediction_2D, include_persistence=True
    )
    plot_error_map(ds_error)
