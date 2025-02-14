import xarray as xr
import mllam_verification as mlverif
import numpy as np

import pytest
    
@pytest.fixture
def ds_reference_fixture():
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds
    
def create_noisy_forecast_samples(ds_reference, n_timesteps=12, n_forecasts=4, noise_std=1.0, noise_bias=0.0):
    ds = ds_reference.copy()
    samples = []
    for i in range(0, n_forecasts):
        ds_sample = ds.isel(time=slice(i, i+n_timesteps))
        t0 = ds_sample.time[0].values
        for var in ds_sample.data_vars:
            ds_sample[var] = ds_sample[var] + np.random.normal(noise_bias, noise_std, ds_sample[var].shape)
        elapsed_forecast_duration = ds_sample.time - t0
        ds_sample.coords["start_time"] = t0
        ds_sample["elapsed_forecast_duration"] = elapsed_forecast_duration
        ds_sample = ds_sample.swap_dims({"time": "elapsed_forecast_duration"})
        samples.append(ds_sample)
        
    ds_predictions = xr.concat(samples, dim="start_time")
    return ds_predictions


def test_time_alignment(ds_reference_fixture):
    ds_prediction = create_noisy_forecast_samples(ds_reference_fixture)

    # check the round trip
    ds_prediction_by_utc = mlverif.time_alignment.elapsed_to_utc(ds_prediction)
    ds_predictions_reconstructed = mlverif.time_alignment.utc_to_elapsed(ds_prediction_by_utc)
    xr.testing.assert_equal(ds_predictions_reconstructed, ds_prediction)