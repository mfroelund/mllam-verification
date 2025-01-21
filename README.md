# mllam-verification
Verification of neural-lam, e.g. performance of the model relative to a reference, persistence, etc.

## Running the Tool from CLI

To run the tool from CLI, use the following command:

```
Usage: python -m mllam_verification [OPTIONS] COMMAND [ARGS]...

  Main entry point for the CLI

Options:
  -c, --config FILENAME  Path to the config file
  --help                 Show this message and exit.

Commands:
  plot    Execute plotting of error dataset.
  verify  Execute verification of the prediction against the reference...

```

E.g. to verify a prediction against a reference dataset, use the following command:

```bash
mllam-verification verify --config /path/to/config.yaml
```

If no `--config` option is provided, the tool will use the example config file located at `mllam_verification/example.yaml`.

### Available Commands, Options and Arguments
#### `verify`
```
Usage: python -m mllam_verification verify [OPTIONS]

  Execute verification of the prediction against the reference dataset.

Options:
  --overwrite  Overwrite existing output datasets
  --help       Show this message and exit.
```

#### `plot`
```
Usage: python -m mllam_verification plot [OPTIONS] [DATASETS]...

  Execute plotting of error dataset.

Options:
  --plottype [line|map]  Type of plot to create
  --saveplots            Whether to save plots or not
  --help                 Show this message and exit.
```
where `DATASETS...` is a list of paths to the datasets to plot.

## Using the Tool from Python Script or Jupyter Notebook
You can also use the tool from a Python script or Jupyter Notebook. Here is an example:

```python
from mllam_verification.config import Config
from mllam_verification.verify import verify_prediction
from mllam_verification.plot import plot_verification_results
from pathlib import Path
import yaml

# Load config
with open("/path/to/config.yaml", "r") as file:
    config = Config(**yaml.safe_load(file))

# Verify prediction
verify_prediction(config)

# Plot results
plot_verification_results(config, datasets=[Path("/path/to/dataset")], plottype="line", saveplots=True)
```

## Adjusting the YAML Config File
The YAML config file should follow this structure:

```yaml
schema_version: v0.1.0

inputs:
  datasets:
    reference:
      path: /path/to/reference_dataset.zarr
    predictions:
      - path: /path/to/prediction_dataset.zarr
  variables:
    - variable1
    - variable2
  coord_ranges:
    time:
      start: YYYY-MM-DDTHH:MM
      end: YYYY-MM-DDTHH:MM
      step: PTnH

methods:
  - object: mllam_verification.operations.statistics.calculate_global_error
    name: global_error
    include_persistence: true
  - object: mllam_verification.operations.statistics.calculate_error_per_gridpoint
    name: error_per_gridpoint
    include_persistence: true

output:
  path: /path/to/output
```

The `inputs` section should contain the paths to the reference and prediction datasets to compare and the time range to consider.

The `methods` section should contain the operations to perform on the datasets. The `methods.object` setting should be a python module string to an existing python object.

The `output` section should contain the path to the output directory. This directory is used when saving datasets and plots.

## Expected Structure of the Datasets
#### Reference Dataset
The reference dataset should have coordinates [time, grid_index] (for 1D plots) or [time, x, y] (for 2D plots).

#### Prediction Dataset
The prediction dataset should have coordinates [analysis_time, elapsed_forecast_duration, grid_index] (for 1D plots) or [analysis_time, elapsed_forecast_duration, x, y] (for 2D plots). 

