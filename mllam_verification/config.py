"""Module for parsing and validating a config file."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.types import ImportString


@pydantic_dataclass
class Config:
    """Class for validating the Config section of the config file"""

    schema_version: str
    inputs: "Inputs"
    methods: List["Method"]
    output: "Output"


@pydantic_dataclass
class Method:
    """Class for validating the Method section of the config file"""

    object: ImportString
    name: str
    include_persistence: bool


@pydantic_dataclass
class Dataset:
    """Class for validating the Dataset section of the config file"""

    path: Path


@pydantic_dataclass
class TimeRange:
    """Class for validating the TimeRange section of the config file"""

    start: datetime
    end: datetime
    step: timedelta


@pydantic_dataclass
class Inputs:
    """Class for validating the Inputs section of the config file"""

    datasets: "Datasets"
    variables: List[str]
    coord_ranges: "CoordRanges"


@pydantic_dataclass
class Datasets:
    """Class for validating the Datasets section of the config file"""

    reference: Dataset
    predictions: List[Dataset]


@pydantic_dataclass
class CoordRanges:
    """Class for validating the CoordRanges section of the config file"""

    time: TimeRange


@pydantic_dataclass
class Output:
    """Class for validating the Output section of the config file"""

    path: Path
