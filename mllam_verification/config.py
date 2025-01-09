from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.types import ImportString


@pydantic_dataclass
class Config:
    schema_version: str
    inputs: "Inputs"
    methods: List["Method"]
    output: "Output"


@pydantic_dataclass
class Method:
    object: ImportString
    name: str
    include_persistence: bool


@pydantic_dataclass
class Dataset:
    path: Path


@pydantic_dataclass
class TimeRange:
    start: datetime
    end: datetime
    step: timedelta


@pydantic_dataclass
class Inputs:
    datasets: "Datasets"
    variables: List[str]
    coord_ranges: "CoordRanges"


@pydantic_dataclass
class Datasets:
    reference: Dataset
    predictions: List[Dataset]


@pydantic_dataclass
class CoordRanges:
    time: TimeRange


@pydantic_dataclass
class Output:
    path: Path
