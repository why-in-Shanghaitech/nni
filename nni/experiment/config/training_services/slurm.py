# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Author: Haoyi Wu

"""
Configuration for slurm training service.

"""

__all__ = ['SlurmConfig', 'SlurmResourceConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import warnings

from typing_extensions import Literal

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .. import utils

@dataclass(init=False)
class SlurmResourceConfig(ConfigBase):
    gres: Optional[str] = None
    time: Optional[str] = None
    time_min: Optional[str] = None
    partition: Optional[str] = None
    exclude: Optional[str] = None

    def _validate_canonical(self):
        super()._validate_canonical()
        # TODO: add more validations?


@dataclass(init=False)
class SlurmConfig(TrainingServiceConfig):
    platform: Literal['slurm'] = 'slurm'
    resource: SlurmResourceConfig
    useSbatch: bool = False

    def _validate_canonical(self):
        super()._validate_canonical()
        # TODO: add more validations?