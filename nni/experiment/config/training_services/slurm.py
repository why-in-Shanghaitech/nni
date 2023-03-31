# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Author: Haoyi Wu

"""
Configuration for slurm training service.

"""

__all__ = ['SlurmConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict
import warnings

from typing_extensions import Literal

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .. import utils


def stringify(obj):
    return None if obj is None else str(obj).replace('_', '-')

@dataclass(init=False)
class SlurmConfig(TrainingServiceConfig):
    platform: Literal['slurm'] = 'slurm'
    resource: Dict[str, Union[str, None]]
    useSbatch: bool = False

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        self.resource = {
            stringify(key): stringify(value)
            for key, value in self.resource.items()
        }

    def _validate_canonical(self):
        super()._validate_canonical()
        # TODO: add more validations?