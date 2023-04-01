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
    useWandb: bool = True
    wandbAccount: Optional[Dict[str, str]] = None

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        self.resource = {
            stringify(key): stringify(value)
            for key, value in self.resource.items()
        }

    def _validate_canonical(self):
        super()._validate_canonical()
        
        if self.useWandb:
            try:
                import wandb
                if wandb.__version__ != '0.12.9':
                    print(f"WARNING: Version of wandb is {wandb.__version__} instead of 0.12.9. The codes might not work as expected.")
            except:
                raise ImportError('Please install wandb by "pip install wandb==0.12.9" to enable weight and bias.')
            
            if not wandb.login(anonymous='allow'):
                raise RuntimeError('wandb api key is not correctly configured. Please try "wandb login --anonymously" to generate an account.')
        
            if self.wandbAccount is None:
                # expose username and apikey to webui
                from wandb.sdk.lib import apikey
                from wandb.apis import internal
                from six.moves.urllib.parse import urlencode
                app_url = wandb.util.app_url(wandb.Settings().base_url)
                api = internal.Api()
                if api.settings().get("anonymous") != "true":
                    qs = ""
                else:
                    api_key = apikey.api_key(settings=wandb.Settings())
                    qs = "?" + urlencode({"apiKey": api_key})
                entity = wandb.setup(settings=wandb.Settings())._get_entity()
                self.wandbAccount = {
                    'app_url': app_url,
                    'entity': entity,
                    'qs': qs
                }