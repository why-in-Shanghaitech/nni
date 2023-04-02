<div align="center">
<img src="docs/img/nni_slurm_logo.png" width="600"/>
</div>

<br/>

# NNI with SLURM and W&B

This is a patch for NNI that builds on version v2.10. Now a new training service is available: `slurm`!

## Who might be interested? <sup style="font-size: x-small;">[1]</sup>

- You use [NNI](https://github.com/microsoft/nni/) to run your machine learning experiments?
- Your ML experiments run on compute nodes without internet access (for example, using a batch system)?
- Your compute nodes and your head/login node (with internet) have access to a shared file system?

Then this package might be useful. For [Weights & Biases](https://wandb.ai/) users, you might be interested in this [alternative](https://github.com/klieret/wandb-offline-sync-hook).

<div align="center">

![SLURM](https://user-images.githubusercontent.com/43395692/229334079-cb31e6ef-3e31-4228-8370-9eb430a40365.png)

</div>

Currently, this patch only supports [SLURM](https://slurm.schedmd.com/overview.html), but it's quite simple to extend to other workload managers (e.g. [PBS](https://www.openpbs.org/)).

## Usage

This package is built upon NNI. If you are new to NNI, please refer to the [official documents](https://nni.readthedocs.io/en/stable).

If you are a professional NNI user. Simply change your training service to `slurm` (other types are not affected, change it back if needed):

```yaml
trainingService:
  platform: slurm
  resource:
    gres: gpu:NVIDIAGeForceRTX2080Ti:1  # request 1 2080Ti for each trial
    time: 1000                          # wall time for each trial
    partition: critical                 # request partition critical for resource allocation
  useSbatch: false
  useWandb: true
```

or if you use a python script:

```python
experiment = Experiment('slurm')
experiment.config.training_service.resource = {
    'gres': 'gpu:NVIDIAGeForceRTX2080Ti:1',  # request 1 2080Ti for each trial
    'time': 1000,                            # wall time for each trial
    'partition': 'critical'                  # request partition critical for resource allocation
}
experiment.config.training_service.useSbatch = False
experiment.config.training_service.useWandb = True
```

Then run `nnictl create --config config.yaml` or execute the python script on **the login node**. It will start the NNI server on the login node and submit slurm jobs.

### SLURM Training Service

There are only 4 parameters in `slurm` training service:
- `platform`: `str`. Must be `slurm`.
- `resource`: `Dict[str, str]`. Arguments to submit **a single job** to SLURM system. Do not add hyphens (`-` or `--`) at the front. Depending on the ways to submit the job (`srun`, `sbatch`), the options might be a little bit different. See SLURM docs ([srun](https://slurm.schedmd.com/srun.html), [sbatch](https://slurm.schedmd.com/sbatch.html)) for more details. Feel free to use numbers -- it will automatically convert to string when reading the config.
- `useSbatch`: `Optional[bool]`. Use `sbatch` to submit jobs instead of `srun`. The good side is: When the login node crashes accidentally, your job will not be affected. The bad side is: It has a buffer so that the output is delayed (metrics will not be affected). Default: `False`.
- `useWandb`: `Optional[bool]`. Summit the trail logs to W&B. If you have logged in W&B in your system before, it will use your account. Otherwise, it will automatically create an anonymous account for you. Default: `True`.

### Example

You can find a complete project example [here](https://github.com/whyNLP/nni-slurm/tree/dev-v2.10-slurm/examples/slurm). It is modified from the NNI official tutorial [HPO Quickstart with PyTorch](https://nni.readthedocs.io/zh/stable/tutorials/hpo_quickstart_pytorch/model.html).

## W&B Support

W&B provides more detailed experiment analysis (e.g. params importance, machine status, .etc).

<div align="center">

<img width="932" alt="image" src="https://user-images.githubusercontent.com/43395692/229332749-3bc5a557-f052-465f-8b0f-f4628616c1ff.png">

</div>

If you enabled `useWandb` (by default), then you are expected to see a new tab on the navigate bar:

<div align="center">

<img width="347" alt="image" src="https://user-images.githubusercontent.com/43395692/229325608-84a98df8-950b-4c3a-8ecc-86c1c0b9856b.png">

</div>

This is the web link to the W&B project of this experiment. After a trial succeeds, you could also see a link to this trial:

<div align="center">

<img width="824" alt="image" src="https://user-images.githubusercontent.com/43395692/229326602-c52d2e05-048f-4137-bd7b-a2c556c62bcd.png">

</div>

**Caution:** W&B link will only be valid if at least one of the trials succeeds. Only succeeded trials will be recorded by W&B.

By default, W&B link will be available for 7 days. If you want to keep the data for future analysis, you may claim the experiment to your account. If you have logged in W&B account on the login node, then the experiment will automatically save to your account.

## How to Install

Just download this patch wheel and do `pip install`:

```sh
wget https://github.com/whyNLP/nni-slurm/releases/download/v2.11/nni-2.11-py3-none-manylinux1_x86_64.whl
pip install nni-2.11-py3-none-manylinux1_x86_64.whl
```

## How to Uninstall

Simply do `pip uninstall nni` will completely remove this patch from your system.

## Trouble Shooting
### Error: /lib64/libstdc++.so.6: version `CXXABI_1.3.8' not found
It might because a low version of gcc. You might want to install gcc through [spack](https://github.com/spack/spack), which only requires a few lines of commands to install packages.
See microsoft#4914 for more details.

### Failed to establish a new connection: [Errno 111] Connection refused
This problem might have been fixed in the upcoming NNI 3.0. This patch uses a temporary fix: give it more retries.
See microsoft#3496 for more details.

### ValueError: ExperimentConfig: type of experiment_name (None) is not typing.Optional[str]
It has been fixed in this patch.
See microsoft#5468 for more details.

## Questions
### Can I run experiments using NNI without this patch?
It depends. There are some solutions:
1. Run in `local` mode with srun command. Potential problem: Login node cannot use tail-stream. Listen on file change will fail. The behaviour is that no metric could be updated. 
2. Run in `remote` mode with srun command, but connect to `localhost`. Potential problem: tmp folder does not sync to compute node. Also, you might not be able to visit login node on the compute node.

See microsoft#1939, microsoft#3717 for more details.

### Will you create a pull request to NNI?
I have no plan to create a pull request. This patch is not fully tested. The code style is not fully consistent with NNI requirements. I develop this patch for personal use only.

## Reference
- [1] [Wandb Offline Sync Hook](https://github.com/klieret/wandb-offline-sync-hook)