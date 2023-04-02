# Example for slurm

*This is an example modified from [HPO quickstart pytorch](https://nni.readthedocs.io/zh/stable/tutorials/hpo_quickstart_pytorch/main.html)*.

All commands run on the login node.

## Use python scripts

**Step 1.** Prepare the environment and downlaod the dataset:

```sh
pip install torch torchvision
python download.py
```

**Step 2.** Change the slurm settings in line 74-82, `main.py`:

```python
## slurm
experiment = Experiment('slurm')
experiment.config.training_service.resource = {
    'gres': 'gpu:NVIDIAGeForceRTX2080Ti:1',  # request 1 2080Ti for each trial
    'time': 1000,                            # wall time for each trial
    'partition': 'critical'                  # request partition critical for resource allocation
}
experiment.config.training_service.useSbatch = False
experiment.config.training_service.useWandb = True
```

You should change the resource requested for each trial based on your cluster requirements and your own needs.

**Step 3.** Start the experiment:

```sh
python main.py
```

It will start the NNI server.

## Use command line tools

**Step 1.** Prepare the environment and downlaod the dataset:

```sh
pip install torch torchvision
python download.py
```

**Step 2.** Change the slurm settings in line 13-20, `config.yaml`:

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

You should change the resource requested for each trial based on your cluster requirements and your own needs.

**Step 3.** Start the experiment:

```sh
nnictl create --config config.yaml
```

It will start the NNI server.

## SLURM Training Service

There are only 4 parameters in `slurm` training service:
- `platform`: `str`. Must be `slurm`.
- `resource`: `Dict[str, str]`. Arguments to submit **a single job** to SLURM system. Do not add hyphens (`-` or `--`) at the front. Depending on the ways to submit the job (`srun`, `sbatch`), the options might be a little bit different. See SLURM docs ([srun](https://slurm.schedmd.com/srun.html), [sbatch](https://slurm.schedmd.com/sbatch.html)) for more details. Feel free to use numbers -- it will automatically convert to string when reading the config.
- `useSbatch`: `Optional[bool]`. Use `sbatch` to submit jobs instead of `srun`. The good side is: When the login node crashes accidentally, your job will not be affected. The bad side is: It has a buffer so that the output is delayed (metrics will not be affected). Default: `False`.
- `useWandb`: `Optional[bool]`. Summit the trail logs to W&B. If you have logged in W&B in your system before, it will use your account. Otherwise, it will automatically create an anonymous account for you. Default: `True`.

## Questions

### Why use `.yaml` instead of `.yaml`?

The official site of YAML has made an declaration [here](https://yaml.org/faq.html).