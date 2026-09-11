## LCFM Experiments

Special thanks to [CleanRL](https://github.com/vwxyzjn/cleanrl) for their PPO-Transformer-XL implementation, which we adapted for this work.

### Requirements

- Linux (the environment was tested on `linux-64`)
- Conda (Miniconda, Miniforge, or Anaconda)
- An NVIDIA GPU with a recent driver (`nvidia-smi` should report CUDA Version 12.6 or higher)

PyTorch is pinned to `torch==2.12.0+cu126`, which supports GPUs from compute capability 5.0 to 9.0 (roughly GTX 900 series through RTX 40 series / H100). The CUDA 12.6 build is used deliberately so that older Pascal GPUs (e.g. GTX 10 series) keep working. Do not upgrade to a `cu128` or newer build unless you know your GPU is supported. If you have a newer GPU (RTX 50 series or Blackwell), replace `cu126` with `cu128` in both torch lines of `environment.yml`.

### Setup

From the project root:

```bash
# Recreate the Conda environment
# (add `-n <name>` to use a different name if `clean_rl_mn` already exists)
conda env create --file environment.yml

# Activate the environment
conda activate clean_rl_mn

# Check that PyTorch can see the GPU (should print True and your GPU name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Running the experiments

```bash
# Create the W&B sweep
# You will be prompted to log in to Weights & Biases if necessary.
python run_sweep.py create --config configs/sweeps/lcfm_experiments.yaml
```

The W&B project name is configured as `cleanRL`. The W&B entity is left unset in the configuration, so W&B uses the entity associated with the account you are currently logged into.

After creating the sweep, the script prints the full sweep ID and the exact command needed to join it. The output should look something like this:

```text
(clean_rl_mn) ➜  ppo_trxl python run_sweep.py create --config configs/sweeps/lcfm_experiments.yaml
Starting main.py...
Folder sweep: 5 training config(s) found.
  configs/train_config/lcfm_experiments/as_pretrained.yaml
  configs/train_config/lcfm_experiments/categorical.yaml
  configs/train_config/lcfm_experiments/gaussian.yaml
  configs/train_config/lcfm_experiments/m5_base.yaml
  configs/train_config/lcfm_experiments/position_moving.yaml
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/<USER>/.netrc.
Create sweep with ID: <SWEEP_ID>
Sweep URL: https://wandb.ai/<ENTITY>/cleanRL/sweeps/<SWEEP_ID>
Sweep created: <ENTITY>/cleanRL/<SWEEP_ID>
Join with:  python run_sweep.py join --config configs/sweeps/lcfm_experiments.yaml --sweep-id <ENTITY>/cleanRL/<SWEEP_ID>
```

Copy and run the printed join command:

```bash
python run_sweep.py join \
    --config configs/sweeps/lcfm_experiments.yaml \
    --sweep-id <ENTITY>/cleanRL/<SWEEP_ID>
```

The join command can be run in multiple terminals or on multiple machines to launch several workers in parallel. Each machine needs the environment set up as above and must be logged into a W&B account with access to the entity that created the sweep.
