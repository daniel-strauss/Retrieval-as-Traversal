## LCFM Experiments

Special thanks to [CleanRL](https://github.com/vwxyzjn/cleanrl) for their PPO-Transformer-XL implementation, which we adapted for this work.

### Running the experiments

From the project root:

```bash
# Recreate the Conda environment
conda env create --file environment.yml

# Activate the environment
conda activate clean_rl_mn

# Create the W&B sweep
# You will be prompted to log in to Weights & Biases if necessary.
python run_sweep.py create --config configs/sweeps/lcfm_experiments.yaml
```

The W&B project name is configured as `cleanRL`. The W&B entity is left unset in the configuration, so W&B uses the entity associated with the account you are currently logged into.

After creating the sweep, the script prints the full sweep ID and the exact command needed to join it.

The printed command has the form:

```bash
python run_sweep.py join \
    --config configs/sweeps/lcfm_experiments.yaml \
    --sweep-id <ENTITY>/cleanRL/<SWEEP_ID>
```

Usually, you can simply copy and run the command printed by `run_sweep.py`. It should look something like 

```bash
(clean_rl_mn) ➜  ppo_trxl python run_sweep.py create --config configs/sweeps/lcfm_experiments.yaml
Starting main.py...
Folder sweep: 5 training config(s) found.
  configs/train_config/lcfm_experiments/as_pretrained.yaml
  configs/train_config/lcfm_experiments/categorical.yaml
  configs/train_config/lcfm_experiments/gaussian.yaml
  configs/train_config/lcfm_experiments/m5_base.yaml
  configs/train_config/lcfm_experiments/position_moving.yaml
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/zzzzzzzz/.netrc.
Create sweep with ID: yyyyy
Sweep URL: https://wandb.ai/xxxxxxx/cleanRL/sweeps/xxxx
Sweep created: xxxxxxxxxxx/cleanRL/yyyyyyyy
Join with:  python run_sweep.py join --config configs/sweeps/lcfm_experiments.yaml --sweep-id zzzz/xxxx/yyyyy
```

The join command can be run in multiple terminals or on multiple machines to launch several workers in parallel.
