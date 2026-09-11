"""Entry point for PPO-TrXL training.

Usage
-----
Minimal (uses the default log config):
    python main.py --train-conf configs/train_config/cleanrl_minigrid_example.yaml

With a custom log config:
    python main.py \\
        --train-conf configs/train_config/cleanrl_minigrid_example.yaml \\
        --log-conf  configs/log_config/default.yaml

Config layout
-------------
  configs/
    log_conf/
      default.yaml          ← logging / W&B / video settings  (LogConfig)
    train_conf/
      <experiment>.yaml     ← env, PPO and TrXL hyperparameters  (TrainConfig)

Derived fields (do NOT put these in any YAML — computed automatically)
----------------------------------------------------------------------
    batch_size      = num_envs * num_steps
    minibatch_size  = batch_size // num_minibatches
    num_iterations  = total_timesteps // batch_size
"""



# setting this env var to allow CuBLAS to use a deterministic algorithm for backward passes 
# (at the cost of some speed)
# TODO: consider making this configurable via a command-line argument or YAML config
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import typing
from dataclasses import dataclass, fields
from pathlib import Path
import tyro
import torch
import yaml

from src_new.config import Config, LogConfig, TrainConfig
from src_new.trainer.trainer import Trainer


####
# Options for Debugging and Development
####

# colored tracebacks with rich
from rich.traceback import install
install(width=220, suppress=[tyro, torch, yaml])

# show shape and full tensors in debbuger
# from torch._tensor_str import _tensor_str
# def _tensor_repr(t):
#    torch.set_printoptions(threshold=int(1e9), linewidth=200)
#    return f"# shape={list(t.shape)}, dtype={t.dtype}\n" + _tensor_str(t, indent=0)
# torch.Tensor.__repr__ = _tensor_repr # type: ignore


print("Starting main.py...")

# ---------------------------------------------------------------------------
# CLI: only the two YAML paths are accepted here; all hyperparameters live in
# the YAML files.
# ---------------------------------------------------------------------------

_DEFAULT_LOG_CONFIG = Path("configs/log_config/default.yaml")


@dataclass
class MainArgs:
    train_conf: Path
    """Path to the train-config YAML (e.g. configs/train_config/cleanrl_minigrid_example.yaml)."""
    log_conf: Path = _DEFAULT_LOG_CONFIG
    """Path to the log-config YAML (default: configs/log_config/default.yaml)."""


# ---------------------------------------------------------------------------
# YAML → dataclass loading helpers
# ---------------------------------------------------------------------------

def _build_field_map(dc):
    """Return {name: Field} for every field in dataclass *dc*."""
    return {f.name: f for f in fields(dc)}


def _coerce(field_map: dict, key: str, value):
    """Coerce a YAML scalar to the annotated type declared in *field_map*."""
    annotation = field_map[key].type

    # Resolve Optional[X] → X
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        non_none = [a for a in annotation.__args__ if a is not type(None)]
        if value is None:
            return None
        annotation = non_none[0]

    if value is None:
        return None

    # bool must come before int because bool is a subclass of int
    if annotation is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"true", "1", "yes"}

    try:
        return annotation(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config field '{key}': cannot coerce {value!r} to {annotation}"
        ) from exc


def _load_section(path: Path, dc, *, exclude: set[str] | None = None):
    """Load *path* as YAML and return a populated instance of dataclass *dc*.

    Parameters
    ----------
    path:
        YAML file to read.
    dc:
        Target dataclass type (e.g. ``LogConfig`` or ``TrainConfig``).
    exclude:
        Field names that must NOT appear in the YAML (e.g. derived fields).
    """
    with path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping at the top level.")

    field_map = _build_field_map(dc)
    exclude = exclude or set()
    allowed = {name for name in field_map if name not in exclude}

    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}\n"
            f"Allowed keys: {sorted(allowed)}"
        )

    coerced = {key: _coerce(field_map, key, value) for key, value in raw.items()}
    return dc(**coerced)


# Shared config loader for both single runs and sweep workers.
def load_config(train_config_path: Path, log_config_path: Path) -> Config:
    """Load the train/log YAML files into a runtime config object."""
    for label, path in (("train-config", train_config_path), ("log-config", log_config_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    log = _load_section(log_config_path, LogConfig)
    train = _load_section(
        train_config_path,
        TrainConfig,
        exclude={"batch_size", "minibatch_size", "num_iterations"},
    )
    return Config(log=log, train=train)


# Centralized W&B init so sweep and non-sweep runs behave the same way.
def init_wandb_run(config: Config, *, extra_config: dict | None = None) -> None:
    """Initialise W&B and pull any sweep overrides back into the dataclasses."""
    import wandb

    wandb_config = config.to_wandb_config()
    if extra_config:
        wandb_config.update(extra_config)

    wandb.init(
        project=config.log.wandb_project_name,
        entity=config.log.wandb_entity,
        sync_tensorboard=True,
        config=wandb_config,
        save_code=True,
    )
    config.apply_wandb_overrides(wandb.config)


# Reusable training entrypoint once a Config object is ready.
def run_training(config: Config, *, initialize_wandb: bool = True, extra_wandb_config: dict | None = None) -> None:
    """Run training from an already-loaded config."""
    if config.log.track and initialize_wandb:
        init_wandb_run(config, extra_config=extra_wandb_config)

    Trainer(config).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli = tyro.cli(MainArgs)
    try:
        config = load_config(cli.train_conf, cli.log_conf)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    run_training(config)


if __name__ == "__main__":
    main()
