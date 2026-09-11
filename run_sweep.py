"""Custom W&B sweep launcher with support for tied parameters.

Usage
-----
Step 1 – create the sweep once (registers it with W&B, prints the sweep ID):

    python run_sweep.py create --config configs/sweeps/entropy_seed_grid.yaml
    -> prints the sweep ID in the terminal

Step 2 – join the sweep with one agent per terminal (repeat in as many terminals
or machines as desired, at any point during the sweep):

    python run_sweep.py join --config configs/sweeps/entropy_seed_grid.yaml --sweep-id <entity/project/ID>

To scale up: open another terminal and run the join command again.
To scale down: kill any agent process (Ctrl+C or kill <pid>). The agent's
current run will be marked "crashed" on W&B after the heartbeat times out
(~30 s). To retry that sample, go to the W&B sweep page, open the crashed run,
and click "Re-run".

The sweep config must be passed to join because it contains the train_config /
log_config paths and tied-parameter definitions, which are stripped before
upload and are therefore not recoverable from W&B via the sweep ID alone.
"""

from pathlib import Path
from typing import Any

import tyro
import yaml

from main import load_config, run_training

app = tyro.extras.SubcommandApp()


# Read the custom sweep YAML from disk.
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Sweep config {path} must be a YAML mapping at the top level.")
    return raw


# Split W&B-native parameters from repo-specific tied parameters.
def _extract_ties(parameters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    untied: dict[str, Any] = {}
    ties: dict[str, str] = {}

    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Parameter '{name}' must be a mapping, got {type(spec).__name__}.")
        if "tie" in spec:
            tie_target = spec["tie"]
            if not isinstance(tie_target, str):
                raise ValueError(f"Parameter '{name}' has a non-string tie target: {tie_target!r}.")
            ties[name] = tie_target
            continue
        untied[name] = spec

    return untied, ties


# Resolve tied parameters after W&B has sampled the untied ones.
def _resolve_tied_overrides(overrides: dict[str, Any], ties: dict[str, str]) -> dict[str, Any]:
    resolved = dict(overrides)
    visiting: set[str] = set()

    def resolve(name: str) -> Any:
        if name in resolved:
            return resolved[name]
        if name not in ties:
            raise ValueError(
                f"Tied parameter '{name}' references '{ties.get(name)}', but no value is available."
            )
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"Detected cyclic parameter ties: {cycle}")

        visiting.add(name)
        target = ties[name]
        value = resolve(target)
        resolved[name] = value
        visiting.remove(name)
        return value

    for tied_name in ties:
        resolve(tied_name)

    return resolved


# ---------------------------------------------------------------------------
# Folder-sweep helpers
# ---------------------------------------------------------------------------

# Internal W&B parameter name used to pass the selected train-config path.
_FOLDER_SWEEP_PARAM = "sweep_train_config_path"


def _is_folder_sweep(raw: dict[str, Any]) -> bool:
    return "train_config_folder" in raw


def _build_folder_sweep_config(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], Path, list[Path]]:
    """Convert a folder-sweep YAML into a W&B grid sweep config.

    The folder-sweep YAML must contain:
      train_config_folder: path/to/folder/   # all *.yaml files become runs
      log_config: path/to/log_config.yaml

    An optional ``parameters`` block may be included; every key/value pair will
    be cross-producted with the train-config paths (grid method) and applied as
    overrides on top of the loaded config at runtime.  Tied parameters (``tie:``)
    are supported and resolved the same way as in parameter sweeps.

    All other top-level keys (name, metric, …) are forwarded verbatim to W&B.
    The method defaults to 'grid' so every combination is run exactly once.

    Returns
    -------
    wandb_sweep_config
        Ready to pass to ``wandb.sweep()``.
    ties
        Tied-parameter mapping (may be empty).
    log_config_path
        Shared log config for all runs.
    train_config_paths
        Sorted list of discovered training-config files.
    """
    try:
        train_config_folder = Path(raw.pop("train_config_folder"))
        log_config_path = Path(raw.pop("log_config"))
    except KeyError as exc:
        raise ValueError(f"Missing required folder-sweep key: {exc.args[0]}") from exc

    train_config_paths = sorted(
        p for p in train_config_folder.rglob("*") if p.suffix in {".yaml", ".yml"}
    )
    if not train_config_paths:
        raise ValueError(f"No YAML files found in {train_config_folder}")

    extra_parameters = raw.pop("parameters", {}) or {}
    untied_parameters, ties = _extract_ties(extra_parameters)

    raw.setdefault("method", "grid")
    raw["parameters"] = {
        _FOLDER_SWEEP_PARAM: {"values": [str(p) for p in train_config_paths]},
        **untied_parameters,
    }
    return raw, ties, log_config_path, train_config_paths


# ---------------------------------------------------------------------------
# Parameter-sweep helpers
# ---------------------------------------------------------------------------


# Convert the custom sweep YAML into a W&B sweep config plus local metadata.
def _build_wandb_sweep_config(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], Path, Path]:
    try:
        train_config = Path(raw.pop("train_config"))
        log_config = Path(raw.pop("log_config"))
    except KeyError as exc:
        raise ValueError(f"Missing required sweep key: {exc.args[0]}") from exc

    parameters = raw.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Sweep config must contain a non-empty 'parameters' mapping.")

    untied_parameters, ties = _extract_ties(parameters)
    raw["parameters"] = untied_parameters
    return raw, ties, train_config, log_config


@app.command
def create(config: Path) -> None:
    """Create a new W&B sweep from the custom sweep YAML."""
    raw_config = _load_yaml(config)

    import wandb

    if _is_folder_sweep(raw_config):
        wandb_sweep_config, _, log_config_path, train_config_paths = _build_folder_sweep_config(
            raw_config
        )
        base_config = load_config(train_config_paths[0], log_config_path)
        print(f"Folder sweep: {len(train_config_paths)} training config(s) found.")
        for p in train_config_paths:
            print(f"  {p}")
    else:
        wandb_sweep_config, _, train_config_path, log_config_path = _build_wandb_sweep_config(
            raw_config
        )
        base_config = load_config(train_config_path, log_config_path)

    base_config.log.track = True

    sweep_id = wandb.sweep(
        sweep=wandb_sweep_config,
        project=base_config.log.wandb_project_name,
        entity=base_config.log.wandb_entity,
    )
    entity = base_config.log.wandb_entity or wandb.api.default_entity
    full_id = f"{entity}/{base_config.log.wandb_project_name}/{sweep_id}"
    print(f"Sweep created: {full_id}")
    print(f"Join with:  python run_sweep.py join --config {config} --sweep-id {full_id}")


@app.command
def join(config: Path, sweep_id: str) -> None:
    """Join an existing W&B sweep as an agent.

    Args:
        config: Path to the custom sweep YAML (same file used for create).
        sweep_id: Full sweep ID returned by the create command: entity/project/abc123.
    """
    raw_config = _load_yaml(config)

    import wandb

    if _is_folder_sweep(raw_config):
        _, ties, log_config_path, train_config_paths = _build_folder_sweep_config(raw_config)
        # Use the first config just to resolve project / entity for wandb.agent.
        base_config = load_config(train_config_paths[0], log_config_path)
        _wandb_project = base_config.log.wandb_project_name
        _wandb_entity = base_config.log.wandb_entity

        def run_one() -> None:
            with wandb.init(
                project=_wandb_project,
                entity=_wandb_entity,
                sync_tensorboard=True,
                save_code=True,
            ):
                overrides = dict(wandb.config)
                train_config_path = Path(overrides.pop(_FOLDER_SWEEP_PARAM))
                run_config = load_config(train_config_path, log_config_path)
                run_config.log.track = True
                resolved_overrides = _resolve_tied_overrides(overrides, ties)
                run_config.apply_wandb_overrides(resolved_overrides)
                wandb.config.update(run_config.to_wandb_config(), allow_val_change=True)
                run_training(run_config, initialize_wandb=False)

    else:
        _, ties, train_config_path, log_config_path = _build_wandb_sweep_config(raw_config)
        base_config = load_config(train_config_path, log_config_path)
        _wandb_project = base_config.log.wandb_project_name
        _wandb_entity = base_config.log.wandb_entity

        def run_one() -> None:
            run_config = load_config(train_config_path, log_config_path)
            run_config.log.track = True

            with wandb.init(
                project=run_config.log.wandb_project_name,
                entity=run_config.log.wandb_entity,
                sync_tensorboard=True,
                save_code=True,
            ):
                overrides = dict(wandb.config)
                resolved_overrides = _resolve_tied_overrides(overrides, ties)
                run_config.apply_wandb_overrides(resolved_overrides)
                wandb.config.update(run_config.to_wandb_config(), allow_val_change=True)
                run_training(run_config, initialize_wandb=False)

    wandb.agent(
        sweep_id,
        function=run_one,
        entity=_wandb_entity,
        project=_wandb_project,
    )


def main() -> None:
    app.cli()


if __name__ == "__main__":
    main()
