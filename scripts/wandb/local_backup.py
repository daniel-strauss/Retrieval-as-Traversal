#!/usr/bin/env python3
"""
Full local backup of Weights & Biases runs before deleting them.

For every run in <entity>/<project> this script saves, under
./wandb_backup/<project>/<run_id>/:
  - metadata.json   -> id, name, state, tags, created_at, url, etc.
  - config.json      -> run config (hyperparameters)
  - summary.json      -> final summary metrics
  - history.csv        -> full (unsampled) step-by-step metric history
  - files/                -> every file logged to the run (code, logs, media, checkpoints...)
  - artifacts/            -> every artifact logged BY the run (models, datasets, tables...)

Usage:
    pip install wandb pandas
    export WANDB_API_KEY=xxxx        # or run `wandb login` first
    python wandb_backup.py <entity>/<project> [--out ./wandb_backup] [--skip-artifacts] [--skip-files]

Then, once you've verified the backup, delete runs with:
    api.run("entity/project/run_id").delete()
or clean up storage-heavy artifacts/files via the API/UI as needed.
"""

import argparse
import json
import os
import sys
import traceback

import wandb


def safe(obj):
    """Make an object JSON-serializable, best-effort."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def backup_run(run, out_dir, skip_files=False, skip_artifacts=False):
    run_dir = os.path.join(out_dir, run.id)
    os.makedirs(run_dir, exist_ok=True)

    # --- metadata ---
    metadata = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "tags": run.tags,
        "created_at": str(run.created_at),
        "url": run.url,
        "entity": run.entity,
        "project": run.project,
        "group": run.group,
        "job_type": run.job_type,
        "notes": run.notes,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=safe)

    # --- config & summary ---
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(run.config), f, indent=2, default=safe)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(dict(run.summary), f, indent=2, default=safe)

    # --- full metric history (unsampled) ---
    try:
        import pandas as pd
        rows = list(run.scan_history())  # unsampled, unlike run.history()
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    except Exception as e:
        print(f"  [warn] history export failed for {run.id}: {e}")

    # --- files (code, logs, media, checkpoints saved during the run) ---
    if not skip_files:
        files_dir = os.path.join(run_dir, "files")
        os.makedirs(files_dir, exist_ok=True)
        for f in run.files():
            if f.size == 0:
                continue
            try:
                f.download(root=files_dir, replace=True, exist_ok=True)
            except Exception as e:
                print(f"  [warn] failed to download file {f.name} for run {run.id}: {e}")

    # --- artifacts logged by this run (models, datasets, tables) ---
    if not skip_artifacts:
        art_dir = os.path.join(run_dir, "artifacts")
        try:
            for art in run.logged_artifacts():
                dest = os.path.join(art_dir, art.name.replace("/", "_"))
                try:
                    art.download(root=dest)
                except Exception as e:
                    print(f"  [warn] failed to download artifact {art.name}: {e}")
        except Exception as e:
            print(f"  [warn] could not list artifacts for run {run.id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Back up all W&B runs in a project locally.")
    parser.add_argument("path", help="entity/project, e.g. myteam/myproject")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--skip-files", action="store_true", help="skip downloading run files")
    parser.add_argument("--skip-artifacts", action="store_true", help="skip downloading logged artifacts")
    parser.add_argument("--filters", default=None, help='optional wandb API filter, e.g. \'{"state":"finished"}\'')
    args = parser.parse_args()

    api = wandb.Api()
    filters = json.loads(args.filters) if args.filters else None
    runs = api.runs(args.path, filters=filters)

    project_name = args.path.split("/")[-1]
    out_dir = os.path.join(args.out, project_name)
    os.makedirs(out_dir, exist_ok=True)

    total = len(runs)
    print(f"Found {total} runs in {args.path}. Backing up to {out_dir}/")

    for i, run in enumerate(runs, 1):
        print(f"[{i}/{total}] {run.id} ({run.name})")
        try:
            backup_run(run, out_dir, skip_files=args.skip_files, skip_artifacts=args.skip_artifacts)
        except Exception:
            print(f"  [error] failed to back up run {run.id}")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()