"""
Delete unfinished W&B runs, but only if they're under a size threshold.

Safety features:
  - Runs over --max-size-mib are NEVER deleted, no matter what.
  - Prints a full table of candidate runs (name, state, highest step, size, date)
    before doing anything.
  - Reports how many runs were omitted for being too large.
  - Requires an explicit "yes" typed by the user before deleting anything.

Usage:
    python delete_all_unfinished.py <entity>/<project> [--max-size-mib 200]
"""

import argparse

import wandb

exclude_state = "finished"


def human_size(num_bytes: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PiB"


def run_size_bytes(run) -> int:
    """Sum size of run files + artifacts logged by this run (bytes)."""
    total = 0
    for f in run.files():
        total += f.size or 0
    for art in run.logged_artifacts():
        total += art.size or 0

    return total


def main():
    parser = argparse.ArgumentParser(description="Delete unfinished W&B runs under a size cap.")
    parser.add_argument("path", help="entity/project")
    parser.add_argument(
        "--max-size-mib",
        type=float,
        default=200,
        help="never delete runs at or above this size (MiB). Default: 200",
    )

    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        default=True,
        help="also delete artifacts logged by the run (default: on)",
    )
    args = parser.parse_args()

    max_size_bytes = args.max_size_mib * 1024 * 1024

    api = wandb.Api()
    runs = api.runs(args.path, filters={"state": {"$ne": exclude_state}})

    print(f"Scanning {len(runs)} unfinished runs in {args.path} ...")

    candidates = []
    omitted = []

    for i, run in enumerate(runs, 1):
        print(f"  [{i}/{len(runs)}] inspecting {run.id} ({run.name})...", end="\r")
        size_bytes = run_size_bytes(run)
        step = run.summary.get("_step", "N/A")
        created = str(run.created_at)[:19] if run.created_at else "N/A"

        info = {
            "id": run.id,
            "name": run.name or "(unnamed)",
            "state": run.state,
            "step": step,
            "size_bytes": size_bytes,
            "created": created,
            "run": run,
        }

        if size_bytes >= max_size_bytes:
            omitted.append(info)
        else:
            candidates.append(info)

    print(" " * 80, end="\r")  # clear progress line

    if not candidates and not omitted:
        print("No unfinished runs found. Nothing to do.")
        return

    # --- print table of deletion candidates ---
    candidates.sort(key=lambda r: r["size_bytes"], reverse=True)

    name_w = max([len(c["name"]) for c in candidates] + [4]) + 2
    id_w = max([len(c["id"]) for c in candidates] + [2]) + 2

    header = f"{'ID':<{id_w}}{'NAME':<{name_w}}{'STATE':<10}{'STEP':<10}{'SIZE':<10}{'CREATED':<20}"
    print("\nRuns to be deleted:")
    print(header)
    print("-" * len(header))
    for c in candidates:
        print(
            f"{c['id']:<{id_w}}{c['name']:<{name_w}}{c['state']:<10}"
            f"{c['step']!s:<10}{human_size(c['size_bytes']):<10}{c['created']:<20}"
        )

    total_size = sum(c["size_bytes"] for c in candidates)
    print(f"\n{len(candidates)} run(s) will be deleted, freeing ~{human_size(total_size)}.")
    print(
        f"{len(omitted)} run(s) omitted for being >= {args.max_size_mib} MiB (not deleted, safety limit)."
    )

    if omitted:
        print("\nOmitted (too large) runs:")
        for o in omitted:
            print(f"  {o['id']}  {o['name']:<{name_w}}  {human_size(o['size_bytes'])}")

    if not candidates:
        print("\nNo runs under the size threshold to delete. Exiting.")
        return

    # --- confirmation ---
    answer = (
        input(f"\nType 'yes' to permanently delete these {len(candidates)} run(s): ")
        .strip()
        .lower()
    )
    if answer != "yes":
        print("Aborted. No runs were deleted.")
        return

    raise NotImplementedError("LINES BELOW ONLY ADDED AFTER SUCCESSFULL backup")
    # print("\nDeleting...")
    # for c in candidates:
    #    c["run"].delete(delete_artifacts=args.delete_artifacts)
    #    print(f"  deleted {c['id']} ({c['name']})")

    # print("Done.")


if __name__ == "__main__":
    main()
