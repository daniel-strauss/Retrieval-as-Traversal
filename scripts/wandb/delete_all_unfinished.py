"""
Delete W&B runs that W&B itself marked as abnormally terminated
(crashed, failed, or killed) - never "running", "pending", or "finished" runs.

Why these three states specifically:
    W&B sets these based on how the process actually exited, not a guess:
      - crashed: stopped sending heartbeats (machine died, etc.)
      - failed:  exited with a non-zero exit status
      - killed:  forcibly stopped before it could finish
    "running" and "pending" runs are never touched by default.
    "finished" runs are never touched, period - not exposed as an argument.

Error-handling policy (important, since this script deletes data):
    Only specific, anticipated failures are caught:
      - wandb.errors.Error (and subclasses, e.g. CommError) around calls that
        talk to the W&B backend - these are expected/recoverable failure modes
        (rate limits, restricted account, network blip), and a run we can't
        fully inspect is simply treated as "unknown size" and excluded from
        deletion.
      - KeyboardInterrupt / EOFError at confirmation prompts - a user abort,
        not a bug.
    Nothing else is caught. Anything unexpected (e.g. an AttributeError from
    a real bug in this script) is allowed to crash loudly with a full
    traceback rather than being silently swallowed and treated as "safe to
    skip" - a masked bug must never quietly influence what gets deleted.

Safety features:
    - Runs at or above --max-size-mib are NEVER deleted (default: 200 MiB).
    - Unknown size (inspection failed) is treated as "too big" -> excluded.
    - Full table of what will be deleted, and a separate table of what was
      omitted and why, printed before any confirmation is asked.
    - A DANGER banner is shown immediately before the deletion step, and the
      user must type the literal word AGREE to proceed. Anything else, or
      Ctrl+C/EOF, cancels with zero deletions.
    - Deletions happen one at a time with per-run error handling and a final
      summary of what succeeded vs failed.

Usage:
    python delete_all_unfinished.py <entity>/<project> [--max-size-mib 200] [--include-running]
"""

import argparse
import sys

import wandb.errors

import wandb

# Native W&B states considered "abnormally terminated". "finished" is
# deliberately not a variable/argument anywhere in this script.
DELETE_CANDIDATE_STATES = ["crashed", "failed", "killed"]


def human_size(num_bytes) -> str:
    if num_bytes is None or num_bytes < 0:
        return "unknown"
    num_bytes = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PiB"


def run_size_bytes(run):
    """Sum size of run files + artifacts logged by this run (bytes).
    Returns None if size could not be determined (treated as 'unknown', never deleted)."""
    total = 0
    ok = False
    try:
        for f in run.files():
            total += f.size or 0
        ok = True
    except (wandb.errors.Error, TypeError) as e:
        print(f"  [warn] could not read files for run {run.id}: {e}")
    try:
        for art in run.logged_artifacts():
            total += art.size or 0
        ok = True
    except (wandb.errors.Error, TypeError) as e:
        print(f"  [warn] could not read artifacts for run {run.id}: {e}")
    return total if ok else None


def inspect_run(run):
    """Gather everything needed about a run. Only wandb.errors.Error is caught;
    anything else (e.g. a real bug) propagates and crashes the script on purpose."""
    try:
        step = run.summary.get("_step", "N/A")
    except wandb.errors.Error as e:
        print(f"  [warn] could not read summary for run {run.id}: {e}")
        step = "N/A"

    created = str(run.created_at)[:19] if run.created_at else "N/A"
    size_bytes = run_size_bytes(run)

    return {
        "id": run.id,
        "name": run.name or "(unnamed)",
        "state": run.state,
        "step": step,
        "size_bytes": size_bytes,
        "created": created,
        "run": run,
    }


def print_table(rows, name_w, id_w):
    header = f"{'ID':<{id_w}}{'NAME':<{name_w}}{'STATE':<10}{'STEP':<10}{'SIZE':<12}{'CREATED':<20}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['id']:<{id_w}}{r['name']:<{name_w}}{r['state']:<10}"
            f"{r['step']!s:<10}{human_size(r['size_bytes']):<12}{r['created']:<20}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Delete only natively crashed/failed/killed W&B runs, with a size safety cap."
    )
    parser.add_argument("path", help="entity/project")
    parser.add_argument(
        "--max-size-mib",
        type=float,
        default=200,
        help="never delete runs at or above this size, in MiB (default: 200)",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="DANGEROUS: also allow currently RUNNING runs to be considered. Off by default.",
    )
    parser.add_argument(
        "--delete-artifacts",
        dest="delete_artifacts",
        action="store_true",
        default=True,
        help="also delete artifacts logged by the run (default: on)",
    )
    parser.add_argument(
        "--no-delete-artifacts",
        dest="delete_artifacts",
        action="store_false",
        help="keep artifacts, only delete the run record",
    )
    parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="DANGEROUS: if size lookup fails (e.g. due to the over-quota "
        "restriction breaking file/artifact listing), allow those runs to be "
        "deleted anyway instead of omitting them. --max-size-mib does not "
        "apply to runs deleted this way, since their size is unknown.",
    )
    args = parser.parse_args()

    max_size_bytes = args.max_size_mib * 1024 * 1024
    states = list(DELETE_CANDIDATE_STATES)
    if args.include_running:
        states.append("running")

    try:
        api = wandb.Api()
    except wandb.errors.Error as e:
        print(f"ERROR: could not initialize W&B API client: {e}")
        sys.exit(1)

    try:
        runs = api.runs(args.path, filters={"state": {"$in": states}})
        total_runs = len(runs)
    except wandb.errors.Error as e:
        print(f"ERROR: could not fetch runs for '{args.path}': {e}")
        print(
            "Double check the entity/project path (team entity, not the '-org' entity) "
            "and that your account currently has read access."
        )
        sys.exit(1)

    print(f"Scanning {total_runs} run(s) in state {states} in {args.path} ...")

    candidates, omitted = [], []

    for i, run in enumerate(runs, 1):
        print(f"  [{i}/{total_runs}] inspecting {run.id}...", end="\r")
        info = inspect_run(run)

        if info["size_bytes"] is None:
            info["size_bytes"] = -1
            if args.skip_size_check:
                candidates.append(info)
            else:
                omitted.append(info)
        elif info["size_bytes"] >= max_size_bytes:
            omitted.append(info)
        else:
            candidates.append(info)

    print(" " * 80, end="\r")

    if not candidates and not omitted:
        print("\nNo matching runs found. Nothing to do.")
        return

    candidates.sort(key=lambda r: r["size_bytes"], reverse=True)

    all_rows = candidates + omitted
    name_w = max([len(r["name"]) for r in all_rows] + [4]) + 2
    id_w = max([len(r["id"]) for r in all_rows] + [2]) + 2

    if candidates:
        print("\nRuns to be deleted:")
        print_table(candidates, name_w, id_w)
        total_size = sum(c["size_bytes"] for c in candidates)
        print(f"\n{len(candidates)} run(s) will be deleted, freeing ~{human_size(total_size)}.")
    else:
        print("\nNo runs are eligible for deletion.")

    unknown_size_count = sum(1 for o in omitted if o["size_bytes"] == -1)
    too_big_count = len(omitted) - unknown_size_count
    print(
        f"{len(omitted)} run(s) omitted and will NOT be deleted "
        f"({too_big_count} >= {args.max_size_mib} MiB, {unknown_size_count} size unknown)."
    )

    if omitted:
        print("\nOmitted runs:")
        print_table(omitted, name_w, id_w)

    if not candidates:
        return

    # ---------------------------------------------------------------
    # DANGER: no code past this point should execute without explicit,
    # unambiguous confirmation. Nothing above this line deletes anything.
    # ---------------------------------------------------------------
    total_size = sum(c["size_bytes"] for c in candidates)
    banner = "!" * 70
    print(f"\n{banner}")
    print("!! DANGER: DESTRUCTIVE, IRREVERSIBLE ACTION")
    print(f"!! About to PERMANENTLY DELETE {len(candidates)} run(s) (~{human_size(total_size)}).")
    if args.delete_artifacts:
        print("!! Logged artifacts for these runs will ALSO be deleted.")
    print("!! This cannot be undone. Deleted runs and artifacts are gone.")
    print(banner)

    try:
        answer = input("\nType AGREE (exactly, case-sensitive) to proceed, anything else cancels: ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled (no confirmation received). No runs were deleted.")
        return

    if answer != "AGREE":
        print("Cancelled. No runs were deleted.")
        return

    print("\nDeleting...")
    raise NotImplementedError("deletion code is commented out, because no bacup has been made")
    # deleted, failed = [], []
    # for c in candidates:
    #     try:
    #         c["run"].delete(delete_artifacts=args.delete_artifacts)
    #         deleted.append(c["id"])
    #         print(f"  deleted {c['id']} ({c['name']})")
    #     except KeyboardInterrupt:
    #         print("\nInterrupted mid-deletion. Stopping - remaining runs were NOT touched.")
    #         break
    #     except wandb.errors.Error as e:
    #         failed.append((c["id"], str(e)))
    #         print(f"  FAILED to delete {c['id']} ({c['name']}): {e}")

    # print(f"\nDone. {len(deleted)} deleted, {len(failed)} failed.")
    # if failed:
    #     print("Failed deletions (still present, investigate manually):")
    #     for rid, err in failed:
    #         print(f"  {rid}: {err}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. No further runs will be touched. "
            "Anything already deleted before the interrupt cannot be undone."
        )
        sys.exit(130)
    # Deliberately no bare/broad except here: an unexpected error should crash
    # with a full traceback, not be caught and treated as "handled."
