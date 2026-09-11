"""Test that training runs are deterministic given the same seed."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# ── constants ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
CONFIG_DIR = TEST_DIR / "configs"
LOG_CONF = TEST_DIR / "logging.yaml"

# Regex that matches the per-update training log lines.
# Captures everything *except* SPS (which is timing-dependent).
_LINE_RE = re.compile(
    r"^\s*(\d+)\s+SPS=\s*\d+\s+"
    r"(return=[\-\d.]+\s+"
    r"length=[\-\d.]+\s+"
    r"pi_loss=[\-\d.]+\s+"
    r"v_loss=[\-\d.]+\s+"
    r"entropy=[\-\d.]+\s+"
    r"r_loss=[\-\d.]+\s+"
    r"value=[\-\d.]+\s+"
    r"adv=[\-\d.]+)",
    re.MULTILINE,
)

N_LINES_TO_COMPARE = 4


def _collect_configs():
    """Yield (config_name, config_path) for every yaml in the configs dir."""
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        yield p.stem, str(p)


def _run_training(
    config_path: str, label: str = "", max_metric_lines: int = N_LINES_TO_COMPARE
) -> str:
    """Run main.py and return its output, killing the process after enough metric lines.

    The subprocess is terminated as soon as *max_metric_lines* metric log lines
    have been captured, so we don't have to wait for the full training run.
    Output is also streamed to the console in real time.
    """
    cmd = [
        sys.executable,
        "-u", # unbuffered stdout/stderr  
        str(REPO_ROOT / "main.py"),
        "--train_conf",
        config_path,
        "--log_conf",
        str(LOG_CONF),
    ]
    if label:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
    )
    collected: list[str] = []
    metric_count = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        collected.append(line)
        if _LINE_RE.search(line):
            metric_count += 1
            if metric_count >= max_metric_lines:
                print(
                    f"  [test] Got {metric_count} metric lines – terminating subprocess.",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "".join(collected)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed (exit {proc.returncode}):\n{''.join(collected[-50:])}")
    return "".join(collected)


def _parse_metric_lines(output: str, n: int = N_LINES_TO_COMPARE) -> list[tuple[str, str]]:
    """Return the first *n* (update_number, metrics_string) pairs from output."""
    matches = _LINE_RE.findall(output)
    return [(num, metrics.strip()) for num, metrics in matches[:n]]


@pytest.mark.parametrize(
    "config_name,config_path",
    list(_collect_configs()),
    ids=lambda x: x if isinstance(x, str) and not x.endswith(".yaml") else None,
)
def test_deterministic_seeding(config_name, config_path):
    """Two runs with the same config must produce identical metric lines."""
    out_a = _run_training(config_path)
    out_b = _run_training(config_path)

    lines_a = _parse_metric_lines(out_a)
    lines_b = _parse_metric_lines(out_b)

    assert len(lines_a) >= N_LINES_TO_COMPARE, (
        f"Run A produced only {len(lines_a)} metric lines (need {N_LINES_TO_COMPARE}).\n"
        f"Output:\n{out_a[-2000:]}"
    )
    assert len(lines_b) >= N_LINES_TO_COMPARE, (
        f"Run B produced only {len(lines_b)} metric lines (need {N_LINES_TO_COMPARE}).\n"
        f"Output:\n{out_b[-2000:]}"
    )

    for i, ((num_a, m_a), (num_b, m_b)) in enumerate(zip(lines_a, lines_b)):
        assert num_a == num_b, f"Update number mismatch at line {i}: {num_a!r} vs {num_b!r}"
        assert m_a == m_b, (
            f"Metrics differ at update {num_a} (line {i}):\n  Run A: {m_a}\n  Run B: {m_b}"
        )
