"""Capture machine specs and library versions, per the assignment's ground
rule: "Run everything on one machine and note its specs and library
versions." Writes a plain-text file meant to be quoted directly in the
report's methodology section.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import multiprocessing
import platform
import subprocess
import sys
from pathlib import Path

LIBRARIES = ["faiss-cpu", "faiss-gpu", "hnswlib", "h5py", "numpy", "pandas", "matplotlib", "pytest"]


def _library_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_gb() -> str:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1e6:.1f} GB"
    except OSError:
        pass
    return "unknown"


def _gpu_info() -> str:
    try:
        import faiss

        n_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    except Exception:
        n_gpus = 0
    if n_gpus == 0:
        return "none visible to faiss (CPU-only run)"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        names = out.stdout.strip().splitlines() or ["(name unavailable)"]
    except Exception:
        names = ["(nvidia-smi unavailable)"]
    return f"{n_gpus} visible to faiss: {', '.join(names)}"


def collect_environment_info() -> str:
    lines = [
        "Environment",
        "===========",
        f"Captured: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"OS: {platform.platform()}",
        f"CPU: {_cpu_model()} ({multiprocessing.cpu_count()} logical cores)",
        f"RAM: {_total_ram_gb()}",
        f"Python: {sys.version.split()[0]}",
        f"GPU: {_gpu_info()}",
        "",
        "Library versions:",
    ]
    for lib in LIBRARIES:
        lines.append(f"  {lib}: {_library_version(lib)}")
    return "\n".join(lines) + "\n"


def write_environment_info(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(collect_environment_info())
    print(f"wrote {out_path}")
