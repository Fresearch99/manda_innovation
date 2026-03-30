"""
Single entry point for the manda_innovation construction pipeline.

This runner executes the split construction modules in sequence using a shared
Python namespace so that later scripts can rely on objects created by earlier
ones, similar to running notebook/script sections in order.

Usage
-----
From the construction folder:
    python run_construction.py

Optional flags:
    --dry-run    Print the execution order and validate file existence only.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

EXECUTION_ORDER = [
    "01_setup_helpers.py",
    "02_patent_panel_construction.py",
    "03_exploration_exploitation.py",
    "04_mobility_and_mover_metrics.py",
    "05_technology_similarity.py",
    "06_firm_fundamentals.py",
    "07_linking_and_manda.py",
    "08_final_panels.py",
]


def print_execution_order() -> None:
    """Print the construction-module execution order."""
    print("Construction pipeline order:")
    for i, name in enumerate(EXECUTION_ORDER, start=1):
        print(f"  {i}. {name}")



def validate_pipeline(base_dir: Path) -> list[Path]:
    """Return ordered script paths and raise if any expected file is missing."""
    script_paths = [base_dir / name for name in EXECUTION_ORDER]
    missing = [str(path) for path in script_paths if not path.exists()]
    if missing:
        missing_text = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "The construction pipeline is incomplete. Missing expected module(s):"
            f"{missing_text}"
        )
    return script_paths



def run_pipeline(base_dir: Path, dry_run: bool = False) -> None:
    """Execute all construction modules in order with a shared global namespace."""
    script_paths = validate_pipeline(base_dir)
    print_execution_order()

    if dry_run:
        print("\nDry run only. No modules were executed.")
        return

    print(f"\nRunning construction pipeline from: {base_dir}")

    # Shared namespace so variables/functions created in early stages remain
    # available to later stages, which is the closest behavior to the original
    # monolithic script split across files.
    shared_globals: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(script_paths[0]),
        "PIPELINE_CONTEXT": SimpleNamespace(base_dir=base_dir, execution_order=EXECUTION_ORDER),
    }

    for i, script_path in enumerate(script_paths, start=1):
        print(f"\n[{i}/{len(script_paths)}] Running {script_path.name} ...", flush=True)
        shared_globals["__file__"] = str(script_path)
        try:
            shared_globals = runpy.run_path(
                str(script_path),
                init_globals=shared_globals,
                run_name="__main__",
            )
        except Exception as exc:  # pragma: no cover - for runtime debugging
            print(f"\nPipeline failed in {script_path.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            raise

    print("\nConstruction pipeline completed successfully.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manda_innovation construction pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution order and validate that all expected files exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(base_dir=Path(__file__).resolve().parent, dry_run=args.dry_run)
