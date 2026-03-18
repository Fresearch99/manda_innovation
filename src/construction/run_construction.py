"""
Execution guide for the manda_innovation construction pipeline.

Purpose
-------
This module gives a clean top-level entry point and documents the intended
execution order after the original monolithic script was split into topical
modules. The split is designed to stay as faithful as possible to the original
construction logic while making the code base easier to read and maintain.

Important note
--------------
The repository does not contain the confidential source data needed to run the
full pipeline. A user who clones the project should first update the local paths
in `config.py`, verify the required input files exist, and then run either:

1. `pipeline_reference.py` for the closest single-file version of the original, or
2. the section modules in the sequence documented below.
"""

from __future__ import annotations

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
    """Print the recommended execution order for the split construction modules."""
    print("Recommended execution order for src/construction/:")
    for i, name in enumerate(EXECUTION_ORDER, start=1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    print_execution_order()
