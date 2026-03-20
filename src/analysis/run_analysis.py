from __future__ import annotations

"""
Repository-level analysis orchestrator.

This file mirrors the role of ``src/construction/run_construction.py`` in the
construction pipeline: it is a readable guide showing the two main cleaned
analysis entry points.

The underlying project is large, so the firm-panel and inventor-year
workflows remain separate to keep memory usage manageable and to preserve the
logic of the original script.
"""

from analysis.run_firm_panel_analysis import main as run_firm_panel_analysis
from analysis.run_inventor_year_analysis import main as run_inventor_year_analysis


def main() -> None:
    # Run the two major analysis branches in the same order they naturally
    # follow in the cleaned repository. Users can comment out one branch if
    # they only want to inspect or execute part of the workflow.
    run_firm_panel_analysis()
    run_inventor_year_analysis()


if __name__ == "__main__":
    main()
