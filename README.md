# M&A and Innovation

This repository contains cleaned code and documentation for a research project studying how mergers and acquisitions affect inventor mobility and innovation outcomes.

## Current status
This is a public-facing repository under construction. Cleaned scripts, documentation, and selected non-confidential outputs will be added over time.

## Repository structure
- `src/` cleaned scripts
- `notebooks/` reproducible notebooks
- `docs/` design notes and project documentation
- `figures/` selected figures
- `tables/` selected tables
- `data/` placeholder folders only; confidential data are not included

## Data
The underlying data used in this project are not publicly distributed in this repository.

## Author
Dominik Jurek

## Construction code
The main data-construction pipeline lives in `src/construction/`.

Suggested entry points:
- `src/construction/run_construction.py` — orchestrator and execution order guide
- `src/construction/pipeline_reference.py` — near-faithful single-file reference version of the original construction script, with the unused standalone inventor move panel section removed
- `src/construction/sections/` equivalent logic split across topical modules for easier navigation

## Notes for public use
The repository does not include the confidential raw and linked data required to run the full pipeline end to end. Path configuration, source file names, and cache/output targets should be adapted locally in `src/construction/config.py`.


## Analysis code
The cleaned analysis pipeline lives in `src/analysis/`.

Suggested entry points:
- `src/analysis/run_analysis.py` — simple orchestrator that runs the two main cleaned analysis branches in sequence
- `src/analysis/run_firm_panel_analysis.py` — firm-panel baseline, heterogeneity, advanced methods, and placebo workflow
- `src/analysis/run_inventor_year_analysis.py` — inventor-year event-study, heterogeneity, and optional downsampled advanced workflow
- `src/analysis/pipeline_reference.py` — guide to the cleaned analysis split and scope relative to the original monolithic analysis file
- `src/analysis/sections/` — equivalent logic split across topical modules for easier navigation

Main topical modules inside `src/analysis/sections/`:
- `firm_analysis.py` — stacked firm-panel preparation, matching, baseline DiD/event-study, and heterogeneity layers
- `advanced_methods.py` — advanced firm-panel estimators including causal forest, synthetic-control style routines, Sun-Abraham, and BJS
- `placebos.py` — placebo assignment and placebo test helpers for significant firm-panel results
- `inventor_year.py` — inventor-year role-vs-control preparation, heterogeneity utilities, downsampling, and inventor-year advanced-method helpers
- `utils.py` — shared helper functions used across the cleaned analysis pipeline

Scope notes:
- the old inventor mover panel is intentionally excluded because it is no longer part of the active project;
- the Callaway-Sant'Anna method for the firm panel is intentionally excluded from the cleaned public-facing analysis split;
- inventor-year CSDID helper utilities remain in the code, but are not wired into the default runner because their original execution block appears archived/commented rather than active.

Path configuration, local confidential inputs, and output targets should be adapted locally in `src/analysis/config.py`, just as the construction pipeline paths may need local adjustment in `src/construction/config.py`.
