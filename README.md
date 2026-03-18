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
