# Construction workflow

This folder contains a GitHub-ready reorganization of the original construction
script `MandA_and_exploration_direction_v14.py`.

## Design goals
- preserve the original logic as closely as possible
- split the code into topical files that are easier to inspect and edit
- keep original variable names where practical so the refactor is easy to audit
- remove the standalone inventor move panel construction because that output is not used downstream
- add more explanatory file-level comments so a new user understands what each block produces

## What was removed
Only the standalone inventor move panel construction in Section 15.4 was dropped.
Move identification, mover benchmarking, rolling similarity measures, and the
inventor-year / event-study panels that depend on move information were retained.

## Files
- `pipeline_reference.py`: near-faithful single-file version of the original script, minus the unused standalone move-panel block
- `01_setup_helpers.py` to `08_final_panels.py`: split section files preserving the original construction order
- `archive/original_master_script.py`: untouched uploaded source for comparison
- `config.py`: centralized path settings for local adaptation
