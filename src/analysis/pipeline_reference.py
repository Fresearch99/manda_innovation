"""
Analysis pipeline reference
===========================

This repository-facing file is a guide to the cleaned analysis split.
Unlike the construction side, the monolithic analysis script was decomposed
directly into reusable modules and runner files rather than being preserved
here as another long executable script.

Use this file as a map:

- ``run_analysis.py`` runs the two main cleaned branches in sequence.
- ``run_firm_panel_analysis.py`` runs the firm-panel workflow.
- ``run_inventor_year_analysis.py`` runs the inventor-year workflow.
- ``sections/`` contains the logic split into topical modules.

Notes on scope relative to the original source script:

- The old inventor-mover panel code was removed because it is no longer used.
- The firm-panel Callaway-Sant'Anna path was removed because it is not part
  of the current public-facing workflow.
- Inventor-year CSDID utilities were retained, but the default runner does
  not execute them because the corresponding invocation in the source file
  appears archived/commented rather than active.
"""
