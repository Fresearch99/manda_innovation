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
"""
