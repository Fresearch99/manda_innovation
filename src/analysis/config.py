from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisConfig:
    """Central configuration for the analysis workflow."""

    base_project_path: Path = Path(
        os.environ.get("MANDA_PROJECT_PATH", ".")
    ).expanduser().resolve()
    
    version: int = int(os.environ.get("MANDA_VERSION", "1"))

    analysis_window: tuple[int, int] = (1980, 2020)
    event_study_window: tuple[int, int] = (-5, 5)
    event_study_ref_k: int = -1

    run_inventor_year_advanced: bool = True
    advanced_alpha: float = 0.05
    inventor_year_csdid_bootstrap_draws: int = 30

    @property
    def data_path(self) -> Path:
        return self.base_project_path / f"Patents/Data/Inventor_Mobility__v{self.version}"

    @property
    def out_path(self) -> Path:
        return self.data_path / "Model_outputs"

    @property
    def plot_path(self) -> Path:
        return self.out_path / "Plots"

    @property
    def inventor_mobility_path(self) -> Path:
        return self.data_path / "final_inventor_move_panel_annual.pkl"

    @property
    def firm_panel_path(self) -> Path:
        return self.data_path / "firm_year_panel.pkl"

    @property
    def manda_event_path(self) -> Path:
        return self.data_path / "final_firm_year_ma_event_study_panel.pkl"

    @property
    def inventor_ma_es_panel_path(self) -> Path:
        return self.data_path / "inventor_ma_event_study_panel.pkl"

    @property
    def inventor_year_ma_panel_path(self) -> Path:
        return self.data_path / "inventor_year_panel_ma_inventors.pkl"

    def ensure_output_dirs(self) -> None:
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)
