from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import AnalysisConfig


@dataclass
class LoadedData:
    """Container for the project datasets used by the active analysis."""

    firm_panel: pd.DataFrame
    inventor_ma_event_study_panel: pd.DataFrame
    inventor_year_ma_panel: pd.DataFrame
    firm_lag: pd.DataFrame


DEFAULT_FIRM_X = [
    "log_sale",
    "log_mv",
    "leverage",
    "market_to_book",
    "roa",
    "cash",
    "sale_growth",
]


def build_firm_lag_controls(firm_panel: pd.DataFrame) -> pd.DataFrame:
    """Create or recover lagged firm controls used downstream."""
    firm_x = [c for c in DEFAULT_FIRM_X if c in firm_panel.columns]
    firm_lag = firm_panel[["permco", "data_year", *firm_x]].copy()
    firm_lag = firm_lag.sort_values(["permco", "data_year"])

    existing_lag1 = [f"lag1_{c}" for c in firm_x if f"lag1_{c}" in firm_panel.columns]
    if existing_lag1:
        firm_lag = firm_panel[["permco", "data_year", *existing_lag1]].copy()
        firm_lag = firm_lag.rename(columns={"permco": "permco_event"})
        return firm_lag

    for c in firm_x:
        firm_lag[f"lag1_{c}"] = firm_lag.groupby("permco", sort=False)[c].shift(1)

    firm_lag = firm_lag.rename(columns={"permco": "permco_event"})
    return firm_lag[["permco_event", "data_year", *[f"lag1_{c}" for c in firm_x]]]


def merge_firm_controls_into_inventor_panels(
    inventor_ma_event_study_panel: pd.DataFrame,
    inventor_year_ma_panel: pd.DataFrame,
    firm_lag: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach lagged firm controls to both inventor-level panels."""
    es_panel = inventor_ma_event_study_panel.copy()
    iy_panel = inventor_year_ma_panel.copy()

    if "permco_event" not in es_panel.columns and "permco_assigned" in es_panel.columns:
        es_panel = es_panel.rename(columns={"permco_assigned": "permco_event"})

    es_panel = es_panel.merge(firm_lag, on=["permco_event", "data_year"], how="left")

    if "permco_assigned" in iy_panel.columns:
        tmp = iy_panel.copy()
        tmp["permco_assigned"] = pd.to_numeric(tmp["permco_assigned"], errors="coerce")
        tmp["data_year"] = pd.to_numeric(tmp["data_year"], errors="coerce")
        tmp = tmp.dropna(subset=["permco_assigned", "data_year"]).copy()
        tmp["permco_assigned"] = tmp["permco_assigned"].astype("int64")
        tmp["data_year"] = tmp["data_year"].astype("int64")

        iy_panel = tmp.merge(
            firm_lag.rename(columns={"permco_event": "permco_assigned"}),
            on=["permco_assigned", "data_year"],
            how="left",
        )

    return es_panel, iy_panel


def load_analysis_data(config: AnalysisConfig) -> LoadedData:
    """Load the datasets used by the cleaned analysis."""
    config.ensure_output_dirs()

    firm_panel = pd.read_pickle(config.manda_event_path)
    inventor_ma_event_study_panel = pd.read_pickle(config.inventor_ma_es_panel_path)
    inventor_year_ma_panel = pd.read_pickle(config.inventor_year_ma_panel_path)

    firm_lag = build_firm_lag_controls(firm_panel)
    inventor_ma_event_study_panel, inventor_year_ma_panel = merge_firm_controls_into_inventor_panels(
        inventor_ma_event_study_panel,
        inventor_year_ma_panel,
        firm_lag,
    )

    return LoadedData(
        firm_panel=firm_panel,
        inventor_ma_event_study_panel=inventor_ma_event_study_panel,
        inventor_year_ma_panel=inventor_year_ma_panel,
        firm_lag=firm_lag,
    )
