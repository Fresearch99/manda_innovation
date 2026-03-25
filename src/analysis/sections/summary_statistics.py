from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _event_time_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["event_time", "years_from_ma_deal"])


def _firm_id_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["permco", "permco_event"])


def _inventor_id_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["inventor_id"])


def _unit_id_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["stack_unit_id", "inv_event_id"])


def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan


def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.median()) if s.notna().any() else np.nan


def _safe_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.std()) if s.notna().any() else np.nan


def _treated_series(df: pd.DataFrame, role: str | None = None) -> pd.Series:
    if "Treated" in df.columns:
        return pd.to_numeric(df["Treated"], errors="coerce").fillna(0).astype(int)
    if role is not None and "ma_deal_role" in df.columns:
        return (df["ma_deal_role"].astype(str) == str(role)).astype(int)
    if "is_treated" in df.columns:
        return pd.to_numeric(df["is_treated"], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index, dtype=int)


# -----------------------------------------------------------------------------
# Generic panel summaries
# -----------------------------------------------------------------------------

def make_panel_overview(
    df: pd.DataFrame,
    *,
    panel_name: str,
    role: str | None = None,
) -> pd.DataFrame:
    event_time_col = _event_time_col(df)
    firm_col = _firm_id_col(df)
    inventor_col = _inventor_id_col(df)
    unit_col = _unit_id_col(df)
    treated = _treated_series(df, role=role)

    rows: list[tuple[str, object]] = [("panel_name", panel_name), ("n_rows", int(len(df)))]

    if firm_col is not None:
        rows.append(("n_unique_firms", int(df[firm_col].nunique(dropna=True))))
    if inventor_col is not None:
        rows.append(("n_unique_inventors", int(df[inventor_col].nunique(dropna=True))))
    if unit_col is not None:
        rows.append(("n_unique_units", int(df[unit_col].nunique(dropna=True))))
    if event_time_col is not None:
        rows.extend(
            [
                ("event_time_min", float(pd.to_numeric(df[event_time_col], errors="coerce").min())),
                ("event_time_max", float(pd.to_numeric(df[event_time_col], errors="coerce").max())),
                (
                    "n_pre_rows",
                    int((pd.to_numeric(df[event_time_col], errors="coerce") < 0).sum()),
                ),
                (
                    "n_post_rows",
                    int((pd.to_numeric(df[event_time_col], errors="coerce") >= 0).sum()),
                ),
            ]
        )

    rows.extend(
        [
            ("n_treated_rows", int(treated.sum())),
            ("n_control_rows", int((1 - treated).sum())),
            ("treated_row_share", float(treated.mean()) if len(treated) else np.nan),
        ]
    )

    if unit_col is not None:
        tmp = pd.DataFrame({"unit": df[unit_col], "treated": treated}).drop_duplicates()
        rows.extend(
            [
                ("n_treated_units", int(tmp["treated"].sum())),
                ("n_control_units", int((1 - tmp["treated"]).sum())),
                ("treated_unit_share", float(tmp["treated"].mean()) if len(tmp) else np.nan),
            ]
        )

    return pd.DataFrame(rows, columns=["metric", "value"])


# -----------------------------------------------------------------------------
# Baseline means / medians in the pre period
# -----------------------------------------------------------------------------

def make_pre_period_baseline_means(
    df: pd.DataFrame,
    *,
    outcomes: Iterable[str],
    role: str | None = None,
    panel_name: str,
) -> pd.DataFrame:
    event_time_col = _event_time_col(df)
    if event_time_col is None:
        raise ValueError("Could not find an event-time column. Expected 'event_time' or 'years_from_ma_deal'.")

    treated = _treated_series(df, role=role)
    pre_mask = pd.to_numeric(df[event_time_col], errors="coerce") < 0
    pre_df = df.loc[pre_mask].copy()
    pre_treated = treated.loc[pre_mask]

    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        if outcome not in pre_df.columns:
            continue

        s_all = pd.to_numeric(pre_df[outcome], errors="coerce")
        s_treated = pd.to_numeric(pre_df.loc[pre_treated == 1, outcome], errors="coerce")
        s_control = pd.to_numeric(pre_df.loc[pre_treated == 0, outcome], errors="coerce")

        rows.append(
            {
                "panel_name": panel_name,
                "outcome": outcome,
                "pre_period_n": int(s_all.notna().sum()),
                "treated_pre_n": int(s_treated.notna().sum()),
                "control_pre_n": int(s_control.notna().sum()),
                "pre_mean_all": _safe_mean(s_all),
                "pre_median_all": _safe_median(s_all),
                "pre_sd_all": _safe_std(s_all),
                "pre_mean_treated": _safe_mean(s_treated),
                "pre_median_treated": _safe_median(s_treated),
                "pre_sd_treated": _safe_std(s_treated),
                "pre_mean_control": _safe_mean(s_control),
                "pre_median_control": _safe_median(s_control),
                "pre_sd_control": _safe_std(s_control),
                "treated_minus_control_pre_mean": _safe_mean(s_treated) - _safe_mean(s_control),
                "treated_minus_control_pre_median": _safe_median(s_treated) - _safe_median(s_control),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Public write helpers used by the run_* scripts
# -----------------------------------------------------------------------------

def write_firm_summary_outputs(
    *,
    out_dir: str | Path,
    df_role_raw: pd.DataFrame,
    stacked_df: pd.DataFrame,
    role_name: str,
    headline_outcomes: Iterable[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_overview = make_panel_overview(df_role_raw, panel_name=f"firm_raw_{role_name}", role=role_name)
    stacked_overview = make_panel_overview(stacked_df, panel_name=f"firm_stacked_{role_name}", role=role_name)
    pd.concat([raw_overview, stacked_overview], ignore_index=True).to_csv(
        out_dir / f"summary_panel_overview_firm_{role_name}.csv", index=False
    )

    baseline = make_pre_period_baseline_means(
        stacked_df,
        outcomes=headline_outcomes,
        role=role_name,
        panel_name=f"firm_stacked_{role_name}",
    )
    baseline.to_csv(out_dir / f"summary_pre_period_baseline_means_firm_{role_name}.csv", index=False)


def write_inventor_summary_outputs(
    *,
    out_dir: str | Path,
    inv_es: pd.DataFrame,
    role_tag: str,
    role_name: str,
    headline_outcomes: Iterable[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = make_panel_overview(inv_es, panel_name=role_tag, role=role_name)
    overview.to_csv(out_dir / f"summary_panel_overview_{role_tag}.csv", index=False)

    baseline = make_pre_period_baseline_means(
        inv_es,
        outcomes=headline_outcomes,
        role=role_name,
        panel_name=role_tag,
    )
    baseline.to_csv(out_dir / f"summary_pre_period_baseline_means_{role_tag}.csv", index=False)