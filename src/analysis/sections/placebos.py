from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.sections.advanced_methods import run_borusyak_jaravel, run_sun_abraham
from analysis.sections.firm_analysis import run_did_and_event_study_for_all_outcomes


def build_placebo_treatment(
    df: pd.DataFrame,
    placebo_type: str,
    lead_lag: int = 3,
    unit_col: str = "permco",
    time_col: str = "data_year",
    treat_col: str = "Treated",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create lead or permuted placebo treatment timing on a panel."""
    df_pl = df.copy()
    for name in (unit_col, time_col):
        if name not in df_pl.columns:
            if isinstance(df_pl.index, pd.MultiIndex) and name in df_pl.index.names:
                df_pl[name] = df_pl.index.get_level_values(name)
            elif df_pl.index.name == name:
                df_pl[name] = df_pl.index
            else:
                raise ValueError(f"`{name}` not found in columns or index.")
    if isinstance(df_pl.index, pd.MultiIndex):
        if unit_col in df_pl.index.names or time_col in df_pl.index.names:
            df_pl = df_pl.reset_index(drop=True)
    elif df_pl.index.name in (unit_col, time_col):
        df_pl = df_pl.reset_index(drop=True)

    treated = df_pl[df_pl[treat_col] == 1]
    if treated.empty:
        return df_pl
    g_real = treated[[unit_col, time_col]].groupby(unit_col)[time_col].min()

    if placebo_type == "lead":
        g_lead = g_real - lead_lag
        g_lead.name = "G_lead"
        g_real.name = "G_real"
        df_pl = df_pl.merge(g_lead, on=unit_col, how="left").merge(g_real, on=unit_col, how="left")
        df_pl[treat_col] = ((df_pl["G_lead"].notna()) & (df_pl[time_col] >= df_pl["G_lead"])).astype(int)
        mask = df_pl["G_real"].isna() | (df_pl[time_col] <= df_pl["G_real"] - 1)
        df_pl = df_pl.loc[mask].copy().drop(columns=["G_lead", "G_real"])
    elif placebo_type == "permute":
        if rng is None:
            rng = np.random.default_rng()
        g_perm = pd.Series(data=rng.permutation(g_real.values), index=g_real.index, name="G_perm")
        df_pl = df_pl.merge(g_perm, on=unit_col, how="left")
        df_pl[treat_col] = ((df_pl["G_perm"].notna()) & (df_pl[time_col] >= df_pl["G_perm"])).astype(int)
        g_real.name = "G_real"
        df_pl = df_pl.merge(g_real, on=unit_col, how="left")
        mask = df_pl["G_real"].isna() | (df_pl[time_col] <= df_pl["G_real"] - 1)
        df_pl = df_pl.loc[mask].copy().drop(columns=["G_real", "G_perm"])
    else:
        raise ValueError("placebo_type must be 'lead' or 'permute'.")

    df_pl["Post_Treated"] = df_pl[treat_col].astype(int)
    treated_pl = df_pl.loc[df_pl[treat_col] == 1, [unit_col, time_col]]
    if not treated_pl.empty:
        g_pl = treated_pl.groupby(unit_col, sort=False)[time_col].min().rename("cohort_pl")
        if "cohort" in df_pl.columns:
            df_pl["cohort"] = df_pl[unit_col].map(g_pl).combine_first(df_pl["cohort"])
        else:
            df_pl["cohort"] = df_pl[unit_col].map(g_pl)
        df_pl["event_time"] = df_pl[time_col] - df_pl["cohort"]
    return df_pl


def run_placebos_for_significant(
    sig_tgt: pd.DataFrame,
    sig_acq: pd.DataFrame,
    stacked_target: pd.DataFrame,
    stacked_acquiror: pd.DataFrame,
    df_target: pd.DataFrame,
    df_acquiror: pd.DataFrame,
    controls: list[str],
    plc_out_dir: Path,
    plc_plot_dir: Path,
    alpha: float = 0.05,
    lead_lag: int = 3,
    random_state: int = 42,
):
    """Run the active placebo suite for significant firm-level outcomes."""
    tgt_outcomes = set(sig_tgt.loc[sig_tgt["p_value"] < alpha, "outcome"])
    acq_outcomes = set(sig_acq.loc[sig_acq["p_value"] < alpha, "outcome"])
    if not tgt_outcomes and not acq_outcomes:
        return

    covs_tgt_stack = [c for c in controls if c in stacked_target.columns]
    covs_acq_stack = [c for c in controls if c in stacked_acquiror.columns]

    for y in sorted(tgt_outcomes.union(acq_outcomes)):
        rng = np.random.default_rng(random_state)
        if y in tgt_outcomes and y in df_target.columns:
            df_lead = build_placebo_treatment(df_target, "lead", lead_lag=lead_lag, unit_col="permco", time_col="data_year", treat_col="Treated")
            if not isinstance(df_lead.index, pd.MultiIndex) or df_lead.index.nlevels != 2:
                df_lead = df_lead.reset_index(drop=True).set_index(["permco", "data_year"]).sort_index()
            if df_lead["Treated"].nunique() >= 2:
                run_did_and_event_study_for_all_outcomes(df_lead, "target_placebo_lead", [y], controls, plc_out_dir / "did" / "lead", plc_plot_dir / "did" / "lead", window=range(-5, lead_lag), omit=-1)
            df_perm = build_placebo_treatment(df_target, "permute", unit_col="permco", time_col="data_year", treat_col="Treated", rng=rng)
            if not isinstance(df_perm.index, pd.MultiIndex) or df_perm.index.nlevels != 2:
                df_perm = df_perm.reset_index(drop=True).set_index(["permco", "data_year"]).sort_index()
            run_did_and_event_study_for_all_outcomes(df_perm, "target_placebo_perm", [y], controls, plc_out_dir / "did" / "perm", plc_plot_dir / "did" / "perm")
            stack_lead = build_placebo_treatment(stacked_target, "lead", lead_lag=lead_lag, unit_col="permco", time_col="data_year", treat_col="Treated")
            stack_lead = stack_lead.loc[stack_lead["event_time"].between(-5, lead_lag - 1)].copy()
            run_sun_abraham(stack_lead, y, "target_placebo_lead", covs_tgt_stack, out_dir=plc_out_dir / "sun_abraham" / "lead", plot_dir=plc_plot_dir / "sun_abraham" / "lead")
            run_borusyak_jaravel(stack_lead, y, "target_placebo_lead", covs_tgt_stack, n_boot=0, out_dir=plc_out_dir / "bjs" / "lead", plot_dir=plc_plot_dir / "bjs" / "lead")
            stack_perm = build_placebo_treatment(stacked_target, "permute", unit_col="permco", time_col="data_year", treat_col="Treated", rng=rng)
            run_sun_abraham(stack_perm, y, "target_placebo_perm", covs_tgt_stack, out_dir=plc_out_dir / "sun_abraham" / "perm", plot_dir=plc_plot_dir / "sun_abraham" / "perm")
            run_borusyak_jaravel(stack_perm, y, "target_placebo_perm", covs_tgt_stack, n_boot=0, out_dir=plc_out_dir / "bjs" / "perm", plot_dir=plc_plot_dir / "bjs" / "perm")

        if y in acq_outcomes and y in df_acquiror.columns:
            df_lead = build_placebo_treatment(df_acquiror, "lead", lead_lag=lead_lag, unit_col="permco", time_col="data_year", treat_col="Treated")
            if not isinstance(df_lead.index, pd.MultiIndex) or df_lead.index.nlevels != 2:
                df_lead = df_lead.reset_index(drop=True).set_index(["permco", "data_year"]).sort_index()
            if df_lead["Treated"].nunique() >= 2:
                run_did_and_event_study_for_all_outcomes(df_lead, "acquiror_placebo_lead", [y], controls, plc_out_dir / "did" / "lead", plc_plot_dir / "did" / "lead", window=range(-5, lead_lag), omit=-1)
            df_perm = build_placebo_treatment(df_acquiror, "permute", unit_col="permco", time_col="data_year", treat_col="Treated", rng=rng)
            if not isinstance(df_perm.index, pd.MultiIndex) or df_perm.index.nlevels != 2:
                df_perm = df_perm.reset_index(drop=True).set_index(["permco", "data_year"]).sort_index()
            run_did_and_event_study_for_all_outcomes(df_perm, "acquiror_placebo_perm", [y], controls, plc_out_dir / "did" / "perm", plc_plot_dir / "did" / "perm")
            stack_lead = build_placebo_treatment(stacked_acquiror, "lead", lead_lag=lead_lag, unit_col="permco", time_col="data_year", treat_col="Treated")
            stack_lead = stack_lead.loc[stack_lead["event_time"].between(-5, lead_lag - 1)].copy()
            run_sun_abraham(stack_lead, y, "acquiror_placebo_lead", covs_acq_stack, out_dir=plc_out_dir / "sun_abraham" / "lead", plot_dir=plc_plot_dir / "sun_abraham" / "lead")
            run_borusyak_jaravel(stack_lead, y, "acquiror_placebo_lead", covs_acq_stack, n_boot=0, out_dir=plc_out_dir / "bjs" / "lead", plot_dir=plc_plot_dir / "bjs" / "lead")
            stack_perm = build_placebo_treatment(stacked_acquiror, "permute", unit_col="permco", time_col="data_year", treat_col="Treated", rng=rng)
            run_sun_abraham(stack_perm, y, "acquiror_placebo_perm", covs_acq_stack, out_dir=plc_out_dir / "sun_abraham" / "perm", plot_dir=plc_plot_dir / "sun_abraham" / "perm")
            run_borusyak_jaravel(stack_perm, y, "acquiror_placebo_perm", covs_acq_stack, n_boot=0, out_dir=plc_out_dir / "bjs" / "perm", plot_dir=plc_plot_dir / "bjs" / "perm")
