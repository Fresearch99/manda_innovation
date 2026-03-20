from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.data import load_analysis_data
from analysis.sections.firm_analysis import run_did_and_event_study_for_all_outcomes
from analysis.sections.inventor_year import (
    downsample_units_for_advanced,
    prepare_inventor_event_panel_for_did_role_vs_control,
    run_csdid_inventor_year,
    run_significant_inv_year_advanced,
)
from analysis.sections.utils import add_all_z_specs


# The active workflow in the original source only runs inventor-year baseline
# DiD/event-study, several heterogeneity layers, and optional Sun-Abraham on a
# downsampled panel. Inventor-year CSDID remains available as a utility, but its
# call is intentionally not executed here because the original loop had it
# archived in a commented block.


def main() -> None:
    np.random.seed(42)
    config = AnalysisConfig()
    data = load_analysis_data(config)

    inv_year_outcomes = [
        "total_patents",
        "cites",
        "xi_real",
        "novelty_score_group",
        "exploration_inv",
        "exploitation_inv",
        "is_move_year",
    ]
    inv_year_controls = [c for c in ["team_size", "inventor_age", "nca_enforce_score"] if c in data.inventor_ma_event_study_panel.columns]
    firm_controls = [c for c in data.firm_lag.columns if c.startswith("lag1_")]

    invy_out_dir = config.out_path / "inventor_year_event_study"
    invy_plot_dir = config.plot_path / "inventor_year_event_study"
    invy_adv_out_dir = invy_out_dir / "advanced"
    invy_adv_plot_dir = invy_plot_dir / "advanced"
    invy_out_dir.mkdir(parents=True, exist_ok=True)
    invy_plot_dir.mkdir(parents=True, exist_ok=True)
    invy_adv_out_dir.mkdir(parents=True, exist_ok=True)
    invy_adv_plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for role in ["acquiror", "target"]:
        for firm_tag, fctrl in [("firm", firm_controls), ("nofirm", [])]:
            for tag, nearest_only, xcontrols in [("nn1_nox", True, []), ("nn1_x", True, inv_year_controls)]:
                all_ctrls = xcontrols + fctrl
                inv_es = prepare_inventor_event_panel_for_did_role_vs_control(
                    inventor_ma_event_panel=data.inventor_ma_event_study_panel,
                    role=role,
                    window=config.event_study_window,
                    outcomes=inv_year_outcomes,
                    controls=all_ctrls,
                    nearest_only=nearest_only,
                    add_within_firm_rank_z=True,
                    min_firm_inventors_for_rank=4,
                )

                role_tag = f"inv_year_{role}_vs_control_{tag}_{firm_tag}"
                sig = run_did_and_event_study_for_all_outcomes(
                    df=inv_es,
                    role=role_tag,
                    outcomes=[y for y in inv_year_outcomes if y in inv_es.columns],
                    controls=[c for c in all_ctrls if c in inv_es.columns],
                    OUT_DIR=invy_out_dir,
                    PLOT_DIR=invy_plot_dir,
                    window=range(config.event_study_window[0], config.event_study_window[1] + 1),
                    omit=config.event_study_ref_k,
                )
                results.append(sig)

                # ---------------------------------------------------------
                # Optional: Callaway-Sant'Anna / CSDID for inventor-year panel
                # ---------------------------------------------------------
                # This is intentionally kept as a compact block so that it can
                # easily be commented out in GitHub-facing runs if desired.
                #
                # We use the same role-specific inventor-year panel that feeds
                # the baseline DiD/event-study specification. The cohort columns
                # (e.g., cs_g_year_target / cs_g_year_acquiror) are expected to
                # already be present in the prepared panel.
                #
                # Suggested default window:
                #   E_PRE = abs(event_study_ref_k)  -> usually 2
                #   E_MAX = 3                       -> modest post horizon
                #
                # Bootstrapping is kept small here to limit runtime in a public
                # repo example. Increase B later if needed.
                for cs_outcome in ["is_move_year", "total_patents", "cites"]:
                    if cs_outcome not in inv_es.columns:
                        continue
                    try:
                        run_csdid_inventor_year(
                            df_full=inv_es,
                            outcome=cs_outcome,
                            role_tag=role_tag,
                            role=role,
                            strict_role=True,
                            covariates=[c for c in all_ctrls if c in inv_es.columns],
                            id_col="inventor_id",
                            t_col="data_year",
                            out_dir=invy_out_dir / "csdid",
                            plot_dir=invy_plot_dir / "csdid",
                            min_cohort_size=200,
                            E_PRE=abs(config.event_study_ref_k),
                            E_MAX=3,
                            B=99,
                            seed=42,
                            verbose=True,
                        )
                    except Exception as exc:
                        print(f"[{role_tag} | {cs_outcome}] inventor-year CSDID skipped/failed: {exc}", flush=True)
                # To disable again, simply comment out the try/except block above.
                    
                for z_col in ["Z_upper_half_cum_patents_within_firm"]:
                    if z_col not in inv_es.columns:
                        continue
                    sig_z_rank = run_did_and_event_study_for_all_outcomes(
                        df=inv_es,
                        role=f"{role_tag}_ddd_{z_col}",
                        outcomes=[y for y in inv_year_outcomes if y in inv_es.columns],
                        controls=[c for c in all_ctrls if c in inv_es.columns],
                        OUT_DIR=invy_out_dir,
                        PLOT_DIR=invy_plot_dir,
                        window=range(config.event_study_window[0], config.event_study_window[1] + 1),
                        omit=config.event_study_ref_k,
                        include_triple=True,
                        interaction_col=z_col,
                        interaction_name=z_col,
                        interaction_type="binary",
                    )
                    results.append(sig_z_rank)

                inv_es = add_all_z_specs(
                    inv_es,
                    entity_col="inv_event_id",
                    time_col="data_year",
                    log_sales_col="lag1_log_sale",
                    deal_col="ma_deal_value",
                    mcap_col="lag1_log_mv",
                    mcap_is_log=True,
                    rel_col="event_time",
                )
                inv_z_specs = [
                    ("Z_log_sales_q2", "categorical"),
                    ("Z_log_sales_q3", "categorical"),
                    ("Z_log_sales_q5", "categorical"),
                    ("Z_log_sales_cont", "continuous"),
                    ("Z_deal_rel_q2", "categorical"),
                    ("Z_deal_rel_q3", "categorical"),
                    ("Z_deal_rel_q5", "categorical"),
                    ("Z_deal_rel_cont", "continuous"),
                ]
                for z_col, z_type in inv_z_specs:
                    if z_col not in inv_es.columns:
                        continue
                    sig_z = run_did_and_event_study_for_all_outcomes(
                        df=inv_es,
                        role=f"{role_tag}_ddd_{z_col}",
                        outcomes=[y for y in inv_year_outcomes if y in inv_es.columns],
                        controls=[c for c in all_ctrls if c in inv_es.columns],
                        OUT_DIR=invy_out_dir,
                        PLOT_DIR=invy_plot_dir,
                        window=range(config.event_study_window[0], config.event_study_window[1] + 1),
                        omit=config.event_study_ref_k,
                        include_triple=True,
                        interaction_col=z_col,
                        interaction_name=z_col,
                        interaction_type=z_type,
                    )
                    results.append(sig_z)

                gc.collect()
                if config.run_inventor_year_advanced:
                    if role == "acquiror":
                        inv_es_adv = downsample_units_for_advanced(inv_es, role=role, max_treated_units=30_000, max_control_units=30_000, seed=42)
                    else:
                        inv_es_adv = downsample_units_for_advanced(inv_es, role=role, max_treated_units=10_000, max_control_units=30_000, seed=42)
                    del inv_es
                    gc.collect()
                    run_significant_inv_year_advanced(
                        inv_es=inv_es_adv,
                        sig_df=sig,
                        role_tag=role_tag,
                        controls=all_ctrls,
                        out_dir=invy_adv_out_dir,
                        plot_dir=invy_adv_plot_dir,
                        alpha=config.advanced_alpha,
                        window=config.event_study_window,
                        ref_k=config.event_study_ref_k,
                        bjs_n_boot=200,
                        run_bjs=False,
                    )

    pd.concat(results, ignore_index=True).to_csv(invy_out_dir / "inventor_year_event_study_significance_selected.csv", index=False)


if __name__ == "__main__":
    main()
