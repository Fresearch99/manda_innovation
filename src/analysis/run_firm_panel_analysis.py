from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.sections.advanced_methods import run_borusyak_jaravel, run_causal_forest, run_sun_abraham, run_synth_for_all_units
from analysis.config import AnalysisConfig
from analysis.data import load_analysis_data
from analysis.sections.firm_analysis import (
    build_matched_stacked_did_simple,
    compare_pre_post_ttests,
    prepare_firm_did_inputs,
    run_all_heterogeneity_specs_for_panel,
)
from analysis.sections.placebos import run_placebos_for_significant


# This script is intentionally imperative and readable. The goal is that a user
# cloning the repository can open one file and understand the full active
# firm-level workflow without needing to hunt through notebook cells.


def main() -> None:
    np.random.seed(42)
    config = AnalysisConfig()
    data = load_analysis_data(config)

    did_df, df_acquiror, df_target, controls, outcomes = prepare_firm_did_inputs(
        data.firm_panel,
        config.analysis_window,
        config.event_study_window,
    )

    stacked_acquiror = build_matched_stacked_did_simple(df_acquiror, event_window=config.event_study_window)
    stacked_target = build_matched_stacked_did_simple(df_target, event_window=config.event_study_window)

    pre_post_acq = compare_pre_post_ttests(df_acquiror, stacked_acquiror, covariates=["log_sale", "log_mv"])
    pre_post_tgt = compare_pre_post_ttests(df_target, stacked_target, covariates=["log_sale", "log_mv"])
    pre_post_acq.to_csv(config.out_path / "pre_post_balance_acquiror.csv", index=False)
    pre_post_tgt.to_csv(config.out_path / "pre_post_balance_target.csv", index=False)

    stacked_acquiror, sig_acq = run_all_heterogeneity_specs_for_panel(
        panel=stacked_acquiror,
        role_name="acquiror",
        outcomes=outcomes,
        controls=controls,
        out_dir=config.out_path,
        plot_dir=config.plot_path,
    )
    stacked_target, sig_tgt = run_all_heterogeneity_specs_for_panel(
        panel=stacked_target,
        role_name="target",
        outcomes=outcomes,
        controls=controls,
        out_dir=config.out_path,
        plot_dir=config.plot_path,
    )

    pd.concat([sig_acq, sig_tgt], ignore_index=True).to_csv(config.out_path / "baseline_significance_all.csv", index=False)

    # Advanced methods only for outcomes significant in the baseline stacked DiD.
    for y in sorted(set(sig_tgt.loc[sig_tgt["p_value"] < 0.05, "outcome"]).union(set(sig_acq.loc[sig_acq["p_value"] < 0.05, "outcome"]))):
        covs_tgt_stack = [c for c in controls if c in stacked_target.columns]
        covs_acq_stack = [c for c in controls if c in stacked_acquiror.columns]
        if y in set(sig_tgt.loc[sig_tgt["p_value"] < 0.05, "outcome"]):
            run_causal_forest(stacked_target, y, covs_tgt_stack, "target", out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")
            run_synth_for_all_units(stacked_target, y, "target", out_dir=config.out_path / "advanced")
            run_sun_abraham(stacked_target, y, "target", controls=covs_tgt_stack, out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")
            run_borusyak_jaravel(stacked_target, y, "target", controls=covs_tgt_stack, out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")
        if y in set(sig_acq.loc[sig_acq["p_value"] < 0.05, "outcome"]):
            run_causal_forest(stacked_acquiror, y, covs_acq_stack, "acquiror", out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")
            run_synth_for_all_units(stacked_acquiror, y, "acquiror", out_dir=config.out_path / "advanced")
            run_sun_abraham(stacked_acquiror, y, "acquiror", controls=covs_acq_stack, out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")
            run_borusyak_jaravel(stacked_acquiror, y, "acquiror", controls=covs_acq_stack, out_dir=config.out_path / "advanced", plot_dir=config.plot_path / "advanced")

    run_placebos_for_significant(
        sig_tgt=sig_tgt,
        sig_acq=sig_acq,
        stacked_target=stacked_target,
        stacked_acquiror=stacked_acquiror,
        df_target=df_target,
        df_acquiror=df_acquiror,
        controls=controls,
        plc_out_dir=config.out_path / "placebos",
        plc_plot_dir=config.plot_path / "placebos",
    )


if __name__ == "__main__":
    main()
