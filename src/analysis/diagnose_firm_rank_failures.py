from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.data import load_analysis_data
from analysis.sections.firm_analysis import (
    prepare_firm_did_inputs,
    build_matched_stacked_did_simple,
)


def summarize_one_outcome(df_panel, outcome, x_vars, role_name, event_window):
    d = df_panel.reset_index().copy() if isinstance(df_panel.index, pd.MultiIndex) else df_panel.copy()

    cols_needed = [outcome] + [c for c in x_vars if c in d.columns]
    missing_cols = [c for c in [outcome, *x_vars] if c not in d.columns]
    if outcome not in d.columns:
        return {
            "role": role_name,
            "outcome": outcome,
            "note": "outcome missing from panel",
        }

    dd = d.dropna(subset=cols_needed).copy()

    out = {
        "role": role_name,
        "outcome": outcome,
        "N_rows_after_dropna": len(dd),
        "N_firms": dd["permco"].nunique() if "permco" in dd.columns else np.nan,
        "N_years": dd["data_year"].nunique() if "data_year" in dd.columns else np.nan,
        "treated_rows": int(dd["Treated"].sum()) if "Treated" in dd.columns else np.nan,
        "control_rows": int((1 - dd["Treated"]).sum()) if "Treated" in dd.columns else np.nan,
        "post_rows": int(dd["Post"].sum()) if "Post" in dd.columns else np.nan,
        "post_treated_rows": int(dd["Post_Treated"].sum()) if "Post_Treated" in dd.columns else np.nan,
        "missing_expected_cols": ", ".join(missing_cols) if missing_cols else "",
    }

    for x in ["Treated", "Post", "Post_Treated", *x_vars]:
        if x in dd.columns:
            xx = pd.to_numeric(dd[x], errors="coerce")
            out[f"{x}_nunique"] = xx.nunique(dropna=True)
            out[f"{x}_std"] = xx.std()
            out[f"{x}_all_zero"] = bool(xx.fillna(0).eq(0).all())
        else:
            out[f"{x}_nunique"] = np.nan
            out[f"{x}_std"] = np.nan
            out[f"{x}_all_zero"] = np.nan

    y = pd.to_numeric(dd[outcome], errors="coerce")
    out["y_nonmissing"] = int(y.notna().sum())
    out["y_mean"] = y.mean()
    out["y_std"] = y.std()
    out["y_min"] = y.min()
    out["y_p1"] = y.quantile(0.01) if y.notna().any() else np.nan
    out["y_p5"] = y.quantile(0.05) if y.notna().any() else np.nan
    out["y_p50"] = y.quantile(0.50) if y.notna().any() else np.nan
    out["y_p95"] = y.quantile(0.95) if y.notna().any() else np.nan
    out["y_p99"] = y.quantile(0.99) if y.notna().any() else np.nan
    out["y_max"] = y.max()
    out["y_zero_share"] = float((y == 0).mean()) if y.notna().any() else np.nan
    out["y_neg_share"] = float((y < 0).mean()) if y.notna().any() else np.nan

    if "event_time" in dd.columns:
        ev = pd.to_numeric(dd["event_time"], errors="coerce")
        out["event_time_min"] = ev.min()
        out["event_time_max"] = ev.max()
        for k in range(event_window[0], event_window[1] + 1):
            out[f"n_event_time_{k}"] = int((ev == k).sum())

    return out


def add_collinearity_diagnostics(df_panel, outcome, x_vars, role_name):
    d = df_panel.reset_index().copy() if isinstance(df_panel.index, pd.MultiIndex) else df_panel.copy()

    if outcome not in d.columns:
        return {
            "role": role_name,
            "outcome": outcome,
            "n_xvars": np.nan,
            "matrix_rank": np.nan,
            "rank_deficiency": np.nan,
            "smallest_singular_value": np.nan,
            "condition_number": np.nan,
            "exact_duplicate_pairs": "",
            "perfect_corr_pairs": "",
            "note": "outcome missing from panel",
        }, pd.DataFrame()

    used_x = [x for x in x_vars if x in d.columns]
    cols_needed = [outcome] + used_x
    dd = d.dropna(subset=cols_needed).copy()

    if not used_x:
        return {
            "role": role_name,
            "outcome": outcome,
            "n_xvars": 0,
            "matrix_rank": np.nan,
            "rank_deficiency": np.nan,
            "smallest_singular_value": np.nan,
            "condition_number": np.nan,
            "exact_duplicate_pairs": "",
            "perfect_corr_pairs": "",
            "note": "no regressors present",
        }, pd.DataFrame()

    X = dd[used_x].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    reg_rows = []
    for c in X.columns:
        s = X[c]
        firms_with_var = np.nan
        years_with_var = np.nan

        if "permco" in dd.columns:
            firms_with_var = int(dd.assign(_x=s).groupby("permco")["_x"].nunique(dropna=True).gt(1).sum())
        if "data_year" in dd.columns:
            years_with_var = int(dd.assign(_x=s).groupby("data_year")["_x"].nunique(dropna=True).gt(1).sum())

        reg_rows.append({
            "role": role_name,
            "outcome": outcome,
            "regressor": c,
            "n_nonmissing": int(s.notna().sum()),
            "nunique": int(s.nunique(dropna=True)),
            "std": float(s.std()) if s.notna().any() else np.nan,
            "all_zero": bool(s.fillna(0).eq(0).all()),
            "constant": bool(s.nunique(dropna=True) <= 1),
            "firms_with_within_variation": firms_with_var,
            "years_with_cross_section_variation": years_with_var,
        })

    X_num = X.fillna(0.0).to_numpy(dtype=float)
    n_cols = X_num.shape[1]
    rank = int(np.linalg.matrix_rank(X_num)) if n_cols > 0 else 0

    smallest_sv = np.nan
    condition_number = np.nan
    if X_num.size > 0:
        try:
            svals = np.linalg.svd(X_num, full_matrices=False, compute_uv=False)
            if len(svals) > 0:
                smallest_sv = float(svals[-1])
                if np.isfinite(svals[0]) and np.isfinite(svals[-1]) and svals[-1] > 0:
                    condition_number = float(svals[0] / svals[-1])
                elif np.isfinite(svals[0]) and svals[-1] == 0:
                    condition_number = np.inf
        except Exception:
            pass

    dup_pairs = []
    corr_pairs = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = X[cols[i]]
            b = X[cols[j]]

            if a.equals(b):
                dup_pairs.append(f"{cols[i]}=={cols[j]}")
                continue

            mask = a.notna() & b.notna()
            if mask.sum() >= 2:
                try:
                    corr = np.corrcoef(a[mask], b[mask])[0, 1]
                    if np.isfinite(corr) and abs(corr) > 0.999999:
                        corr_pairs.append(f"{cols[i]}~{cols[j]}")
                except Exception:
                    pass

    out_row = {
        "role": role_name,
        "outcome": outcome,
        "n_xvars": n_cols,
        "matrix_rank": rank,
        "rank_deficiency": n_cols - rank,
        "smallest_singular_value": smallest_sv,
        "condition_number": condition_number,
        "exact_duplicate_pairs": " | ".join(dup_pairs),
        "perfect_corr_pairs": " | ".join(corr_pairs),
        "note": "",
    }

    return out_row, pd.DataFrame(reg_rows)


def main():
    config = AnalysisConfig()
    data = load_analysis_data(config)

    did_df, df_acquiror, df_target, controls, analysis_outcomes = prepare_firm_did_inputs(
        data.firm_panel,
        config.analysis_window,
        config.event_study_window,
    )

    stacked_acquiror = build_matched_stacked_did_simple(
        df_acquiror,
        event_window=config.event_study_window,
    )
    stacked_target = build_matched_stacked_did_simple(
        df_target,
        event_window=config.event_study_window,
    )

    print("Loaded and rebuilt firm analysis panels.")
    print(f"stacked_acquiror shape: {stacked_acquiror.shape}")
    print(f"stacked_target shape:   {stacked_target.shape}")
    print(f"controls used: {controls}")

    problem_outcomes = [
        "avg_rel_top1_quality_departures",
        "avg_rel_top10_quality_departures",
        "avg_rel_novelty_departures",
        "avg_rel_exploration_departures",
        "avg_rel_top1_quality_arrivals",
        "avg_rel_top10_quality_arrivals",
        "avg_rel_novelty_arrivals",
        "avg_rel_exploration_arrivals",
        "avg_rel_patents_arrivals",
        "avg_rel_cites_arrivals",
        "avg_novelty",
        "exploration_firm",
        "exploitation_firm",
        "rolling_self_similarity",
        "log1p_total_patents",
    ]

    rows = []
    col_summary_rows = []
    reg_diag_tables = []

    for outcome in problem_outcomes:
        rows.append(summarize_one_outcome(stacked_acquiror, outcome, controls, "acquiror", config.event_study_window))
        rows.append(summarize_one_outcome(stacked_target, outcome, controls, "target", config.event_study_window))

        acq_summary, acq_regs = add_collinearity_diagnostics(stacked_acquiror, outcome, controls, "acquiror")
        tgt_summary, tgt_regs = add_collinearity_diagnostics(stacked_target, outcome, controls, "target")
        col_summary_rows.extend([acq_summary, tgt_summary])
        if not acq_regs.empty:
            reg_diag_tables.append(acq_regs)
        if not tgt_regs.empty:
            reg_diag_tables.append(tgt_regs)

    diag = pd.DataFrame(rows)
    col_summary = pd.DataFrame(col_summary_rows)
    reg_diag = pd.concat(reg_diag_tables, ignore_index=True) if reg_diag_tables else pd.DataFrame()

    diag = diag.merge(
        col_summary,
        on=["role", "outcome"],
        how="left",
        suffixes=("", "_collinearity"),
    )

    front = [
        "role", "outcome", "N_rows_after_dropna", "N_firms", "N_years",
        "treated_rows", "control_rows", "post_rows", "post_treated_rows",
        "Treated_nunique", "Post_nunique", "Post_Treated_nunique",
        "Treated_all_zero", "Post_all_zero", "Post_Treated_all_zero",
        "n_xvars", "matrix_rank", "rank_deficiency", "smallest_singular_value", "condition_number",
        "exact_duplicate_pairs", "perfect_corr_pairs",
        "y_nonmissing", "y_mean", "y_std", "y_min", "y_p1", "y_p5",
        "y_p50", "y_p95", "y_p99", "y_max", "y_zero_share", "y_neg_share",
        "missing_expected_cols"
    ]
    diag = diag[[c for c in front if c in diag.columns] + [c for c in diag.columns if c not in front]]

    print("\n=== FULL DIAGNOSTIC TABLE ===")
    print(diag.to_string(index=False))

    flag_rows = []
    for _, r in diag.iterrows():
        flags = []

        n = r.get("N_rows_after_dropna", np.nan)
        pt = r.get("post_treated_rows", np.nan)
        ptn = r.get("Post_Treated_nunique", np.nan)
        yn50 = r.get("y_p50", np.nan)
        yn99 = r.get("y_p99", np.nan)
        ymax = r.get("y_max", np.nan)
        rank_def = r.get("rank_deficiency", np.nan)

        if pd.notna(n) and n < 500:
            flags.append("small usable sample")
        if pd.notna(pt) and pt == 0:
            flags.append("no post-treated rows")
        if pd.notna(ptn) and ptn <= 1:
            flags.append("Post_Treated no variation")
        if pd.notna(r.get("Post_nunique")) and r.get("Post_nunique") <= 1:
            flags.append("Post no variation")
        if pd.notna(r.get("Treated_nunique")) and r.get("Treated_nunique") <= 1:
            flags.append("Treated no variation")
        if pd.notna(rank_def) and rank_def > 0:
            flags.append(f"rank deficient ({int(rank_def)})")
        if isinstance(r.get("exact_duplicate_pairs", ""), str) and r.get("exact_duplicate_pairs", ""):
            flags.append("exact duplicate regressors")
        if isinstance(r.get("perfect_corr_pairs", ""), str) and r.get("perfect_corr_pairs", ""):
            flags.append("near-perfect regressor correlation")
        if pd.notna(yn50) and pd.notna(yn99) and abs(yn50) > 1e-12:
            if abs(yn99 / yn50) > 50:
                flags.append("very heavy upper tail")
        if pd.notna(ymax) and pd.notna(yn99) and abs(yn99) > 1e-12:
            if abs(ymax / yn99) > 5:
                flags.append("extreme max beyond p99")

        flag_rows.append({
            "role": r["role"],
            "outcome": r["outcome"],
            "flags": "; ".join(flags) if flags else ""
        })

    flagged = pd.DataFrame(flag_rows)
    flagged = flagged[flagged["flags"] != ""].copy()

    print("\n=== FLAGGED CASES ===")
    if flagged.empty:
        print("No obvious issues flagged by these simple rules.")
    else:
        print(flagged.to_string(index=False))

    if not reg_diag.empty:
        print("\n=== REGRESSOR-LEVEL DIAGNOSTICS (first 60 rows) ===")
        print(reg_diag.head(60).to_string(index=False))

    diag.to_csv(config.out_path / "firm_failed_outcome_diagnostics.csv", index=False)
    flagged.to_csv(config.out_path / "firm_failed_outcome_diagnostics_flagged.csv", index=False)
    col_summary.to_csv(config.out_path / "firm_failed_outcome_collinearity_summary.csv", index=False)
    reg_diag.to_csv(config.out_path / "firm_failed_outcome_regressor_diagnostics.csv", index=False)

    print("\nSaved:")
    print(config.out_path / "firm_failed_outcome_diagnostics.csv")
    print(config.out_path / "firm_failed_outcome_diagnostics_flagged.csv")
    print(config.out_path / "firm_failed_outcome_collinearity_summary.csv")
    print(config.out_path / "firm_failed_outcome_regressor_diagnostics.csv")


if __name__ == "__main__":
    main()
