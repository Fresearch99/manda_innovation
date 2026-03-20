from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

from analysis.sections.utils import add_all_z_specs, add_panel_index, extract_r2, fe_panel_ols, save_regression_table


def prepare_firm_did_inputs(
    firm_panel: pd.DataFrame,
    analysis_window: tuple[int, int],
    event_study_window: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build the baseline firm panels and core variable lists.

    This preserves the source logic:
    - treated firms are acquirors or targets;
    - controls are `no_recent_MandA` firms;
    - lagged controls are created by shifting within firm.
    """
    base_df = firm_panel[firm_panel["data_year"].between(*analysis_window)].copy()

    control_df = base_df[base_df["ma_deal_role"] == "no_recent_MandA"].copy()
    did_df = base_df[base_df["ma_deal_role"].isin(["acquiror", "target"])].copy()
    did_df = did_df[did_df["years_from_ma_deal"].between(*event_study_window)]
    did_df = pd.concat([did_df, control_df], axis=0)

    did_df["Treated"] = (did_df["ma_deal_role"] != "no_recent_MandA").astype(int)
    did_df["Post"] = (did_df["years_from_ma_deal"].fillna(-1) >= 0).astype(int)
    did_df["Post_Treated"] = did_df["Treated"] * did_df["Post"]
    did_df["gname"] = np.where(
        did_df["years_from_ma_deal"].isna(),
        0,
        did_df["data_year"] - did_df["years_from_ma_deal"],
    ).astype(int)

    outcomes = [
        "total_patents", "cites", "xi_real",
        "backward_cites", "self_cites", "top1_patents",
        "top10_to_2_patents", "cited_patents", "uncited_patents",
        "exploration_firm", "exploitation_firm",
        "num_inventors", "departing_inventors_count",
        "sum_patents_pre_move_departures", "sum_cites_pre_move_departures",
        "sum_xi_real_pre_move_departures", "arriving_inventors_count",
        "sum_patents_pre_move_arrivals", "sum_cites_pre_move_arrivals",
    ]
    for y in outcomes:
        if y in did_df.columns:
            did_df[f"log1p_{y}"] = np.log1p(did_df[y])

    controls = [
        "lag1_log_sale",
        "lag1_leverage",
        "lag1_market_to_book",
        "lag1_roa",
        "lag1_cash",
        "lag1_sale_growth",
    ]
    base_cols = [c.replace("lag1_", "", 1) for c in controls]
    did_df = did_df.sort_values(["permco", "data_year"])
    did_df[[f"lag1_{c}" for c in base_cols]] = did_df.groupby("permco", sort=False)[base_cols].shift(1)

    df_acquiror = did_df[did_df["ma_deal_role"].isin(["acquiror", "no_recent_MandA"])].copy().reset_index()
    df_target = did_df[did_df["ma_deal_role"].isin(["target", "no_recent_MandA"])].copy().reset_index()

    analysis_outcomes = [
        "avg_novelty", "exploration_firm",
        "exploitation_firm", "rolling_self_similarity",
        "log1p_total_patents", "log1p_cites",
        "log1p_xi_real", "log1p_backward_cites", "log1p_self_cites",
        "log1p_top1_patents", "log1p_top10_to_2_patents", "log1p_cited_patents",
        "log1p_uncited_patents", "log1p_exploration_firm",
        "log1p_exploitation_firm", "log1p_num_inventors",
        "log1p_departing_inventors_count",
        "log1p_sum_patents_pre_move_departures",
        "log1p_sum_cites_pre_move_departures",
        "log1p_sum_xi_real_pre_move_departures",
        "log1p_arriving_inventors_count", "log1p_sum_patents_pre_move_arrivals",
        "avg_rel_top1_quality_departures", "avg_rel_top10_quality_departures",
        "avg_rel_novelty_departures", "avg_rel_exploration_departures",
        "avg_rel_patents_departures", "avg_rel_cites_departures",
        "avg_rel_top1_quality_arrivals", "avg_rel_top10_quality_arrivals",
        "avg_rel_novelty_arrivals", "avg_rel_exploration_arrivals",
        "avg_rel_patents_arrivals", "avg_rel_cites_arrivals",
    ]

    return did_df, df_acquiror, df_target, controls, analysis_outcomes


def build_matched_stacked_did_simple(
    df: pd.DataFrame,
    event_window: tuple[int, int] = (-5, 5),
    allow_reuse_controls: bool = False,
    ps_caliper: float | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    """Build the matched stacked panel used by the firm-level baseline analysis.

    Matching logic is intentionally preserved from the original file:
    - estimate cohort-specific propensity scores at t-1 using size and industry;
    - within `sic3`, match treated and control firms using Mahalanobis distance on
      (`log_sale`, `log_mv`);
    - optionally reject poor pairs using a propensity-score caliper.
    """
    required = {"sic3", "log_sale", "log_mv", "gname", "years_from_ma_deal", "Treated", "permco", "data_year"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    stacked_list: list[pd.DataFrame] = []
    used_controls: set = set()
    cohort_years = sorted(df.loc[df["Treated"] == 1, "gname"].dropna().astype(int).unique())

    for cohort_year in cohort_years:
        cov = df[df["data_year"] == (cohort_year - 1)].copy()
        pot = cov[(cov["gname"] == 0) | (cov["gname"] == cohort_year)].copy()
        pot = pot.dropna(subset=["sic3", "log_sale", "log_mv"])
        if pot.empty:
            continue

        try:
            X_cov = pot[["log_sale", "log_mv"]].reset_index(drop=True)
            sic_d = pd.get_dummies(pot["sic3"].astype(str), drop_first=True, prefix="sic3").reset_index(drop=True)
            X_ps = pd.concat([X_cov, sic_d], axis=1)
            y_ps = pot["Treated"].to_numpy()
            if y_ps.min() != y_ps.max():
                lr = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                )
                lr.fit(X_ps, y_ps)
                pot["pscore"] = lr.predict_proba(X_ps)[:, 1]
            else:
                pot["pscore"] = np.nan
        except Exception:
            pot["pscore"] = np.nan

        tr = pot[pot["Treated"] == 1].copy()
        ct = pot[pot["Treated"] == 0].copy()
        if not allow_reuse_controls and used_controls:
            ct = ct[~ct["permco"].isin(used_controls)]
        if tr.empty or ct.empty:
            continue

        matched_ids = []
        for sic_val, g_tr in tr.groupby("sic3"):
            g_ct = ct[ct["sic3"] == sic_val]
            if g_ct.empty:
                continue

            X_tr = g_tr[["log_sale", "log_mv"]].to_numpy(float)
            X_ct = g_ct[["log_sale", "log_mv"]].to_numpy(float)
            X_pool = np.vstack([X_tr, X_ct])
            V = np.cov(X_pool, rowvar=False, ddof=1) + np.eye(X_pool.shape[1]) * 1e-8
            VI = np.linalg.pinv(V)
            dist = cdist(X_tr, X_ct, metric="mahalanobis", VI=VI)
            min_idx = dist.argmin(axis=1)
            chosen_ct = g_ct.iloc[min_idx].copy()

            pairs = pd.DataFrame({"permco_t": g_tr["permco"].values, "permco_c": chosen_ct["permco"].values})
            pairs = (
                pairs
                .merge(g_tr[["permco", "pscore"]].rename(columns={"permco": "permco_t", "pscore": "ps_t"}), on="permco_t", how="left")
                .merge(chosen_ct[["permco", "pscore"]].rename(columns={"permco": "permco_c", "pscore": "ps_c"}), on="permco_c", how="left")
            )
            if ps_caliper is not None:
                pairs = pairs[(pairs["ps_t"].sub(pairs["ps_c"]).abs() <= ps_caliper) | pairs["ps_t"].isna() | pairs["ps_c"].isna()]
            if not pairs.empty:
                matched_ids.append(pairs[["permco_t", "permco_c"]])

        if matched_ids:
            pairs_all = pd.concat(matched_ids, ignore_index=True).drop_duplicates()
            if not allow_reuse_controls:
                used_controls.update(pairs_all["permco_c"].tolist())

            keep_ids = set(pairs_all["permco_t"]).union(set(pairs_all["permco_c"]))
            panel = df[df["permco"].isin(keep_ids)].copy()
            panel["cohort"] = cohort_year
            panel["event_time"] = panel["data_year"] - cohort_year
            lo, hi = event_window
            panel = panel[(panel["event_time"] >= lo) & (panel["event_time"] <= hi)]
            ps_map = pot[["permco", "pscore"]].drop_duplicates(subset=["permco"])
            panel = panel.merge(ps_map, on="permco", how="left")
            stacked_list.append(panel)

    if not stacked_list:
        return pd.DataFrame()

    stacked = pd.concat(stacked_list, ignore_index=True)
    stacked["Post"] = (stacked["event_time"] > 0).astype(int)
    stacked["Post_Treated"] = stacked["Post"] * stacked["Treated"]
    return add_panel_index(stacked, "permco", "data_year")


def compare_pre_post_ttests(df_raw, stacked_panel, covariates=("log_sale", "log_mv")) -> pd.DataFrame:
    """Compare treated-control balance before and after the matching/stacking step."""
    def _ttest_table(df, covariates):
        rows = []
        for v in covariates:
            tmp = df[["Treated", v]].dropna()
            if tmp["Treated"].nunique() < 2:
                rows.append({"variable": v, "mean_t": np.nan, "mean_c": np.nan, "diff": np.nan, "t_stat": np.nan, "p_value": np.nan})
                continue
            x1 = tmp.loc[tmp["Treated"] == 1, v].to_numpy()
            x0 = tmp.loc[tmp["Treated"] == 0, v].to_numpy()
            if x1.size < 2 or x0.size < 2:
                rows.append({"variable": v, "mean_t": np.mean(x1) if x1.size else np.nan, "mean_c": np.mean(x0) if x0.size else np.nan, "diff": np.nan, "t_stat": np.nan, "p_value": np.nan})
                continue
            t_stat, p_val = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
            rows.append({"variable": v, "mean_t": float(np.mean(x1)), "mean_c": float(np.mean(x0)), "diff": float(np.mean(x1) - np.mean(x0)), "t_stat": float(t_stat), "p_value": float(p_val)})
        return pd.DataFrame(rows).sort_values("variable")

    def build_prestack_tminus1_sample(df, covariates):
        cohort_years = df.loc[df["Treated"] == 1, "gname"].dropna().astype(int).unique()
        blocks = []
        for cy in cohort_years:
            t1 = df.loc[df["data_year"] == (cy - 1)].copy()
            if t1.empty:
                continue
            pot = t1[(t1["gname"] == cy) | (t1["gname"] == 0)].copy()
            if pot.empty:
                continue
            blocks.append(pot[["Treated", *covariates]])
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=["Treated", *covariates])

    pre_sample = build_prestack_tminus1_sample(df_raw, list(covariates))
    post_sample = stacked_panel.loc[stacked_panel["event_time"] == -1, ["Treated", *covariates]].copy()
    pre_tbl = _ttest_table(pre_sample, covariates) if not pre_sample.empty else pd.DataFrame()
    post_tbl = _ttest_table(post_sample, covariates) if not post_sample.empty else pd.DataFrame()

    if pre_tbl.empty and post_tbl.empty:
        return pd.DataFrame()

    out = pre_tbl.rename(columns={"mean_t": "pre_mean_t", "mean_c": "pre_mean_c", "diff": "pre_diff", "t_stat": "pre_t", "p_value": "pre_p"}).merge(
        post_tbl.rename(columns={"mean_t": "post_mean_t", "mean_c": "post_mean_c", "diff": "post_diff", "t_stat": "post_t", "p_value": "post_p"}),
        on="variable",
        how="outer",
    )
    return out[["variable", "pre_mean_t", "pre_mean_c", "pre_diff", "pre_t", "pre_p", "post_mean_t", "post_mean_c", "post_diff", "post_t", "post_p"]]


def run_did_and_event_study_for_all_outcomes(
    df,
    role,
    outcomes,
    controls,
    OUT_DIR,
    PLOT_DIR,
    window=range(-5, 6),
    omit=-1,
    alpha=0.05,
    include_triple: bool = False,
    interaction_col: str | None = None,
    interaction_name: str | None = None,
    interaction_type: str = "binary",
):
    """Main baseline runner for firm and inventor-year event studies.

    It supports both plain DiD/event-study estimation and the project's generic
    triple-DiD extension through the `include_triple` block.
    """
    OUT_DIR = Path(OUT_DIR)
    PLOT_DIR = Path(PLOT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    treated_col = "Treated"
    post_treated_col = "Post_Treated"
    event_year_col = "cohort"
    relative_year_col = "event_time"
    outcomes = [y for y in outcomes if y in df.columns]

    df_use = df.copy()
    Zname = None
    triple_vars: list[str] = []
    z_terms: list[str] = []
    triple_main_terms: list[str] = []

    if include_triple and interaction_col and (interaction_col in df_use.columns):
        Zname = interaction_name or interaction_col
        z_raw = pd.to_numeric(df_use[interaction_col], errors="coerce")

        if interaction_type == "continuous":
            df_use[Zname] = z_raw.astype(float)
            z_terms = [Zname]
        elif interaction_type == "categorical":
            z_cat = z_raw.copy()
            dummies = pd.get_dummies(z_cat, prefix=Zname, drop_first=True).astype("int8")
            df_use = pd.concat([df_use, dummies], axis=1)
            z_terms = list(dummies.columns)
        else:
            df_use[Zname] = (z_raw.fillna(0) != 0).astype("int8")
            z_terms = [Zname]

        for zt in z_terms:
            post_z = f"Post_{zt}"
            triple = f"Post_Treated_{zt}"
            z_num = pd.to_numeric(df_use[zt], errors="coerce").fillna(0)
            df_use[post_z] = df_use["Post"] * z_num
            df_use[triple] = df_use["Post_Treated"] * z_num
            triple_vars.extend([post_z, triple])
            triple_main_terms.append(triple)

    controls_all = controls + triple_vars
    role_tag = role if not Zname else f"{role}_ddd_{Zname}"

    sig_rows = []
    for y in outcomes:
        xvars = [post_treated_col] + [c for c in controls_all if c in df_use.columns]
        try:
            res_base = fe_panel_ols(y_var=y, x_vars=xvars, df_panel=df_use, cov="entity")
        except Exception as e:
            print(f"[skip] baseline fit failed for role={role}, outcome={y}: {e}")
            continue

        coef = res_base.params.get(post_treated_col, np.nan)
        pval = res_base.pvalues.get(post_treated_col, np.nan)
        nobs = getattr(res_base, "nobs", np.nan)
        r2 = extract_r2(res_base)

        post_z_hits = []
        triple_hits = []
        for zt in z_terms:
            post_term = f"Post_{zt}"
            triple_term = f"Post_Treated_{zt}"
            if post_term in res_base.params.index:
                post_z_hits.append(f"{post_term}: coef={res_base.params[post_term]:.4f}, p={res_base.pvalues[post_term]:.4f}")
            if triple_term in res_base.params.index:
                triple_hits.append(f"{triple_term}: coef={res_base.params[triple_term]:.4f}, p={res_base.pvalues[triple_term]:.4f}")

        sig_rows.append(
            {
                "role": role,
                "spec": "triple" if Zname else "baseline",
                "triple_var": Zname,
                "outcome": y,
                "coef_Post_Treated": coef,
                "p_value": pval,
                "significant": (pval < alpha) if pd.notna(pval) else False,
                "significant_triple": any(
                    (term in res_base.pvalues.index) and pd.notna(res_base.pvalues[term]) and (res_base.pvalues[term] < alpha)
                    for term in triple_main_terms
                ) if triple_main_terms else False,
                "post_z_terms": " | ".join(post_z_hits) if post_z_hits else "",
                "triple_terms": " | ".join(triple_hits) if triple_hits else "",
                "nobs": nobs,
                "r2_within": r2["r2_within"],
                "r2_between": r2["r2_between"],
                "r2_overall": r2["r2_overall"],
                "r2": r2["r2"],
            }
        )
        save_regression_table(res_base, OUT_DIR / f"baseline_{role_tag}_{y}.html")

        df_es = df_use.copy()
        df_es["event_year"] = df_es[event_year_col]
        df_es["rel_t"] = df_es[relative_year_col]

        for k in window:
            if k == omit:
                continue
            name = f"rt_{k}"
            df_es[name] = ((df_es["rel_t"] == k) & (df_es[treated_col] == 1)).astype(int)
            if z_terms:
                for zt in z_terms:
                    if zt in df_es.columns:
                        df_es[f"{name}__{zt}"] = df_es[name] * pd.to_numeric(df_es[zt], errors="coerce").fillna(0)

        es_dummies = [f"rt_{k}" for k in window if k != omit and f"rt_{k}" in df_es.columns]
        es_dummies_Z = [f"{dname}__{zt}" for dname in es_dummies for zt in z_terms if f"{dname}__{zt}" in df_es.columns]

        if es_dummies:
            try:
                base_controls_es = [c for c in controls if c in df_es.columns]
                res_es = fe_panel_ols(y_var=y, x_vars=es_dummies + es_dummies_Z + base_controls_es, df_panel=df_es, cov="entity")
            except Exception as e:
                print(f"[skip] event-study fit failed for role={role_tag}, outcome={y}: {e}")
            else:
                save_regression_table(res_es, OUT_DIR / f"event_study_{role_tag}_{y}.html")
                es_dummies_fit = [d for d in es_dummies if d in res_es.params.index]
                if not es_dummies_fit:
                    continue

                coefs = res_es.params.loc[es_dummies_fit]
                ses = res_es.std_errors.loc[es_dummies_fit]
                ks = [int(n.split("_")[1]) for n in es_dummies_fit]
                order = np.argsort(ks)
                ks_sorted = np.array(ks)[order]
                coefs_sorted = coefs.to_numpy()[order]
                ses_sorted = ses.to_numpy()[order]

                plt.figure(figsize=(9, 6))
                plt.axhline(0, linestyle="--")
                plt.errorbar(ks_sorted, coefs_sorted, yerr=1.96 * ses_sorted, fmt="o")
                plt.title(f"Event Study: {y} ({role_tag})")
                plt.xlabel(f"Years since event (ref = {omit} omitted)")
                plt.ylabel("Coefficient")
                plt.grid(True, alpha=0.3)
                plt.savefig(PLOT_DIR / f"es_{role_tag}_{y}.png", dpi=200, bbox_inches="tight")
                plt.close()

    sig_df = pd.DataFrame(sig_rows)
    if sig_df.empty:
        print(f"[{role_tag}] No successful fits; skipping summary.")
        return sig_df

    sort_cols = [c for c in ["spec", "p_value", "p_value_triple"] if c in sig_df.columns]
    if sort_cols:
        sig_df = sig_df.sort_values(sort_cols, na_position="last")
    sig_df.to_csv(OUT_DIR / f"baseline_significance_{role_tag}.csv", index=False)

    sig_list = sig_df.loc[sig_df.get("significant", False), "outcome"].tolist()
    print(f"[{role_tag}] Significant Post_Treated at α={alpha}: {sig_list}")
    if "significant_triple" in sig_df.columns:
        sig_list_triple = sig_df.loc[sig_df["significant_triple"], "outcome"].tolist()
        print(f"[{role_tag}] Significant Triple at α={alpha}: {sig_list_triple}")
    return sig_df


def run_all_heterogeneity_specs_for_panel(
    panel: pd.DataFrame,
    role_name: str,
    outcomes: list[str],
    controls: list[str],
    out_dir,
    plot_dir,
):
    """Run baseline and all active firm-level heterogeneity specifications."""
    panel = add_all_z_specs(
        panel,
        entity_col="permco",
        time_col="data_year",
        log_sales_col="log_sale",
        deal_col="ma_deal_value",
        mcap_col="market_value",
        mcap_is_log=False,
        rel_col="event_time",
    )

    sig_base = run_did_and_event_study_for_all_outcomes(panel, role_name, outcomes, controls, out_dir, plot_dir)
    z_specs = [
        ("Z_log_sales_q2", "categorical"),
        ("Z_log_sales_q3", "categorical"),
        ("Z_log_sales_q5", "categorical"),
        ("Z_log_sales_cont", "continuous"),
        ("Z_deal_rel_q2", "categorical"),
        ("Z_deal_rel_q3", "categorical"),
        ("Z_deal_rel_q5", "categorical"),
        ("Z_deal_rel_cont", "continuous"),
    ]

    sig_all = [sig_base]
    for z_col, z_type in z_specs:
        if z_col not in panel.columns:
            continue
        sig_z = run_did_and_event_study_for_all_outcomes(
            df=panel,
            role=role_name,
            outcomes=outcomes,
            controls=controls,
            OUT_DIR=out_dir,
            PLOT_DIR=plot_dir,
            window=range(-5, 6),
            omit=-1,
            include_triple=True,
            interaction_col=z_col,
            interaction_name=z_col,
            interaction_type=z_type,
        )
        sig_all.append(sig_z)

    sig_all = [x for x in sig_all if x is not None and not x.empty]
    if sig_all:
        pd.concat(sig_all, ignore_index=True).to_csv(Path(out_dir) / f"{role_name}_all_heterogeneity_significance.csv", index=False)
    return panel, sig_base
