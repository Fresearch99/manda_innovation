from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from analysis.sections.utils import add_panel_index, fe_panel_ols


def run_causal_forest(
    panel: pd.DataFrame,
    outcome: str,
    base_covs: list[str],
    role_tag: str,
    post_period_window: tuple[int, int] = (1, 3),
    out_dir: Path = Path("."),
    plot_dir: Path = Path("."),
):
    """Estimate heterogeneous treatment effects on the stacked firm panel.

    pre-period X at `event_time = -1`, post-period outcome averaged over a short window, 
    and the stack's treatment indicator used as the binary treatment.
    """
    print(f"[CF] Running Causal Forest for '{role_tag}' on outcome='{outcome}'...")
    if outcome not in panel.columns or not all(c in panel.columns for c in base_covs):
        print("  -> Skipping: outcome or covariates not found in panel.")
        return None

    pre_covs = panel.loc[panel["event_time"] == -1, base_covs].reset_index(level="permco").drop_duplicates("permco")
    post = panel.loc[panel["event_time"].between(*post_period_window), [outcome, "Treated"]].reset_index(level="permco")[["permco", outcome, "Treated"]]
    post_summary = post.groupby("permco", as_index=False).agg(Y_post=(outcome, "mean"), W=("Treated", "first"))
    df_est = pre_covs.merge(post_summary, on="permco", how="inner").dropna()
    if df_est.empty or df_est["W"].nunique() < 2:
        print("  -> Skipping: not enough data after cross-section build.")
        return None

    Y = df_est["Y_post"].to_numpy()
    W = df_est["W"].to_numpy().astype(int)
    X = df_est[base_covs].to_numpy()

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42),
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
    )
    cf.fit(Y, W, X=X)

    X_est = df_est[base_covs]
    ate = float(cf.ate(X=X_est))
    lo, hi = cf.ate_interval(X=X_est, alpha=0.05)
    cates = cf.effect(X=X_est)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"cf_{role_tag}_{outcome}_summary.txt", "w") as f:
        f.write(f"Causal Forest Summary: {outcome} ({role_tag})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Treatment Effect (ATE): {ate:.4f}\n")
        f.write(f"95% CI for ATE: ({lo:.4f}, {hi:.4f})\n\n")
        fi = pd.Series(cf.feature_importances_, index=base_covs).sort_values(ascending=False)
        f.write("Feature Importances for Heterogeneity:\n")
        f.write(fi.to_string())

    df_cates = pd.DataFrame({"permco": df_est["permco"].values, "cate": cates})
    df_cates.to_csv(out_dir / f"cf_{role_tag}_{outcome}_cates.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(cates, bins=40, edgecolor="white", alpha=0.8)
    plt.axvline(ate, color="red", linestyle="--", label=f"ATE = {ate:.3f}")
    plt.title(f"Distribution of CATEs: {outcome.replace('_', ' ').title()}")
    plt.suptitle(f"Role: {role_tag.title()}", y=0.98)
    plt.xlabel("Estimated Heterogeneous Treatment Effect")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"cf_{role_tag}_{outcome}_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {"ATE": ate, "ATE_CI": (float(lo), float(hi)), "n": len(cates), "df_est": df_est, "cates": cates, "df_cates": df_cates}


def run_synth_for_one_unit(
    stack: pd.DataFrame,
    outcome: str,
    role_tag: str,
    min_pre: int = 5,
    min_post: int = 3,
    out_dir: Path = Path("."),
    forced_permco: str | None = None,
    k_window: int = 5,
    pre_fit_window: int | None = 10,
):
    """Run the project's ridge-regularized synthetic control for one treated firm."""
    if outcome not in stack.columns:
        return None

    elig = stack[stack["Treated"] == 1].groupby("permco").filter(
        lambda g: (g["event_time"] < 0).sum() >= min_pre and (g["event_time"] >= 0).sum() >= min_post
    )
    if elig.empty:
        return None

    wide = stack.reset_index().pivot_table(index="data_year", columns="permco", values=outcome)

    if forced_permco is not None:
        chosen_permco = forced_permco
        chosen_cohort = int(elig.reset_index().groupby("permco")["cohort"].first().loc[forced_permco])
    else:
        cand = elig.reset_index().groupby("permco")["cohort"].first()
        chosen_permco, chosen_cohort = None, None
        for pid, cohort in cand.items():
            s = wide.get(pid)
            if s is not None:
                win = s.loc[(s.index >= cohort - 2) & (s.index <= cohort + 2)]
                if (win > 0).any():
                    chosen_permco, chosen_cohort = pid, int(cohort)
                    break
        if chosen_permco is None:
            chosen_permco = elig.index.get_level_values("permco")[0]
            chosen_cohort = int(elig["cohort"].iloc[0])

    donors_all = stack.reset_index().loc[lambda d: d["Treated"] == 0, "permco"].dropna().drop_duplicates().tolist()
    if chosen_permco in donors_all:
        donors_all.remove(chosen_permco)

    pre_mask = wide.index < chosen_cohort
    post_mask = wide.index >= chosen_cohort
    donor_ids = [
        pid for pid in donors_all
        if pid in wide.columns and wide.loc[pre_mask, pid].notna().sum() >= min_pre and wide.loc[post_mask, pid].notna().sum() >= min_post
    ]
    if len(donor_ids) < 2:
        return None

    cols = [chosen_permco] + donor_ids
    df_sc_raw = wide[cols].reset_index().melt(id_vars="data_year", var_name="permco", value_name="y").groupby(["permco", "data_year"], as_index=False)["y"].mean()
    years_treat = df_sc_raw.loc[(df_sc_raw["permco"] == chosen_permco) & (df_sc_raw["y"].notna()), "data_year"].drop_duplicates().sort_values()
    pre_years = years_treat[years_treat < chosen_cohort]

    valid_donors = []
    for pid in donor_ids:
        sub = df_sc_raw[(df_sc_raw["permco"] == pid) & (df_sc_raw["data_year"].isin(years_treat))]
        if (sub["data_year"].nunique() == len(years_treat)) and sub["y"].notna().all():
            pre_vals = sub[sub["data_year"].isin(pre_years)]["y"]
            if pre_vals.notna().sum() >= min_pre and pre_vals.std(skipna=True) > 0:
                valid_donors.append(pid)
    if len(valid_donors) < 2:
        return None

    kept_cols = [chosen_permco] + valid_donors
    wide_kept = wide[wide.columns.intersection(kept_cols)].loc[years_treat].copy()
    if pre_fit_window is None:
        pre_mask_bal = wide_kept.index < chosen_cohort
    else:
        pre_mask_bal = (wide_kept.index < chosen_cohort) & (wide_kept.index >= chosen_cohort - pre_fit_window)

    A_pre = wide_kept.loc[pre_mask_bal, valid_donors].values
    b_pre = wide_kept.loc[pre_mask_bal, chosen_permco].values
    years = wide_kept.index.values
    dist = np.clip(chosen_cohort - years[pre_mask_bal], 0, None)
    dmax = dist.max()
    w_time = ((1 - dist / dmax) ** 2 + 1e-6) if dmax > 0 else np.ones_like(dist) * 1e-6
    sqrtW = np.sqrt(w_time)[:, None]
    A_wls = A_pre * sqrtW
    b_wls = b_pre * sqrtW.ravel()
    lam = 1e-3
    A_aug = np.vstack([A_wls, np.sqrt(lam) * np.eye(A_wls.shape[1])])
    b_aug = np.concatenate([b_wls, np.zeros(A_wls.shape[1])])
    w_ridge, _ = nnls(A_aug, b_aug)

    alpha = b_pre.mean() - (A_pre @ w_ridge).mean()
    A_full = wide_kept[valid_donors].values
    y_synth_balanced = alpha + (A_full @ w_ridge)
    Y_treat_balanced = wide_kept[chosen_permco].values

    paths = pd.DataFrame({"treated": Y_treat_balanced, "synthetic": y_synth_balanced}, index=years_treat)
    paths["gap"] = paths["treated"] - paths["synthetic"]

    unit_dir = out_dir / "SCM" / role_tag / outcome / f"permco_{chosen_permco}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(unit_dir / "paths.csv", index=True)
    pd.DataFrame({"donor_permco": valid_donors, "weight": w_ridge}).to_csv(unit_dir / "weights.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(paths.index, paths["treated"], label="Treated", linewidth=2)
    plt.plot(paths.index, paths["synthetic"], label="Synthetic", linestyle="--")
    plt.axvline(x=int(chosen_cohort) - 0.5, color="black", linestyle=":", label=f"M&A ({chosen_cohort})")
    plt.title(f"Synthetic Control: {outcome.replace('_', ' ').title()}\nFirm {chosen_permco} ({role_tag})")
    plt.xlabel("Year")
    plt.ylabel(outcome)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(int(chosen_cohort) - k_window, int(chosen_cohort) + k_window)
    plt.savefig(unit_dir / "plot.png", dpi=200, bbox_inches="tight")
    plt.close()

    rel = paths.copy()
    rel["k"] = rel.index.to_series().astype(int) - int(chosen_cohort)
    rel["permco"] = chosen_permco
    rel = rel[["permco", "k", "treated", "synthetic", "gap"]]
    rel = rel.loc[rel["k"].between(-k_window, k_window)].reset_index(drop=True)
    return {"permco": chosen_permco, "cohort": chosen_cohort, "rel_gaps": rel}


def run_synth_for_all_units(
    stack: pd.DataFrame,
    outcome: str,
    role_tag: str,
    min_pre: int = 5,
    min_post: int = 3,
    out_dir: Path = Path("."),
    post_window: tuple[int, int] = (0, 5),
    k_window: int = 5,
) -> dict:
    """Run SCM for every eligible treated unit and aggregate by relative event time."""
    elig_ids = stack.loc[stack["Treated"] == 1].groupby("permco").filter(
        lambda g: g.loc[g["event_time"] < 0, outcome].notna().sum() >= min_pre and g.loc[g["event_time"] >= 0, outcome].notna().sum() >= min_post
    ).reset_index()["permco"].drop_duplicates().tolist()
    if not elig_ids:
        return {}

    rel_list = []
    for pid in elig_ids:
        res = run_synth_for_one_unit(stack, outcome, role_tag, min_pre=min_pre, min_post=min_post, out_dir=out_dir, forced_permco=pid, k_window=k_window)
        if res is not None and isinstance(res.get("rel_gaps"), pd.DataFrame):
            rel_list.append(res["rel_gaps"])
    if not rel_list:
        return {}

    rel = pd.concat(rel_list, ignore_index=True)
    rel = rel.loc[rel["k"].between(-k_window, k_window)].copy()
    es = rel.groupby("k", as_index=False)["gap"].agg(att_k="mean", n_units="size").sort_values("k")
    k0, k1 = post_window
    att_overall = es.loc[es["k"].between(k0, k1), "att_k"].mean()

    prefix = out_dir / f"scm_{role_tag}_{outcome}"
    out_dir.mkdir(parents=True, exist_ok=True)
    es.to_csv(Path(f"{prefix}_eventstudy.csv"), index=False)
    rel.to_csv(Path(f"{prefix}_stacked_gaps.csv"), index=False)

    with open(Path(f"{prefix}_summary.txt"), "w") as f:
        f.write(f"Stacked SCM summary: {outcome} ({role_tag})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Eligible treated units: {len(elig_ids)}\n")
        f.write(f"Successful fits: {rel['permco'].nunique()}\n")
        f.write(f"Overall ATT (k in [{k0},{k1}]): {att_overall:.4f}\n")

    ks = es["k"].to_numpy()
    treated_ids = rel["permco"].drop_duplicates().to_list()
    B = 500
    boot_mat = np.full((B, len(ks)), np.nan)
    boot_att_overall = np.full(B, np.nan)
    for b in range(B):
        samp = np.random.choice(treated_ids, size=len(treated_ids), replace=True)
        rb = pd.concat([rel.loc[rel["permco"] == pid] for pid in samp], ignore_index=True)
        es_b = rb.groupby("k", as_index=False)["gap"].mean().rename(columns={"gap": "att_k"}).set_index("k").reindex(ks)["att_k"].to_numpy()
        boot_mat[b, :] = es_b
        mask = (ks >= k0) & (ks <= k1)
        boot_att_overall[b] = np.nanmean(es_b[mask])

    ci_lo = np.nanpercentile(boot_mat, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_mat, 97.5, axis=0)
    att_ci_lo, att_ci_hi = np.nanpercentile(boot_att_overall, [2.5, 97.5])
    with open(Path(f"{prefix}_summary.txt"), "a") as f:
        f.write(f"95% CI for Overall ATT (k in [{k0},{k1}]): ({att_ci_lo:.4f}, {att_ci_hi:.4f})\n")

    plt.figure(figsize=(10, 6))
    plt.axhline(0, linewidth=1)
    plt.axvline(-0.5, linestyle=":", linewidth=1)
    plt.plot(ks, es["att_k"].to_numpy(), label="ATT(k)", linewidth=2)
    plt.fill_between(ks, ci_lo, ci_hi, alpha=0.25, label="95% CI")
    plt.title(f"Stacked Synthetic Control — Event Study\n{outcome.replace('_', ' ').title()} ({role_tag})")
    plt.xlabel("Relative Year k")
    plt.ylabel("Average Gap (Treated − Synthetic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(f"{prefix}_eventstudy_ci.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return {"ATT": att_overall, "event_study": es}


def run_sun_abraham(
    stack: pd.DataFrame,
    outcome: str,
    role_tag: str,
    controls: list[str] | None = None,
    window: tuple[int, int] = (-5, 5),
    ref_k: int = -1,
    out_dir: Path = Path("."),
    plot_dir: Path = Path("."),
    vcov: str = "twoway",
    entity_col: str = "permco",
    time_col: str = "data_year",
):
    """Run the interaction-weighted Sun-Abraham event study."""
    d = stack.copy()
    if isinstance(d.index, pd.MultiIndex) and set([entity_col, time_col]).issubset(d.index.names):
        d = d.reset_index(level=[entity_col, time_col])
    else:
        d = d.reset_index()

    d["g"] = d["cohort"]
    d["rel_k"] = d["event_time"]
    controls_in_use = [c for c in (controls or []) if c in d.columns]
    for c in [outcome, *controls_in_use]:
        if d[c].dtype == bool:
            d[c] = d[c].astype(float)
        elif not np.issubdtype(d[c].dtype, np.number):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d[controls_in_use + [outcome]] = d[controls_in_use + [outcome]].replace([np.inf, -np.inf], np.nan)
    d = d.loc[d[[outcome, *controls_in_use]].apply(np.isfinite).all(axis=1)].copy()

    ent_counts = d.groupby(entity_col)[time_col].nunique()
    d = d[d[entity_col].isin(ent_counts[ent_counts >= 2].index)]
    yr_counts = d.groupby(time_col)[entity_col].nunique()
    d = d[d[time_col].isin(yr_counts[yr_counts >= 2].index)]

    cohorts = sorted(d.loc[d["Treated"] == 1, "g"].dropna().unique().tolist())
    year_min = int(d[time_col].min())
    year_max = int(d[time_col].max())
    ks = [k for k in range(window[0], window[1] + 1) if k != ref_k]

    blocks = {}
    col_order = []
    mask_treated = d["Treated"].eq(1)
    for g in cohorts:
        g_int = int(g)
        k_lo = max(window[0], year_min - g_int)
        k_hi = min(window[1], year_max - g_int)
        gmask = mask_treated & (d["g"] == g_int)
        for k in range(k_lo, k_hi + 1):
            if k == ref_k:
                continue
            sel = gmask & (d["rel_k"] == k)
            if not sel.any():
                continue
            name = f"D_g{g_int}_k{k}"
            blocks[name] = sel.astype(bool)
            col_order.append(name)

    d = pd.concat([d, pd.DataFrame(blocks, index=d.index)], axis=1, copy=False)
    X_cols = col_order + controls_in_use
    res = fe_panel_ols(y_var=outcome, x_vars=X_cols, df_panel=add_panel_index(d, entity_col, time_col), cov=vcov)

    data_year_min = int(d[time_col].min())
    data_year_max = int(d[time_col].max())
    N_g = d.loc[d["Treated"] == 1].groupby("g")[entity_col].nunique()
    rows = []
    for k in ks:
        eligible = [int(g) for g in cohorts if (data_year_min <= int(g) + k <= data_year_max)]
        cols_k = [f"D_g{int(g)}_k{k}" for g in eligible if f"D_g{int(g)}_k{k}" in res.params.index]
        if not cols_k:
            continue
        Ng_elig = N_g.reindex(eligible).dropna()
        denom = float(Ng_elig.sum())
        if denom <= 0.0 or Ng_elig.empty:
            continue
        w = Ng_elig / denom
        a = pd.Series(0.0, index=res.params.index, dtype=float)
        for c in cols_k:
            g = int(c.split("_")[1][1:])
            if g in w.index:
                a[c] = float(w.loc[g])
        beta_iw = float(a @ res.params)
        var_beta = float(a @ res.cov @ a)
        beta_vec = np.array([float(res.params.get(f"D_g{int(g)}_k{k}", 0.0)) for g in w.index], dtype=float)
        w_vec = w.values.astype(float)
        N_elig = float(Ng_elig.sum())
        Vw = (np.diag(w_vec) - np.outer(w_vec, w_vec)) / N_elig if N_elig > 1 else np.zeros((len(w_vec), len(w_vec)), dtype=float)
        if N_elig > 1:
            Vw *= (N_elig / (N_elig - 1.0))
        var_w = float(beta_vec.T @ Vw @ beta_vec)
        se_iw = float(np.sqrt(max(var_beta + var_w, 0.0)))
        rows.append({"event_time": k, "estimate": beta_iw, "std_error": se_iw})

    out = pd.DataFrame(rows).sort_values("event_time")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"sa_{role_tag}_{outcome}_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.errorbar(out["event_time"], out["estimate"], yerr=1.96 * out["std_error"], fmt="o", capsize=4)
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"Sun & Abraham Event Study: {outcome.replace('_', ' ').title()}")
    plt.suptitle(f"Role: {role_tag.title()}", y=0.975)
    plt.xlabel("Years from M&A Event")
    plt.ylabel("IW-ATT")
    plt.grid(True, alpha=0.2)
    plt.savefig(plot_dir / f"sa_{role_tag}_{outcome}_plot.png", dpi=200, bbox_inches="tight")
    plt.close()
    return out, res


def _bjs_impute_tau(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    entity_col: str = "permco",
    time_col: str = "data_year",
    treat_col: str = "Post_Treated",
) -> pd.DataFrame | None:
    """Estimate untreated potential outcomes and return residualized treatment effects."""
    if outcome not in df.columns:
        return None

    df = df.copy()
    df["D_it"] = df[treat_col].astype(int)
    for name in (entity_col, time_col):
        if name not in df.columns:
            if isinstance(df.index, pd.MultiIndex) and name in df.index.names:
                df[name] = df.index.get_level_values(name)
            elif df.index.name == name:
                df[name] = df.index
            else:
                raise ValueError(f"`{name}` not found in columns or index")
    df = df.reset_index(drop=True)

    du = df[df["D_it"] == 0].dropna(subset=[outcome])
    if du.empty or du[entity_col].nunique() < 2 or du[time_col].nunique() < 2:
        return None

    use_covs = [c for c in (controls or []) if c in df.columns]
    if len(use_covs) == 0:
        du2 = du[[entity_col, time_col, outcome]].copy()
        du2[time_col] = pd.to_numeric(du2[time_col], errors="coerce")
        du2 = du2.dropna(subset=[time_col]).copy()
        du2[time_col] = du2[time_col].astype(int)
        lam = du2.groupby(time_col)[outcome].mean()
        lam = lam - lam.mean()
        alpha = pd.Series(0.0, index=du2[entity_col].unique())
        for _ in range(50):
            alpha = (du2[outcome] - du2[time_col].map(lam)).groupby(du2[entity_col]).mean()
            lam = (du2[outcome] - du2[entity_col].map(alpha)).groupby(du2[time_col]).mean()
            lam = lam - lam.mean()
        dm = df.copy()
        dm["_yhat_cf0"] = dm[entity_col].map(alpha).fillna(0.0) + dm[time_col].map(lam).fillna(0.0)
        dm["_tau"] = dm[outcome] - dm["_yhat_cf0"]
        return dm

    res = fe_panel_ols(y_var=outcome, x_vars=use_covs, df_panel=add_panel_index(du, entity_col, time_col))
    fe = res.estimated_effects.squeeze()
    fe.name = "fe_hat"
    beta = res.params
    const = beta.get("const", 0.0) if hasattr(beta, "get") else (beta["const"] if "const" in beta.index else 0.0)
    dm = df.merge(fe, how="left", left_on=[entity_col, time_col], right_index=True)
    dm["fe_hat"] = dm["fe_hat"].fillna(0.0)
    xb = (dm[use_covs].fillna(0.0) @ beta.reindex(use_covs).fillna(0.0)) if use_covs else 0.0
    dm["_yhat_cf0"] = dm["fe_hat"] + xb + const
    dm["_tau"] = dm[outcome] - dm["_yhat_cf0"]
    return dm


def run_borusyak_jaravel(
    df_full: pd.DataFrame,
    outcome: str,
    role_tag: str,
    controls: list[str] | None = None,
    window: tuple[int, int] = (-5, 5),
    n_boot: int = 500,
    random_state: int = 42,
    out_dir: Path = Path("."),
    plot_dir: Path = Path("."),
    entity_col: str = "permco",
    time_col: str = "data_year",
    treat_col: str = "Post_Treated",
    boot_method: str = "cluster_refit",
):
    """Run the Borusyak-Jaravel-Spiess imputation estimator."""
    d = df_full.copy()
    if outcome not in d.columns:
        return None
    d["k"] = d["event_time"]
    d_merged = _bjs_impute_tau(d, outcome, controls or [], entity_col=entity_col, time_col=time_col, treat_col=treat_col)
    if d_merged is None:
        return None

    treated_effects = d_merged[d_merged["D_it"] == 1]
    att_by_k = treated_effects.groupby("k")["_tau"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "att_k", "count": "n_k"})
    att_by_k = att_by_k[att_by_k["k"].between(window[0], window[1])].copy()
    d_merged["_resid_pre"] = d_merged[outcome] - d_merged["_yhat_cf0"]
    _pre = d_merged[(d_merged["k"] < 0) & (d_merged["D_it"] == 0)]
    pre_by_k = _pre.groupby("k")["_resid_pre"].mean().sort_index()
    ks_pre = np.sort(_pre["k"].unique())

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        ks = att_by_k["k"].to_numpy()

        if boot_method == "wild_fixed":
            te = d_merged[d_merged["D_it"] == 1].copy()
            g = te.groupby([entity_col, "k"])["_tau"].agg(sum_tau="sum", n_ck="size").reset_index()
            att_map = att_by_k.set_index("k")["att_k"]
            n_map = att_by_k.set_index("k")["n_k"]
            g["att_k"] = g["k"].map(att_map)
            g["N_k"] = g["k"].map(n_map)
            g["centered"] = g["sum_tau"] - g["att_k"] * g["n_ck"]
            clusters = g[entity_col].dropna().unique()
            boot_mat = np.full((n_boot, len(ks)), np.nan, dtype=float)
            pre_boot_mat = np.full((n_boot, len(ks_pre)), np.nan, dtype=float)
            pre = d_merged[(d_merged["k"] < 0) & (d_merged["D_it"] == 0)].copy()
            if not pre.empty:
                pre_g = pre.groupby([entity_col, "k"])["_resid_pre"].agg(sum_r="sum", n_ck="size").reset_index()
                pre_mean = pre.groupby("k")["_resid_pre"].mean()
                preN = pre.groupby("k")["_resid_pre"].size()
                pre_g["m_k"] = pre_g["k"].map(pre_mean)
                pre_g["N_k"] = pre_g["k"].map(preN)
                pre_g["centered"] = pre_g["sum_r"] - pre_g["m_k"] * pre_g["n_ck"]
            for b in range(n_boot):
                w = rng.choice([-1.0, 1.0], size=len(clusters))
                w_map = dict(zip(clusters, w))
                gw = g.copy()
                gw["w"] = gw[entity_col].map(w_map).fillna(0.0)
                adj = gw.assign(wcent=gw["w"] * gw["centered"]).groupby("k")["wcent"].sum()
                att_star = att_map + (adj / n_map)
                boot_mat[b, :] = [att_star.get(k, np.nan) for k in ks]
                if not pre.empty:
                    pw = pre_g.copy()
                    pw["w"] = pw[entity_col].map(w_map).fillna(0.0)
                    adjp = pw.assign(wcent=pw["w"] * pw["centered"]).groupby("k")["wcent"].sum()
                    m_star = pre_mean + (adjp / preN)
                    pre_boot_mat[b, :] = [m_star.get(k, np.nan) for k in ks_pre]
        else:
            if isinstance(d.index, pd.MultiIndex) and entity_col in d.index.names:
                firms = d.index.get_level_values(entity_col).dropna().unique()
                use_mi = True
            else:
                firms = d[entity_col].dropna().unique()
                use_mi = False
            boot_mat = np.full((n_boot, len(ks)), np.nan, dtype=float)
            pre_boot_mat = np.full((n_boot, len(ks_pre)), np.nan, dtype=float)
            for b in range(n_boot):
                sample_firms = rng.choice(firms, size=len(firms), replace=True)
                db = pd.concat([d[d.index.get_level_values(entity_col) == f] for f in sample_firms], axis=0) if use_mi else pd.concat([d[d[entity_col] == f] for f in sample_firms], ignore_index=True)
                db["k"] = db["event_time"]
                dm_b = _bjs_impute_tau(db, outcome, controls or [], entity_col=entity_col, time_col=time_col, treat_col=treat_col)
                if dm_b is None:
                    continue
                eff_b = dm_b[dm_b["D_it"] == 1]
                tb = eff_b.groupby("k")["_tau"].mean()
                boot_mat[b, :] = [tb.get(k, np.nan) for k in ks]
                dm_b["_resid_pre"] = dm_b[outcome] - dm_b["_yhat_cf0"]
                pre_b = dm_b[(dm_b["k"] < 0) & (dm_b["D_it"] == 0)]
                tb_pre = pre_b.groupby("k")["_resid_pre"].mean()
                pre_boot_mat[b, :] = [tb_pre.get(k, np.nan) for k in ks_pre]

        ci_lo = np.nanpercentile(boot_mat, 2.5, axis=0)
        ci_hi = np.nanpercentile(boot_mat, 97.5, axis=0)
        att_by_k["ci_lo"] = att_by_k["k"].map(dict(zip(ks, ci_lo)))
        att_by_k["ci_hi"] = att_by_k["k"].map(dict(zip(ks, ci_hi)))
        pre_ci_lo = np.nanpercentile(pre_boot_mat, 2.5, axis=0)
        pre_ci_hi = np.nanpercentile(pre_boot_mat, 97.5, axis=0)
    else:
        pre_ci_lo = pre_ci_hi = np.array([])

    stem = f"bjs_{role_tag}_{outcome.replace('log1p_', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    att_by_k.to_csv(out_dir / f"{stem}_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(att_by_k["k"], att_by_k["att_k"], marker="o", label="ATT(k) Estimate")
    if "ci_lo" in att_by_k.columns:
        plt.fill_between(att_by_k["k"], att_by_k["ci_lo"], att_by_k["ci_hi"], alpha=0.2, label="95% Bootstrap CI")
    if not pre_by_k.empty:
        plt.plot(pre_by_k.index, pre_by_k.values, linestyle="--", label="Pre-period residual (diagnostic)", color="gray")
        if n_boot > 0:
            _mask = np.isin(ks_pre, pre_by_k.index.values)
            if _mask.any():
                plt.fill_between(ks_pre[_mask], pre_ci_lo[_mask], pre_ci_hi[_mask], alpha=0.15, label="Pre-period residual CI", color="gray")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"BJS Imputation Event Study: {outcome.replace('_', ' ').title()}")
    plt.suptitle(f"Role: {role_tag.title()}", y=0.92)
    plt.xlabel("Years from M&A Event")
    plt.ylabel("Imputed ATT Estimate")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_dir / f"{stem}_plot.png", dpi=200, bbox_inches="tight")
    plt.close()
    return att_by_k
