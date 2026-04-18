from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def add_panel_index(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    """Create a sorted panel MultiIndex after removing duplicate entity-time rows."""
    return (
        df.copy()
        .drop_duplicates([entity_col, time_col])
        .set_index([entity_col, time_col])
        .sort_index()
    )


def fe_panel_ols(
    y_var: str,
    x_vars: list[str],
    df_panel: pd.DataFrame,
    cov: str = "twoway",
):
    """Run a two-way fixed-effects PanelOLS with safe dtype coercion.

    This is the main regression wrapper reused across the firm and inventor-year
    workflows. The implementation stays close to the original, but the docstring
    is expanded because this function is central to most saved outputs.
    """
    data = df_panel.copy().dropna(subset=[y_var, *x_vars])
    y = data[y_var]
    X = data.loc[:, x_vars].copy()

    bool_cols = [c for c in X.columns if X[c].dtype == bool]
    if bool_cols:
        X.loc[:, bool_cols] = X.loc[:, bool_cols].astype("float64")

    obj_cols = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if obj_cols:
        X.loc[:, obj_cols] = X.loc[:, obj_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    mod = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=True,
    )

    cov = (cov or "").lower()
    if cov == "robust":
        fit_kwargs = dict(cov_type="robust")
    elif cov == "entity":
        fit_kwargs = dict(cov_type="clustered", cluster_entity=True, cluster_time=False)
    elif cov == "time":
        fit_kwargs = dict(cov_type="clustered", cluster_entity=False, cluster_time=True)
    else:
        fit_kwargs = dict(cov_type="clustered", cluster_entity=True, cluster_time=True)

    return mod.fit(low_memory=True, **fit_kwargs)


def extract_r2(res) -> dict[str, float]:
    """Return all available R-squared measures from a linearmodels result."""
    def _f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    return {
        "r2_within": _f(getattr(res, "rsquared_within", None)),
        "r2_between": _f(getattr(res, "rsquared_between", None)),
        "r2_overall": _f(getattr(res, "rsquared_overall", None)),
        "r2": _f(getattr(res, "rsquared", None)),
    }


def save_regression_table(results, file_path: Path, key_var: str | None = None) -> None:
    """Save a compact HTML summary with stars and model-fit information."""
    summary_df = pd.DataFrame(
        {
            "Coefficient": results.params,
            "Std. Error": results.std_errors,
            "T-statistic": results.tstats,
            "P-value": results.pvalues,
        }
    )

    def _stars(p):
        if pd.isna(p):
            return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    summary_df.insert(1, "Stars", summary_df["P-value"].map(_stars))

    nobs = getattr(results, "nobs", None)
    try:
        nobs = int(nobs) if nobs is not None else None
    except Exception:
        pass

    r2 = extract_r2(results)
    fit_df = pd.DataFrame(
        {
            "Statistic": ["R2 (within)", "R2 (between)", "R2 (overall)", "R2", "Observations"],
            "Value": [r2["r2_within"], r2["r2_between"], r2["r2_overall"], r2["r2"], nobs],
        }
    )

    treat_header = ""
    if key_var is not None and key_var in results.params.index:
        coef = results.params[key_var]
        pval = results.pvalues[key_var]
        treat_header = (
            f"<p><b>Treatment ({key_var})</b>: coef = {coef:,.4f}, p = {pval:,.4f} {_stars(pval)}</p>\n"
        )

    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(treat_header)
        f.write("<h3>Coefficient Estimates</h3>\n")
        f.write(summary_df.to_html(float_format=lambda x: f"{x:,.4f}"))
        f.write("<br><h3>Model Fit</h3>\n")
        f.write(fit_df.to_html(index=False, float_format=lambda x: f"{x:,.4f}"))


def ensure_entity_in_columns(df: pd.DataFrame, entity_col: str) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    had_reset = False
    if entity_col not in out.columns:
        if isinstance(out.index, pd.MultiIndex) and entity_col in out.index.names:
            out = out.reset_index()
            had_reset = True
        elif out.index.name == entity_col:
            out = out.reset_index()
            had_reset = True
    return out, had_reset


def restore_index_if_needed(df: pd.DataFrame, original: pd.DataFrame, entity_col: str, time_col: str):
    if isinstance(original.index, pd.MultiIndex) and set([entity_col, time_col]).issubset(original.index.names):
        return add_panel_index(df, entity_col=entity_col, time_col=time_col)
    if original.index.name == entity_col and time_col in df.columns:
        return add_panel_index(df, entity_col=entity_col, time_col=time_col)
    return df


def attach_baseline_z(
    panel: pd.DataFrame,
    *,
    source_col: str,
    out_col: str,
    entity_col: str,
    time_col: str,
    rel_col: str = "event_time",
    baseline_k: int = -1,
    transform=None,
) -> pd.DataFrame:
    d, _ = ensure_entity_in_columns(panel, entity_col)
    if source_col not in d.columns:
        raise KeyError(f"{source_col} not found in panel.")

    base = d.loc[d[rel_col] == baseline_k, [entity_col, source_col]].drop_duplicates(subset=[entity_col]).copy()
    z = pd.to_numeric(base[source_col], errors="coerce")
    if transform is not None:
        z = transform(z)

    z_map = pd.Series(z.values, index=base[entity_col].values, name=out_col)
    d[out_col] = d[entity_col].map(z_map)
    return restore_index_if_needed(d, panel, entity_col, time_col)


def attach_deal_ratio_z(
    panel: pd.DataFrame,
    *,
    deal_col: str,
    denom_col: str,
    out_col: str,
    entity_col: str,
    time_col: str,
    rel_col: str = "event_time",
    baseline_k: int = -1,
    denom_is_log: bool = False,
) -> pd.DataFrame:
    d, _ = ensure_entity_in_columns(panel, entity_col)
    if deal_col not in d.columns:
        raise KeyError(f"{deal_col} not found in panel.")
    if denom_col not in d.columns:
        raise KeyError(f"{denom_col} not found in panel.")

    base = d.loc[d[rel_col] == baseline_k, [entity_col, deal_col, denom_col]].drop_duplicates(subset=[entity_col]).copy()
    deal = pd.to_numeric(base[deal_col], errors="coerce")
    denom = pd.to_numeric(base[denom_col], errors="coerce")
    if denom_is_log:
        denom = np.exp(denom)
    ratio = deal / denom.replace(0, np.nan)

    z_map = pd.Series(ratio.values, index=base[entity_col].values, name=out_col)
    d[out_col] = d[entity_col].map(z_map)
    return restore_index_if_needed(d, panel, entity_col, time_col)


def attach_quantile_z(
    panel: pd.DataFrame,
    *,
    source_col: str,
    out_col: str,
    n_bins: int,
    entity_col: str,
    time_col: str,
    rel_col: str = "event_time",
    baseline_k: int = -1,
    treated_col: str = "Treated",
    use_treated_for_cuts: bool = True,
) -> pd.DataFrame:
    d, _ = ensure_entity_in_columns(panel, entity_col)
    base = d.loc[d[rel_col] == baseline_k, [entity_col, source_col, treated_col]].drop_duplicates(subset=[entity_col]).copy()
    base[source_col] = pd.to_numeric(base[source_col], errors="coerce")

    fit_sample = base.copy()
    if use_treated_for_cuts and treated_col in fit_sample.columns:
        fit_sample = fit_sample[fit_sample[treated_col] == 1].copy()

    x_fit = fit_sample[source_col].dropna()
    if x_fit.empty:
        d[out_col] = np.nan
        return restore_index_if_needed(d, panel, entity_col, time_col)

    _, bins = pd.qcut(x_fit, q=n_bins, labels=False, retbins=True, duplicates="drop")
    base[out_col] = pd.cut(base[source_col], bins=bins, labels=False, include_lowest=True)

    z_map = pd.Series(base[out_col].values, index=base[entity_col].values, name=out_col)
    d[out_col] = d[entity_col].map(z_map)
    return restore_index_if_needed(d, panel, entity_col, time_col)


def add_all_z_specs(
    panel: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    log_sales_col: str,
    deal_col: str,
    mcap_col: str,
    mcap_is_log: bool = False,
    rel_col: str = "event_time",
) -> pd.DataFrame:
    """Add all firm-size and deal-size heterogeneity variants used in the project."""
    out = panel.copy()

    if log_sales_col in out.columns:
        out = attach_baseline_z(
            out,
            source_col=log_sales_col,
            out_col="Z_log_sales_cont",
            entity_col=entity_col,
            time_col=time_col,
            rel_col=rel_col,
            baseline_k=-1,
        )
        for q in [2, 3, 5]:
            out = attach_quantile_z(
                out,
                source_col="Z_log_sales_cont",
                out_col=f"Z_log_sales_q{q}",
                n_bins=q,
                entity_col=entity_col,
                time_col=time_col,
                rel_col=rel_col,
                baseline_k=-1,
                treated_col="Treated",
            )
    else:
        print(f"[Z] skip log-sales Z: missing column '{log_sales_col}'")

    if (deal_col in out.columns) and (mcap_col in out.columns):
        out = attach_deal_ratio_z(
            out,
            deal_col=deal_col,
            denom_col=mcap_col,
            out_col="Z_deal_rel_cont",
            entity_col=entity_col,
            time_col=time_col,
            rel_col=rel_col,
            baseline_k=-1,
            denom_is_log=mcap_is_log,
        )
        for q in [2, 3, 5]:
            out = attach_quantile_z(
                out,
                source_col="Z_deal_rel_cont",
                out_col=f"Z_deal_rel_q{q}",
                n_bins=q,
                entity_col=entity_col,
                time_col=time_col,
                rel_col=rel_col,
                baseline_k=-1,
                treated_col="Treated",
            )
    else:
        missing = [c for c in [deal_col, mcap_col] if c not in out.columns]
        print(f"[Z] skip deal-ratio Z: missing {missing}")

    return out
