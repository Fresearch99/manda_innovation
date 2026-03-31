from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csdid.att_gt import ATTgt

from analysis.sections.advanced_methods import run_borusyak_jaravel, run_sun_abraham
from analysis.sections.firm_analysis import run_did_and_event_study_for_all_outcomes
from analysis.sections.utils import add_all_z_specs, add_panel_index, ensure_entity_in_columns, restore_index_if_needed


def csdid_attgt_table(att: ATTgt) -> pd.DataFrame:
    att = att.summ_attgt()
    res = getattr(att, "results", None)
    if not isinstance(res, dict):
        raise RuntimeError("ATTgt.summ_attgt() did not expose a dict-like `results`.")
    df = pd.DataFrame(res).rename(columns={"group": "gname", "year": "data_year", "post ": "post", "g": "gname", "t": "data_year"})
    for req in ["gname", "data_year", "att"]:
        if req not in df.columns:
            raise RuntimeError(f"Missing `{req}` in att.summ_attgt().results")
    for opt in ["post", "se", "l_se", "u_se", "sig"]:
        if opt not in df.columns:
            df[opt] = np.nan
    for c in ["gname", "data_year", "att", "se", "l_se", "u_se"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    try:
        df["sig"] = df["sig"].astype(bool)
    except Exception:
        df["sig"] = df["sig"].astype(float).fillna(0).astype(int).astype(bool)
    try:
        df["post"] = df["post"].astype(bool)
    except Exception:
        df["post"] = df["post"].astype(float).fillna(0).astype(int).astype(bool)
    df["e"] = df["data_year"] - df["gname"]
    df.loc[df["gname"] == 0, "e"] = np.nan
    return df.sort_values(["gname", "data_year"]).reset_index(drop=True)


def _trim_unidentified_cells_fast(
    d: pd.DataFrame,
    outcome: str,
    id_col: str,
    t_col: str,
    g_col: str,
    min_treated_per_cell: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    X = d.copy()
    X[t_col] = pd.to_numeric(X[t_col], errors="coerce")
    X[g_col] = pd.to_numeric(X[g_col], errors="coerce").fillna(0)
    X = X.dropna(subset=[id_col, t_col, g_col, outcome]).copy()
    X[t_col] = X[t_col].astype(int)
    X[g_col] = X[g_col].astype(int)

    t = X[t_col]
    g = X[g_col]
    is_treated_row = (g != 0) & (t >= g)
    is_control_row = (g == 0) | (g > t)

    tr = X.loc[is_treated_row & X[outcome].notna(), [g_col, t_col]].groupby([g_col, t_col]).size().rename("n_treat").reset_index()
    ctrl = X.loc[is_control_row & X[outcome].notna(), [t_col]].groupby(t_col).size().rename("n_ctrl").reset_index()
    grid = tr.merge(ctrl, on=t_col, how="left").fillna({"n_ctrl": 0})
    good = grid[(grid["n_treat"] >= min_treated_per_cell) & (grid["n_ctrl"] > 0)]
    if good.empty:
        return X.iloc[0:0].copy()

    good_pairs = good[[g_col, t_col]].drop_duplicates()
    treated_rows = X.loc[is_treated_row, [g_col, t_col]].copy()
    treated_rows["_row_id"] = treated_rows.index
    treated_good = treated_rows.merge(good_pairs, on=[g_col, t_col], how="inner")["_row_id"].to_numpy()
    keep_idx = X.index[is_control_row].union(pd.Index(treated_good))
    Y = X.loc[keep_idx].copy()
    if verbose:
        print(f"  -> Identification trim: dropped {len(X) - len(Y):,} rows; {len(good_pairs):,} identified (g,t) cells remain.")
    return Y


def _fit_attgt_point(d, outcome, id_col, t_col, g_col, xformla, est_method):
    att = ATTgt(
        yname=outcome,
        tname=t_col,
        idname=id_col,
        gname=g_col,
        xformla=xformla,
        data=d,
        panel=True,
        allow_unbalanced_panel=True,
        control_group=["notyettreated", "nevertreated"],
        cband=False,
    ).fit(est_method=est_method, base_period="varying", bstrap=False)
    tbl = csdid_attgt_table(att)
    mask_post = (tbl["gname"].fillna(0) != 0) & (tbl["data_year"] >= tbl["gname"])
    vals = tbl.loc[mask_post, "att"]
    if vals.isna().all():
        vals = tbl.loc[(tbl["gname"].fillna(0) != 0), "att"]
    overall = float(np.nanmean(vals.to_numpy()))
    return att, tbl, overall


def run_csdid_inventor_year(
    df_full: pd.DataFrame,
    outcome: str,
    role_tag: str,
    *,
    role: str = "all",
    strict_role: bool = True,
    covariates: list[str] | None = None,
    id_col: str = "inventor_id",
    t_col: str = "data_year",
    out_dir: Path = Path("."),
    plot_dir: Path = Path("."),
    min_cohort_size: int = 200,
    E_PRE: int = 2,
    E_MAX: int = 3,
    B: int = 99,
    seed: int = 42,
    verbose: bool = True,
):
    """Run CSDID on the inventor-year panel."""
    d0 = df_full.copy()
    role_l = str(role).lower()
    g_map = {"all": "cs_g_year_all", "target": "cs_g_year_target", "acquiror": "cs_g_year_acquiror"}
    g_src = g_map.get(role_l)
    if g_src is None:
        raise ValueError(f"role must be one of {list(g_map.keys())}, got '{role}'")

    needed_base = [id_col, t_col, outcome, g_src]
    if g_src != "cs_g_year_all" and "cs_g_year_all" in d0.columns:
        needed_base.append("cs_g_year_all")
    use_covs = [c for c in (covariates or []) if c in d0.columns]
    needed = list(dict.fromkeys(needed_base + use_covs))
    d = d0.loc[:, needed].replace([np.inf, -np.inf], np.nan).copy()
    d[t_col] = pd.to_numeric(d[t_col], errors="coerce")
    d[g_src] = pd.to_numeric(d[g_src], errors="coerce")
    d = d.dropna(subset=[id_col, t_col]).copy()
    d[t_col] = d[t_col].astype(int)
    d["gname"] = pd.to_numeric(d[g_src], errors="coerce").fillna(0).astype(int)

    if role_l != "all" and strict_role:
        if "cs_g_year_all" not in d0.columns:
            raise RuntimeError("strict_role=True requires cs_g_year_all in the panel.")
        all_g = pd.to_numeric(d0["cs_g_year_all"], errors="coerce")
        role_g = pd.to_numeric(d0[g_src], errors="coerce")
        treated_ok = (role_g.notna()) & (all_g.notna()) & (role_g.astype("Int64") == all_g.astype("Int64"))
        treated_ids = set(d0.loc[treated_ok, id_col].dropna().unique())
        control_ids = set(d0.loc[all_g.isna(), id_col].dropna().unique())
        keep_ids = treated_ids.union(control_ids)
        d = d[d[id_col].isin(keep_ids)].copy()
        d["gname"] = pd.to_numeric(d[g_src], errors="coerce").fillna(0).astype(int)

    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    for c in use_covs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[outcome]).sort_values([id_col, t_col]).drop_duplicates([id_col, t_col]).reset_index(drop=True)
    if d.empty or (d["gname"] != 0).sum() == 0:
        return None

    small = d.loc[d["gname"] != 0].groupby("gname")[id_col].nunique()
    small = set(small.index[small < int(min_cohort_size)])
    if small:
        d = d[(~d["gname"].isin(small)) | (d["gname"] == 0)].copy()
    if (d["gname"] != 0).sum() == 0:
        return None

    g_nonzero = d.loc[d["gname"] != 0, "gname"]
    t_min = int(g_nonzero.min() - int(E_PRE))
    t_max = int(g_nonzero.max() + int(E_MAX))
    d = d[d[t_col].between(t_min, t_max)].copy()
    d["e"] = d[t_col] - d["gname"]
    d = d[(d["gname"] == 0) | ((d["gname"] != 0) & d["e"].between(-int(E_PRE), int(E_MAX)))].copy()
    d = d.drop(columns=["e"], errors="ignore")

    has_ctrl_by_t = d.groupby(t_col)["gname"].apply(lambda s: (s.eq(0) | (s > s.name)).any())
    d = d[d[t_col].isin(has_ctrl_by_t.index[has_ctrl_by_t])].copy()
    d = _trim_unidentified_cells_fast(d, outcome, id_col, t_col, "gname", min_treated_per_cell=1, verbose=verbose)
    if d.empty or (d["gname"] != 0).sum() == 0:
        return None

    xformla_cov = f"{outcome} ~ " + (" + ".join(use_covs) if use_covs else "1")
    specs = [("dr", xformla_cov), ("ipw", "~ 1")]
    att = tbl = None
    overall = np.nan
    method_used = xformla_used = None
    for method, xformla in specs:
        try:
            att, tbl, overall = _fit_attgt_point(d, outcome, id_col, t_col, "gname", xformla, method)
            if int(tbl["att"].notna().sum()) > 0:
                method_used, xformla_used = method, xformla
                break
        except Exception:
            continue
    if tbl is None or tbl["att"].notna().sum() == 0:
        raise RuntimeError("All csdid fits failed/returned NaN.")

    tbl2 = tbl.copy()
    tbl2["e"] = tbl2["data_year"] - tbl2["gname"]
    tbl2["e"] = pd.to_numeric(tbl2["e"], errors="coerce").astype(int)
    evt = tbl2.loc[(tbl2["gname"] != 0) & tbl2["att"].notna(), ["gname", "e", "att"]]
    cohort_w = d.loc[d["gname"] != 0].groupby("gname")[id_col].nunique().rename("w").reset_index()
    evt = evt.merge(cohort_w, on="gname", how="left").fillna({"w": 0})
    dyn_tbl = evt.groupby("e").apply(lambda x: np.average(x["att"].to_numpy(), weights=x["w"].to_numpy())).rename("att").reset_index()
    dyn_tbl["se"] = np.nan
    e_grid = dyn_tbl["e"].dropna().sort_values().unique().astype(int)
    overall_dyn = float(np.nanmean(dyn_tbl.loc[dyn_tbl["e"].between(0, int(E_MAX)), "att"].to_numpy()))
    ci = (np.nan, np.nan)

    if int(B) > 0:
        rng = np.random.default_rng(seed)
        id_to_g = d[[id_col, "gname"]].drop_duplicates(subset=[id_col]).copy()
        id_to_g = id_to_g[id_to_g["gname"] != 0].copy()
        if not id_to_g.empty:
            ids = id_to_g[id_col].to_numpy()
            g_for_id = id_to_g["gname"].to_numpy()
            evt0 = evt[["gname", "e", "att"]].copy()
            dyn_boot = {}
            boots_evt = []

            def _wavg(att, w):
                ws = np.sum(w)
                return np.nan if ws <= 0 else float(np.average(att, weights=w))

            for _ in range(int(B)):
                draw_idx = rng.integers(0, len(ids), size=len(ids))
                g_draw = g_for_id[draw_idx]
                uniq_g, counts = np.unique(g_draw, return_counts=True)
                cohort_w_b = pd.DataFrame({"gname": uniq_g.astype(int), "w": counts.astype(float)})
                evt_b = evt0.merge(cohort_w_b, on="gname", how="left").fillna({"w": 0.0})
                dyn_b = evt_b.groupby("e").apply(lambda x: _wavg(x["att"].to_numpy(), x["w"].to_numpy())).rename("att").reset_index()
                dyn_b = pd.DataFrame({"e": e_grid}).merge(dyn_b, on="e", how="left")
                for e_val, a_val in zip(dyn_b["e"].to_numpy(), dyn_b["att"].to_numpy()):
                    if np.isfinite(a_val):
                        dyn_boot.setdefault(float(e_val), []).append(float(a_val))
                post = dyn_b.loc[dyn_b["e"].between(0, int(E_MAX)), "att"].to_numpy()
                if post.size > 0 and np.isfinite(post).any():
                    boots_evt.append(float(np.nanmean(post)))

            dyn_tbl["se"] = dyn_tbl["e"].map(lambda e: np.nan if len(dyn_boot.get(float(e), [])) < 2 else float(np.nanstd(np.array(dyn_boot[float(e)], dtype=float), ddof=1)))
            dyn_tbl["lo"] = dyn_tbl["e"].map(lambda e: np.nan if len(dyn_boot.get(float(e), [])) == 0 else float(np.nanpercentile(np.array(dyn_boot[float(e)], dtype=float), 2.5)))
            dyn_tbl["hi"] = dyn_tbl["e"].map(lambda e: np.nan if len(dyn_boot.get(float(e), [])) == 0 else float(np.nanpercentile(np.array(dyn_boot[float(e)], dtype=float), 97.5)))
            if len(boots_evt) > 0:
                lo, hi = np.percentile(np.array(boots_evt, dtype=float), [2.5, 97.5])
                ci = (float(lo), float(hi))

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    tbl_out = tbl.copy()
    tbl_out["e"] = pd.to_numeric(tbl_out["data_year"] - tbl_out["gname"], errors="coerce")
    tbl_out = tbl_out[(tbl_out["gname"] != 0) & (tbl_out["e"].between(-int(E_PRE), int(E_MAX)))]
    tbl_out.to_csv(out_dir / f"csdid_inv_{role_tag}_{outcome}_attgt.csv", index=False)
    dyn_tbl["overall_att_post_mean"] = float(overall_dyn)
    dyn_tbl["overall_att_post_cell_mean"] = float(overall)
    dyn_tbl["overall_ci_lo"] = float(ci[0]) if np.isfinite(ci[0]) else np.nan
    dyn_tbl["overall_ci_hi"] = float(ci[1]) if np.isfinite(ci[1]) else np.nan
    dyn_tbl.to_csv(out_dir / f"csdid_inv_{role_tag}_{outcome}_dynamic.csv", index=False)

    if dyn_tbl["e"].notna().any():
        plt.figure(figsize=(9, 5.5))
        m = dyn_tbl[["lo", "hi"]].notna().all(axis=1) if {"lo", "hi"}.issubset(dyn_tbl.columns) else pd.Series(False, index=dyn_tbl.index)
        if m.any():
            y = dyn_tbl.loc[m, "att"].to_numpy()
            yerr = np.vstack([y - dyn_tbl.loc[m, "lo"].to_numpy(), dyn_tbl.loc[m, "hi"].to_numpy() - y])
            plt.errorbar(dyn_tbl.loc[m, "e"], y, yerr=yerr, fmt="o", capsize=3)
        else:
            plt.plot(dyn_tbl["e"], dyn_tbl["att"], "o")
        plt.axhline(0, linestyle="--")
        plt.title(f"CSDID (Inventor-year): {outcome.replace('_', ' ').title()}")
        plt.suptitle(role_tag, y=0.95)
        plt.xlabel("Event time (t - g)")
        plt.ylabel("ATT")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"csdid_inv_{role_tag}_{outcome}_plot.png", dpi=200, bbox_inches="tight")
        plt.close()

    return {"overall": {"att": float(overall_dyn), "ci": ci}, "attgt": tbl, "dynamic": dyn_tbl, "clean_data": d, "spec": {"method": method_used, "xformla": xformla_used, "g_src": g_src, "strict_role": strict_role}}


def attach_within_firm_upper_half_z(
    panel: pd.DataFrame,
    *,
    source_col: str,
    out_col: str,
    entity_col: str,
    time_col: str,
    firm_col: str = "permco_event",
    rel_col: str = "event_time",
    baseline_k: int = -1,
    min_firm_inventors: int = 2,
) -> pd.DataFrame:
    d, _ = ensure_entity_in_columns(panel, entity_col)
    needed = [entity_col, time_col, firm_col, rel_col, source_col]
    miss = [c for c in needed if c not in d.columns]
    if miss:
        raise KeyError(f"attach_within_firm_upper_half_z missing columns: {miss}")
    base = d.loc[d[rel_col] == baseline_k, [entity_col, time_col, firm_col, source_col]].copy()
    base[source_col] = pd.to_numeric(base[source_col], errors="coerce")
    base = base.dropna(subset=[firm_col, time_col, source_col]).copy()
    if base.empty:
        d[out_col] = np.nan
        return restore_index_if_needed(d, panel, entity_col, time_col)
    grp = [firm_col, time_col]
    base["_n_firm_year"] = base.groupby(grp)[entity_col].transform("nunique")
    base["_pct"] = base.groupby(grp)[source_col].rank(method="average", pct=True)
    base[out_col] = np.where(base["_n_firm_year"] >= int(min_firm_inventors), (base["_pct"] >= 0.5).astype("float"), np.nan)
    z_map = base[[entity_col, out_col]].drop_duplicates(subset=[entity_col]).set_index(entity_col)[out_col]
    d[out_col] = d[entity_col].map(z_map)
    return restore_index_if_needed(d, panel, entity_col, time_col)


def prepare_inventor_event_panel_for_did_role_vs_control(
    inventor_ma_event_panel: pd.DataFrame,
    role: str,
    window: tuple[int, int],
    outcomes: list[str] | None = None,
    controls: list[str] | None = None,
    nearest_only: bool = False,
    add_within_firm_rank_z: bool = True,
    min_firm_inventors_for_rank: int = 2,
):
    """Prepare the inventor-year event-study panel used in the active analysis loop."""
    lo, hi = window
    role_l = str(role).lower()
    df = inventor_ma_event_panel.copy()
    req = ["inventor_id", "data_year", "years_from_ma_deal", "closest_deal_year", "ma_deal_role"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"inventor_ma_event_panel missing required columns: {missing}")

    df["years_from_ma_deal"] = pd.to_numeric(df["years_from_ma_deal"], errors="coerce")
    df = df[df["years_from_ma_deal"].between(lo, hi)].copy()
    df["ma_deal_role"] = df["ma_deal_role"].astype(str).str.lower()
    if "is_control_event" not in df.columns:
        raise ValueError("No 'is_control_event' column found.")
    df = df[(df["ma_deal_role"] == role_l) | (df["is_control_event"] == 1)].copy()
    if nearest_only and "control_rank" in df.columns:
        df = df[(df["is_control_event"] == 0) | (df["control_rank"] == 1)].copy()
    if "permco_event" not in df.columns and "permco_assigned" in df.columns:
        df = df.rename(columns={"permco_assigned": "permco_event"})

    parts = ["inventor_id", "permco_event", "closest_deal_year", "is_control_event"]
    if "ma_other_party" in df.columns:
        parts.append("ma_other_party")
    if "treated_inventor_id" in df.columns:
        parts.append("treated_inventor_id")
    if "control_rank" in df.columns:
        parts.append("control_rank")
    df["inv_event_id"] = df[parts].astype(str).agg("|".join, axis=1)

    df["Treated"] = (df["ma_deal_role"] == role_l).astype(int)
    df["event_time"] = pd.to_numeric(df["years_from_ma_deal"], errors="coerce").astype(float)
    df["cohort"] = pd.to_numeric(df["closest_deal_year"], errors="coerce").astype(float)
    df["Post"] = (df["event_time"] >= 0).astype(int)
    df["Post_Treated"] = df["Post"] * df["Treated"]
    df["data_year"] = pd.to_numeric(df["data_year"], errors="coerce")
    df = df.dropna(subset=["data_year"]).copy()
    df["data_year"] = df["data_year"].astype(int)

    cols_to_cast = []
    if outcomes is not None:
        cols_to_cast += [c for c in outcomes if c in df.columns]
    if controls is not None:
        cols_to_cast += [c for c in controls if c in df.columns]
    for c in cols_to_cast:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ["Treated", "Post", "Post_Treated"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    if add_within_firm_rank_z:
        if "cum_patents" in df.columns:
            df = attach_within_firm_upper_half_z(
                df,
                source_col="cum_patents",
                out_col="Z_upper_half_cum_patents_within_firm",
                entity_col="inv_event_id",
                time_col="data_year",
                firm_col="permco_event",
                rel_col="event_time",
                baseline_k=-1,
                min_firm_inventors=min_firm_inventors_for_rank,
            )
        if "cum_cites" in df.columns:
            df = attach_within_firm_upper_half_z(
                df,
                source_col="cum_cites",
                out_col="Z_upper_half_cum_cites_within_firm",
                entity_col="inv_event_id",
                time_col="data_year",
                firm_col="permco_event",
                rel_col="event_time",
                baseline_k=-1,
                min_firm_inventors=min_firm_inventors_for_rank,
            )
    return add_panel_index(df, entity_col="inv_event_id", time_col="data_year")


def downsample_units_for_advanced(
    inv_es: pd.DataFrame,
    *,
    role: str,
    max_treated_units: int,
    max_control_units: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample inventor-event units while preserving the full event-time path."""
    df = inv_es.reset_index() if isinstance(inv_es.index, pd.MultiIndex) else inv_es.copy()
    required = ["inventor_id", "closest_deal_year", "permco_event", "ma_deal_role", "ma_other_party"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"downsample_units_for_advanced missing columns: {missing}")
    unit_cols = ["inventor_id", "closest_deal_year", "permco_event", "ma_deal_role", "ma_other_party"]
    units = df.loc[:, unit_cols].drop_duplicates()
    treated_units = units[units["ma_deal_role"].astype(str) == str(role)]
    control_units = units[units["ma_deal_role"].astype(str) == "control"]
    rs = np.random.RandomState(seed)
    keep_t = treated_units.sample(n=min(max_treated_units, len(treated_units)), replace=False, random_state=rs) if len(treated_units) else treated_units
    keep_c = control_units.sample(n=min(max_control_units, len(control_units)), replace=False, random_state=rs) if len(control_units) else control_units
    keep_units = pd.concat([keep_t, keep_c], ignore_index=True).drop_duplicates()
    out = df.merge(keep_units, on=unit_cols, how="inner")
    return add_panel_index(out, entity_col="inv_event_id", time_col="data_year")


def run_significant_inv_year_advanced(
    inv_es: pd.DataFrame,
    sig_df: pd.DataFrame,
    role_tag: str,
    controls: list[str],
    out_dir: Path,
    plot_dir: Path,
    alpha: float,
    window: tuple[int, int],
    ref_k: int,
    bjs_n_boot: int = 100,
    run_bjs: bool = False,
):
    """Run inventor-year advanced methods on significant baseline outcomes only."""
    if sig_df is None or sig_df.empty or (not {"outcome", "p_value"}.issubset(sig_df.columns)):
        return
    sig_outcomes = sig_df.loc[sig_df["p_value"] < alpha, "outcome"].dropna().tolist()
    sig_outcomes = [y for y in sig_outcomes if y in inv_es.columns]
    if not sig_outcomes:
        return
    covs = [c for c in (controls or []) if c in inv_es.columns]
    for y in sig_outcomes:
        run_sun_abraham(
            stack=inv_es,
            outcome=y,
            role_tag=role_tag,
            controls=covs,
            window=window,
            ref_k=ref_k,
            out_dir=out_dir,
            plot_dir=plot_dir,
            entity_col="inv_event_id",
            time_col="data_year",
        )
        if run_bjs:
            run_borusyak_jaravel(
                df_full=inv_es,
                outcome=y,
                role_tag=role_tag,
                controls=covs,
                window=window,
                n_boot=bjs_n_boot,
                out_dir=out_dir,
                plot_dir=plot_dir,
                entity_col="inv_event_id",
                time_col="data_year",
                treat_col="Post_Treated",
                boot_method="wild_fixed",
            )


def summarize_inv_es(df: pd.DataFrame, role: str, window=None, label: str = "") -> None:
    """Print diagnostics for the inventor-year event-study panel."""
    es = df
    unit_cols = []
    for c in ["inventor_id", "closest_deal_year", "permco_event", "deal_id", "event_id"]:
        if c in es.columns:
            unit_cols.append(c)
    if "inventor_id" in unit_cols:
        preferred = ["inventor_id"]
        if "closest_deal_year" in es.columns:
            preferred.append("closest_deal_year")
        if "permco_event" in es.columns:
            preferred.append("permco_event")
        unit_cols = preferred

    if "Treated" in es.columns:
        treated_flag = es["Treated"].astype(int) == 1
    elif "ma_deal_role" in es.columns:
        treated_flag = es["ma_deal_role"].astype(str) == str(role)
    else:
        treated_flag = None

    rel_col = next((c for c in ["rel_year", "event_time", "k", "tau", "event_k"] if c in es.columns), None)
    print(f"\n--- Panel summary: {label or role} ---")
    print(f"Rows: {es.shape[0]:,} | Cols: {es.shape[1]:,}")
    if "inventor_id" in es.columns:
        print(f"Unique inventors: {es['inventor_id'].nunique():,}")
    if unit_cols and all(c in es.columns for c in unit_cols):
        print(f"Unique units ({'+'.join(unit_cols)}): {es.drop_duplicates(unit_cols).shape[0]:,}")
    if treated_flag is not None:
        t_rows = int(treated_flag.sum())
        c_rows = int((~treated_flag).sum())
        print(f"Treated rows: {t_rows:,} | Control rows: {c_rows:,} | Treated share: {t_rows/(t_rows+c_rows+1e-12):.4f}")
        if unit_cols and all(c in es.columns for c in unit_cols):
            tmp = es.loc[:, unit_cols].copy()
            tmp["_treated"] = treated_flag.values
            u = tmp.groupby(unit_cols, as_index=False)["_treated"].max()
            t_u = int(u["_treated"].sum())
            c_u = int((~(u["_treated"].astype(bool))).sum())
            print(f"Treated units: {t_u:,} | Control units: {c_u:,} | Treated unit share: {t_u/(t_u+c_u+1e-12):.4f}")
    if rel_col is not None and unit_cols and all(c in es.columns for c in unit_cols):
        lo, hi = window if window is not None else (int(es[rel_col].min()), int(es[rel_col].max()))
        expected = hi - lo + 1
        g = es.groupby(unit_cols)[rel_col].nunique()
        bad = (g != expected).sum()
        print(f"Balance check using {rel_col} over [{lo},{hi}]: expected {expected} per unit | unbalanced units: {int(bad):,} ({bad/len(g):.2%})")
    try:
        mem_mb = es.memory_usage(deep=True).sum() / (1024**2)
        print(f"Approx df memory: {mem_mb:,.1f} MB")
    except Exception:
        pass
