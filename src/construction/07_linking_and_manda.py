"""
07_linking_and_manda.py

Linktable merge logic and M&A panel construction, including pre-deal technology similarity.

This file was created by splitting the original uploaded construction script
into topical modules. The code below stays intentionally close to the source
so that a line-by-line audit against the original remains easy.
"""

# NOTE:
# The code below preserves the original imperative construction style.
# It is therefore best read as a section file that mirrors the original
# notebook-style pipeline, rather than as a fully re-engineered library.

# %%
#################################################################
# SECTION 14: M&A DATA STRUCTURING
#################################################################
print("--- Section 14: Building M&A Data  ---")

def build_manda_panel(manda_csv, linktable):
    # -----------------------------
    # 0) Utilities
    # -----------------------------
    def _parse_yyyymmdd(s):
        """Parse integers/strings like 20190115 to pandas Timestamp; return NaT if invalid."""
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    def _coerce_dt(col):
        """Ensure link dates are datetime; allow ints/strings; fill open-ended linkenddt."""
        out = pd.to_datetime(col, errors="coerce")
        return out

    # -----------------------------
    # 1) Clean basic fields
    # -----------------------------
    df = manda_csv.copy()
    
    if "Date Announced Numeric" not in df.columns and "Date Announced" in df.columns:
        df["Date Announced Numeric"] = pd.to_datetime(df["Date Announced"], errors="coerce").dt.strftime("%Y%m%d")
    if ("Date Effective Numeric" not in df.columns) and ("Date Effective" in df.columns):
        df["Date Effective Numeric"] = pd.to_datetime(df["Date Effective"], errors="coerce").dt.strftime("%Y%m%d")
 

    # Parse dates
    df["date_announced_dt"] = _parse_yyyymmdd(df.get("Date Announced Numeric"))
    df["date_effective_dt"] = _parse_yyyymmdd(df.get("Date Effective Numeric"))

    # Truncate CUSIPs to 6 (link on CUSIP-6)
    df["target_cusip6"]   = df.get("Target CUSIP", "").astype(str).str.upper().str.slice(0, 6)
    df["acquiror_cusip6"] = df.get("Acquiror CUSIP", "").astype(str).str.upper().str.slice(0, 6)

    # Simple diversifying flags from SDC-provided SIC digests (fallback to NaN if absent)
    df["diversifying_sic3"] = (df.get("Acquiror_SIC_3digits") != df.get("Target_SIC_3digits")).astype("Int64")
    df["diversifying_sic2"] = (df.get("Acquiror_SIC_2digits") != df.get("Target_SIC_2digits")).astype("Int64")

    # Transaction value (million USD)
    df["transaction_value_musd"] = pd.to_numeric(df.get("Value of Transaction ($mil)"), errors="coerce")
    
    # Clean and rename % columns (0–100 Float64)
    def _parse_pct_col(df_, src_col, out_col):
        """Create out_col as a cleaned 0–100 Float64 from src_col (extract first number, coerce, clip; all-NA if missing)."""
        s = df_.get(src_col)
        if s is None:
            df_[out_col] = pd.Series(pd.NA, index=df_.index, dtype="Float64"); return
        cleaned = (
            s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
             .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )
        df_[out_col] = pd.to_numeric(cleaned, errors="coerce").clip(0, 100).astype("Float64")

    for _src, _out in {
        "% of Shares Acq.": "pct_shares_acq_acquiror",
        "% Owned After Trans- action": "pct_owned_after_acquiror",
        "% sought": "pct_sought_acquiror",
    }.items():
        _parse_pct_col(df, _src, _out)


    # Deal outcome: keep Completed/Withdrawn only
    status_clean = df.get("Status", "").astype(str).str.strip().str.title()
    keep = status_clean.isin(["Completed", "Withdrawn"])
    df = df.loc[keep].copy()
    df["deal_outcome"] = status_clean
    df["failed_merger"] = (df["deal_outcome"] == "Withdrawn").astype(int)

    # Year (announcement-based)
    df["announcement_year"] = df["date_announced_dt"].dt.year
    
    # -----------------------------
    # 2) Prepare link table
    # -----------------------------
    lt = linktable.copy()
    # Normalize CUSIP and derive cusip6
    lt["cusip"]  = lt["cusip"].astype(str).str.upper()
    lt["cusip6"] = lt["cusip"].str.slice(0, 6)

    # Ensure datetime and open-ended link ranges
    lt["linkdt"]    = _coerce_dt(lt["linkdt"])
    lt["linkenddt"] = _coerce_dt(lt["linkenddt"])
    lt["linkenddt"] = lt["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    # -----------------------------
    # 3) Link BOTH sides (inner joins) on valid windows
    # -----------------------------
    keep_cols = ["gvkey","naics","permno","permco","sic4","sic3","sic2","cusip6","linkdt","linkenddt"]

    # Link target
    tgt = df.merge(
        lt[keep_cols].rename(columns={c: f"{c}_target" for c in keep_cols}),
        left_on="target_cusip6", right_on="cusip6_target", how="inner"
    )

    # Link acquirer
    both = tgt.merge(
        lt[keep_cols].rename(columns={c: f"{c}_acquiror" for c in keep_cols}),
        left_on="acquiror_cusip6", right_on="cusip6_acquiror", how="inner"
    )

    # Keep observations where announcement date lies within BOTH link ranges
    t = both["date_announced_dt"]
    in_win = (
        (both["linkdt_target"]    <= t) & (t <= both["linkenddt_target"]) &
        (both["linkdt_acquiror"]  <= t) & (t <= both["linkenddt_acquiror"])
    )
    both = both.loc[in_win].copy()

    # Distinct parties (avoid self-mergers; permco different)
    both = both.loc[both["permco_target"] != both["permco_acquiror"]].copy()

    # -----------------------------
    # 4) De-duplicate: last announcement per dyad-year
    # -----------------------------
    both = (both.sort_values("date_announced_dt", ascending=False)
                 .drop_duplicates(["permco_target","permco_acquiror","announcement_year"], keep="first")
                 .copy())

    # -----------------------------
    # 5) Final tidy selection
    # -----------------------------
    keep_final = [
        # Dates & keys
        "date_announced_dt","date_effective_dt","announcement_year","deal_outcome","failed_merger",
        # Transaction size
        "transaction_value_musd",
        # Transaction ownership percentages (acquiror-side)
        "pct_shares_acq_acquiror","pct_owned_after_acquiror","pct_sought_acquiror",
        # SDC diversifying indicators
        "diversifying_sic2","diversifying_sic3",
        # Target IDs & industry
        "gvkey_target","permno_target","permco_target","cusip6_target","naics_target",
        "sic2_target","sic3_target","sic4_target",
        # Acquirer IDs & industry
        "gvkey_acquiror","permno_acquiror","permco_acquiror","cusip6_acquiror","naics_acquiror",
        "sic2_acquiror","sic3_acquiror","sic4_acquiror",
    ]
    manda_df = both[keep_final].copy()

    return manda_df


def build_predeal_tech_similarity(pat_inv_firm_df, manda_df, window_years=5, tech_col='cpc_subclass'):
    """
    Build pre-deal technology similarity between target and acquiror based on
    their patent portfolios in the window [announcement_year - window_years + 1,
    announcement_year], using CPC subclasses as technology classes.

    Reuses the global `_calculate_cosine_similarity` helper and Counter-based
    vectors from Section 11.
    """
    print("\tBuilding pre-deal technology similarity (target vs acquiror)...", flush=True)

    # Restrict to firm-year-tech cells with valid permco and tech class
    cols = ['permco', 'filing_year', tech_col, 'patent_id']
    pat_df = pat_inv_firm_df.loc[:, cols].dropna(subset=['permco', tech_col]).copy()

    # Firm × year × tech → unique patent count
    counts = (
        pat_df
        .groupby(['permco', 'filing_year', tech_col])['patent_id']
        .nunique()
    )

    # Build {(permco, year): Counter(tech -> count)} dictionary
    firm_year_vectors = defaultdict(Counter)
    for (firm_id, year, tech), cnt in counts.items():
        # ensure year is int for safe range comparisons
        firm_year_vectors[(firm_id, int(year))][tech] = cnt

    # Prepare M&A deals subset with valid IDs and years
    deals = (
        manda_df
        .dropna(subset=['permco_target', 'permco_acquiror', 'announcement_year'])
        .copy()
    )

    results = []

    def _aggregate_vector(permco, years):
        """Aggregate firm-year-tech counters over a set of years into one Counter."""
        agg = Counter()
        if pd.isna(permco):
            return agg
        for y in years:
            agg.update(firm_year_vectors.get((permco, y), Counter()))
        return agg

    print("\tComputing cosine similarity for each MandA deal...", flush=True)
    for row in tqdm(deals.itertuples(index=False), total=len(deals)):
        year = int(row.announcement_year)
        window_years_range = range(year - window_years + 1, year + 1)

        tgt_vec = _aggregate_vector(row.permco_target, window_years_range)
        acq_vec = _aggregate_vector(row.permco_acquiror, window_years_range)

        sim = _calculate_cosine_similarity(tgt_vec, acq_vec)

        results.append({
            'permco_target': row.permco_target,
            'permco_acquiror': row.permco_acquiror,
            'announcement_year': year,
            'predeal_tech_similarity_cpc_subclass': sim,
        })

    predeal_df = pd.DataFrame(results).drop_duplicates()
    print(f"\tPre-deal similarity computed for {len(predeal_df)} deals.", flush=True)
    return predeal_df


# Stacked SDC MandA files from 1985 to 2019
manda_csv = pd.read_csv(
    os.path.join(MANDA_DATA_PATH, "MandA_combined_1985_2019.csv"),
    low_memory=False
)
manda_df = build_manda_panel(manda_csv, linktable)

# --- attach pre-deal technology similarity (target vs acquiror) ---
predeal_sim_df = build_predeal_tech_similarity(pat_inv_firm_df, manda_df, window_years=5, tech_col='cpc_subclass')
manda_df = manda_df.merge(
    predeal_sim_df,
    on=['permco_target', 'permco_acquiror', 'announcement_year'],
    how='left'
)

manda_path = os.path.join(INTERMEDIATE_PATH, 'manda.pkl')
manda_df.to_pickle(manda_path)

print("MandA features computed and saved as pickle.", flush=True)
print(f"[MandA] DataFrame size: {manda_df.shape[0]:,} rows × {manda_df.shape[1]:,} cols", flush=True)

