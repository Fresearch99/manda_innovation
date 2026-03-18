"""
06_firm_fundamentals.py

Firm fundamentals construction from Compustat and related accounting/market variables.

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
# SECTION 12: FIRM-LEVEL FUNDAMENTALS DATA
#################################################################
print("--- Section 12: Building Firm-Level Fundamentals Data  ---")

def build_compustat_features(compustat_csv):
    # -----------------------------
    # Helper utilities
    # -----------------------------
    def ratio(n, d):
        """Safe division that returns NaN on 0/NaN denominators and cleans inf."""
        out = n / d
        return out.replace([np.inf, -np.inf], np.nan)

    def winsorize_inplace(s, lower=0.01, upper=0.99):
        """Clip a Series to [lower, upper] quantiles in place. No-op if too few non-nulls."""
        if s.notna().sum() < 50:
            return  # avoid unstable quantiles in tiny groups
        ql, qu = s.quantile([lower, upper])
        s.clip(ql, qu, inplace=True)

    # -----------------------------
    # 1) Load and basic sample prep
    # -----------------------------
    df = compustat_csv.copy()
    df["datadate_dt"] = pd.to_datetime(df["datadate"], format="%Y%m%d", errors="coerce")

    # Exclude financials (6xxx), utilities (49xx), and public sector (9xxx)
    sic_str = df["sic"].astype(str)
    mask_excl = sic_str.str.startswith(("6", "49", "9"))
    df = df.loc[~mask_excl].copy()

    # Keep standardized industrial consolidated domestic data
    fmt = (df["indfmt"] == "INDL") & (df["datafmt"] == "STD") & (df["popsrc"] == "D") & (df["consol"] == "C")
    df = df.loc[fmt].copy()

    # Drop negative core quantities (sales/assets/book equity)
    df = df.loc[~((df["sale"] < 0) | (df["at"] < 0) | (df["teq"] < 0))].copy()

    # Define analysis year: fiscal year ending Jun–Dec → same year, else previous year
    df["data_year"] = np.where(df["datadate_dt"].dt.month >= 6, df["datadate_dt"].dt.year, df["datadate_dt"].dt.year - 1)

    # Keep fiscal periods ~annual (9–15 months)
    if "pddur" in df.columns:
        df = df.loc[(df["pddur"] >= 9) & (df["pddur"] <= 15)].copy()

    # Keep the latest observation per gvkey–fyear
    df = (df.sort_values("datadate_dt", ascending=False)
            .drop_duplicates(["gvkey", "fyear"], keep="first")
            .copy())

    # -----------------------------
    # 2) Core identifiers & industry
    # -----------------------------
    # SIC2/3 for flexible fixed-effects
    def _to_str_prefix(x, k):
        s = str(x).split(".")[0]
        s = re.sub(r"^0*", "", s)  # strip leading zeros
        if s in ("", "nan"):
            return np.nan
        if len(s) == 3:            # if 3 digits, add a leading zero
            s = "0" + s
        return str(s[:k])
    
    df["sic4"] = df["sic"].apply(lambda x: _to_str_prefix(x, 4))
    df["sic3"] = df["sic"].apply(lambda x: _to_str_prefix(x, 3))
    df["sic2"] = df["sic"].apply(lambda x: _to_str_prefix(x, 2))

    # -----------------------------
    # 3) Clean obvious negatives & build market value
    # -----------------------------
    neg_to_nan = [
        "at","intan","sale","mkvalt","ppent","dltt","dlc","csho","prcc_f","xrd",
        "xad","xsga","xlr","xint","cogs","che","capx"
    ]
    for col in neg_to_nan:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Depreciation/book components frequently missing → fill minimal as in literature
    df["dp"]  = df["dp"].fillna(0)
    df["txdb"] = df["txdb"].fillna(0)
    df["re"]   = df["re"].fillna(0)

    # Shares adjusted & market value
    df["market_value"] = df["csho"] * df["prcc_f"]
    # Adjusted shares (better populated than adjex_c / adjex_f)
    df["csho_adj"] = df["csho"] * df.get("ajex", 1)

    # -----------------------------
    # 4) Lags for normalizations
    # -----------------------------
    df = df.sort_values(["gvkey", "datadate_dt"]).copy()
    for col in ["at", "sale", "ppent", "dltt", "dlc", "market_value"]:
        df[f"{col}_lag"] = df.groupby("gvkey")[col].shift(1)

    # -----------------------------
    # 5) CPI merge for real scaling (SA index needs real assets)
    # -----------------------------
    def fetch_cpi_from_fred_to_pickle(series_id="CPIAUCNS", out_dir=INTERMEDIATE_PATH):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        if "observation_date" in df:
            df["DATE"] = pd.to_datetime(df["observation_date"])
        # NEW: rename series to 'cpi'
        df.rename(columns={series_id: "cpi"}, inplace=True)
    
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{series_id}.pkl")
        df.to_pickle(out_path)
        print(f"{series_id} downloaded from FRED and saved as pickle: {out_path}", flush=True)
        return df
    
    cpi = fetch_cpi_from_fred_to_pickle()
    
    # Put dates at month-end (e.g., 2000-01-01 -> 2000-01-31)
    cpi["data_month_dt"] = (
        pd.to_datetime(cpi["DATE"], errors="coerce")
          .dt.to_period("M").dt.to_timestamp("M")
    )
    
    cpi = cpi[["data_month_dt", "cpi"]].copy()


    df["data_month_dt"] = df["datadate_dt"] + pd.offsets.MonthEnd(0)
    df = df.merge(cpi, on="data_month_dt", how="left")
    ref_cpi = cpi.loc[cpi["data_month_dt"] == pd.Timestamp("2004-12-31"), "cpi"]
    ref_cpi = ref_cpi.iloc[0] if not ref_cpi.empty else df["cpi"].dropna().median()

    df["at_real"]   = ratio(df["at"], df["cpi"]) * ref_cpi
    df["randd_real"] = ratio(df["xrd"], df["cpi"]) * ref_cpi
    df.loc[df["at_real"] == 0, "at_real"] = np.nan  # avoid zeros

    # -----------------------------
    # 6) Financing & payout (lean)
    # -----------------------------
    # Dividends: prefer dv, else dvc + dvp
    dv_fallback = (df.get("dvc", 0).fillna(0) + df.get("dvp", 0).fillna(0))
    df["Dividends"] = df.get("dv")
    df["Dividends"] = np.where(df["Dividends"].isna(), dv_fallback, df["Dividends"])
    df["Dividends"] = df["Dividends"].clip(lower=0)  # rule out negative data artifacts

    # Repurchases: simple definition using prstkc; clip negatives
    df["prstkc"] = df.get("prstkc", np.nan)
    df["sstk"]   = df.get("sstk",   np.nan)
    df["Repurchases"] = df["prstkc"].clip(lower=0)
    df["Net_Repurchases"] = (df["prstkc"].fillna(0) - df["sstk"].fillna(0)).clip(lower=0)

    df["Payout"]     = df["Dividends"] + df["Repurchases"]
    df["Net_Payout"] = df["Dividends"] + df["Net_Repurchases"]
    df["dividend_yield"] = ratio(df.get("dvc", 0).fillna(0) + df.get("dvp", 0).fillna(0),
                                 df["market_value"])
    # Repurchases scaled by lagged market value (Nguyen et al. 2020)
    df["repo"] = ratio(df["Repurchases"], df["market_value_lag"])

    # Asset-scaled net payout (concise summary measure)
    df["net_payout_at"] = ratio(df["Net_Payout"], df["at"])

    # -----------------------------
    # 7) Core fundamentals
    # -----------------------------
    df["log_at"] = np.log(df["at"])
    df["log_sale"] = np.log(df["sale"])
    df["log_mv"] = np.log(df["market_value"])

    # Profitability & margins
    df["roa"] = ratio(df.get("oibdp"), df["at_lag"])               # operating income before dep / AT_lag
    df["gross_profit"]  = df["sale"] - df.get("cogs")
    df["gross_margin"]  = ratio(df["gross_profit"], df["sale"])
    df["profit_margin"] = ratio(df.get("oiadp"), df["sale"])
    df["oi_at"]         = ratio(df.get("oibdp"), df["at"])         # operating profitability (Fama-French)

    # Growth & valuation
    df["sale_growth"]   = ratio(df["sale"] - df["sale_lag"], df["sale_lag"])
    df["tobinsq"]       = ratio(df["market_value"] + df["dltt"].fillna(0) + df["dlc"].fillna(0), df["at"])
    df["market_to_book"]= ratio(df["at"] + df["market_value"] - df.get("ceq"), df["at"])

    # Capital structure & funding
    df["leverage"] = ratio(df["dltt"].fillna(0) + df["dlc"].fillna(0), df["at"])
    df["net_debt_funding_rate"]   = ratio(df.get("dltis", 0).fillna(0) - df.get("dltr", 0).fillna(0) + df.get("dlcch", 0).fillna(0), df["at_lag"])
    df["net_equity_funding_rate"] = ratio(df.get("sstk", 0).fillna(0) - df.get("prstkc", 0).fillna(0), df["at_lag"])
    df["net_total_funding_rate"]  = df[["net_debt_funding_rate","net_equity_funding_rate"]].sum(axis=1, min_count=1)

    # Liquidity & cash
    df["cash"] = ratio(df["che"], df["at"])
    df["interest_coverage"] = ratio(df.get("oibdp"), df.get("xint"))
    df.loc[(df.get("xint", 0) > 0) & (df.get("oibdp", 0) < 0), "interest_coverage"] = 0  # cap negatives to 0 (common convention)

    # Investment & intangibles / tangibility
    df["investment"]           = ratio(df["capx"], df["at_lag"])
    df["acquisition"]          = ratio(df.get("aqc"), df["at"])
    df["randd_expenses"]       = ratio(df["xrd"].fillna(0), df["at_lag"])          # R&D intensity
    df["intangible_assets"]    = ratio(df["intan"].fillna(0), df["at"])            # share of intangibles
    df["tangibility_assets"]   = ratio(df.get("ppegt").fillna(0), df["at"])        # broader PPE
    df["fixed_assets"]         = ratio(df["ppent"].fillna(0), df["at"])

    # Working capital
    df["Net_working_capital"]  = df.get("wcap") - df["che"]
    df["net_working_capital"]  = ratio(df["Net_working_capital"], df["at"])

    # Cash flow metrics
    df["cash_flow"] = ratio(df.get("ib").fillna(0) + df["dp"].fillna(0) + df["xrd"].fillna(0), df["at_lag"])
    df["free_cash_flow"] = ratio(df.get("oibdp") - df.get("xint") - df.get("txt") - df["capx"], df["at_lag"])

    # Employment level
    df["Employment"] = df.get("emp")

    # -----------------------------
    # 8) Constraint indices (WW, SA)
    # -----------------------------
    # WW (Whited & Wu, 2006)
    df["CF"]    = ratio(df.get("ib") + df["dp"], df["at"])
    df["DIVPOS"]= ((df.get("dvc", 0).fillna(0) + df.get("dvp", 0).fillna(0)) > 0).astype(int)
    df["TLTD"]  = ratio(df["dltt"].fillna(0) + df["dlc"].fillna(0), df["at"])
    df["LNTA"]  = np.log(df["at"])
    df["SG"]    = df["sale_growth"]
    df["ISG"]   = df.groupby(["data_year", "sic3"])["SG"].transform("mean")

    df["ww_index"] = (-0.091 * df["CF"] - 0.062 * df["DIVPOS"] + 0.021 * df["TLTD"]
                      - 0.044 * df["LNTA"] + 0.102 * df["SG"] - 0.035 * df["ISG"])

    # SA (Hadlock & Pierce, 2010)
    df["ipo_date_dt"] = pd.to_datetime(df.get("ipodate"), format="%Y%m%d", errors="coerce")
    df["ipo_year"]    = df["ipo_date_dt"].dt.year

    # First year with non-missing price as fallback IPO proxy
    first_price_year = (df.loc[~df["prcc_f"].isna(), ["gvkey", "data_year"]]
                          .groupby("gvkey", as_index=False)["data_year"].min()
                          .rename(columns={"data_year":"first_non_missing_price"}))
    df = df.merge(first_price_year, on="gvkey", how="left")

    df["AGE"]  = np.where(~df["ipo_year"].isna(), df["data_year"] - df["ipo_year"],
                          df["data_year"] - df["first_non_missing_price"])
    df["SIZE"] = np.log(df["at_real"])

    # Truncation per paper
    df.loc[df["AGE"]  > 37, "AGE"]  = 37
    df.loc[df["SIZE"] > np.log(4500), "SIZE"] = np.log(4500)

    df["sa_index"] = -0.737*df["SIZE"] + 0.043*(df["SIZE"]**2) - 0.040*df["AGE"]

    # -----------------------------
    # 9) Cash-flow volatility (simple, rolling SD over 5 years)
    # -----------------------------
    df = df.sort_values(["gvkey", "data_year"]).copy()
    df["cf_volatility_rolling"] = (df.groupby("gvkey")["cash_flow"]
                                     .transform(lambda s: s.rolling(5, min_periods=5).std()))

    # -----------------------------
    # 10) Clean infinities and winsorize key ratios
    # -----------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    winsor_cols = [
        "leverage","roa","sale_growth","tobinsq","market_to_book","cash","investment",
        "intangible_assets","tangibility_assets","fixed_assets","net_working_capital",
        "gross_margin","profit_margin","free_cash_flow","cash_flow","interest_coverage",
        "net_debt_funding_rate","net_equity_funding_rate","net_total_funding_rate",
        "net_payout_at","dividend_yield","repo","oi_at"
    ]
    for c in winsor_cols:
        if c in df.columns:
            winsorize_inplace(df[c], lower=0.01, upper=0.99)

    # -----------------------------
    # 11) Final column selection
    # -----------------------------
    keep = [
        # IDs & dating
        "gvkey","datadate_dt","data_month_dt","data_year","fyear",
        # Industry
        "sic","sic2","sic3","sic4",
        # Prices/levels
        "market_value","csho_adj",
        # Size/valuation
        "log_at","log_sale","log_mv","tobinsq","market_to_book",
        # Structure & funding
        "leverage","net_debt_funding_rate","net_equity_funding_rate","net_total_funding_rate",
        # Profitability & growth
        "roa","gross_margin","profit_margin","oi_at","sale_growth",
        # Liquidity & cash
        "cash","interest_coverage","free_cash_flow","cash_flow",
        # Investment & composition
        "investment","acquisition","randd_expenses","randd_real",
        "intangible_assets","tangibility_assets","fixed_assets",
        # Working capital
        "net_working_capital",
        # Payouts
        "dividend_yield","net_payout_at","repo",
        # Employment
        "Employment",
        # Constraints
        "ww_index","sa_index","AGE",
        # Risk
        "cf_volatility_rolling"
    ]
    compustat_df = df[keep].copy()

    return compustat_df

# Load from WRDS data
compustat_csv = pd.read_csv(
    os.path.join(FINANCIAL_DATA_PATH, "comp__funda.csv"),
    low_memory=False
)
compustat_core_df = build_compustat_features(compustat_csv)

compustat_core_path = os.path.join(INTERMEDIATE_PATH, 'compustat_core.pkl')
compustat_core_df.to_pickle(compustat_core_path)

print("Compustat core features computed and saved as pickle.", flush=True)
print(f"[Compustat core] DataFrame size: {compustat_core_df.shape[0]:,} rows × {compustat_core_df.shape[1]:,} cols", flush=True)

# %%
#################################################################
# SECTION 13: ADD LINKTABLE TO MERGE FUNDAMENTALS AND PATENTS
#################################################################
print("--- Section 13: Building Linktable and Merge with Fundamentals  ---")

# --- 1) Load & clean the CRSP–Compustat linktable ----------------------------
LINKTABLE_CSV = r'/Users/dominikjurek/Library/CloudStorage/Dropbox/University/PhD Berkeley/Research/Alice Project/Patent Portfolio and Economic Data/Patent Portfolio Source Data/linktable.csv'

usecols = [
    'gvkey', 'LPERMNO', 'LPERMCO', 'cusip', 'sic', 'naics',
    'LINKDT', 'LINKENDDT', 'tic', 'LINKTYPE', 'LINKPRIM'
]
linktable = pd.read_csv(LINKTABLE_CSV, usecols=usecols, low_memory=False)

# Keep primary, non-duplicate links (WRDS guidance)
linktable = linktable[
    linktable['LINKTYPE'].isin(['LU', 'LC']) &
    linktable['LINKPRIM'].isin(['P', 'C'])
].copy()

# Parse link start/end; fill open-ended links
linktable['linkdt']    = pd.to_datetime(linktable['LINKDT'],    format='%Y%m%d', errors='coerce')
linktable['linkenddt'] = pd.to_datetime(linktable['LINKENDDT'], format='%Y%m%d', errors='coerce')
linktable['linkenddt'] = linktable['linkenddt'].fillna(pd.Timestamp('2021-10-01'))

# Numeric IDs
linktable['permno'] = pd.to_numeric(linktable['LPERMNO'], downcast='integer', errors='coerce')
linktable['permco'] = pd.to_numeric(linktable['LPERMCO'], downcast='integer', errors='coerce')
linktable['gvkey']  = pd.to_numeric(linktable['gvkey'],   downcast='integer', errors='coerce')

# SIC prefixes (2/3/4-digit); treat non-numeric robustly
sic_str = linktable['sic'].astype(str).str.split('.').str[0].str.replace(r'\D', '', regex=True)
linktable['sic4'] = pd.to_numeric(sic_str.str[:4], errors='coerce')
linktable['sic3'] = pd.to_numeric(sic_str.str[:3], errors='coerce')
linktable['sic2'] = pd.to_numeric(sic_str.str[:2], errors='coerce')

# Final linktable columns
linktable = (
    linktable[['gvkey', 'permno', 'permco', 'cusip', 'sic', 'sic2', 'sic3', 'sic4',
               'naics', 'tic', 'linkdt', 'linkenddt']]
    .dropna(subset=['permno'])
    .drop_duplicates()
    .reset_index(drop=True)
)

# --- 2) Merge linktable into Compustat core on valid dates --------------------
# Merge permco by gvkey, then keep rows where datadate is within link window
compustat_core_linked = compustat_core_df.merge(
    linktable[['gvkey', 'permco', 'linkdt', 'linkenddt']],
    on='gvkey', how='inner'
)
compustat_core_linked = compustat_core_linked[
    (compustat_core_linked['datadate_dt'] >= compustat_core_linked['linkdt']) &
    (compustat_core_linked['datadate_dt'] <= compustat_core_linked['linkenddt'])
].drop(columns=['linkdt', 'linkenddt'])

# De-duplicate (rare collisions across link ranges)
compustat_core_linked = compustat_core_linked.drop_duplicates(subset=['permco', 'data_year']).reset_index(drop=True)

# --- 3) Save ------------------------------------------------------------------
out_path = os.path.join(INTERMEDIATE_PATH, 'compustat_core_linked.pkl')
compustat_core_linked.to_pickle(out_path)
print(f"Linktable built, merged with Compustat core, and saved: {out_path}", flush=True)
print(f"[Compustat] DataFrame size: {compustat_core_linked.shape[0]:,} rows × {compustat_core_linked.shape[1]:,} cols", flush=True)

