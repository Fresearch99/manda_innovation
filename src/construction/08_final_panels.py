"""
08_final_panels.py

Final analysis panels, including the firm-year panel, M&A event-study panel, inventor-year panel for inventors ever at M&A firms, and the inventor × M&A event-study panel. The unused standalone inventor move panel block from original Section 15.4 has been removed here.

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
# SECTION 15: CREATE FINAL ANALYSIS PANELS
#################################################################
print("--- Section 15: Creating Final Analysis Panels ---")

# ---------------------------------------------------------------
# 15.1. Build Final Firm-Year Analysis Panel
# ---------------------------------------------------------------
print("\tBuilding the final Firm-Year analysis panel...")
firm_year_patent_metrics = pat_inv_firm_df.groupby(['permco', 'filing_year']).agg(
    total_patents=('patent_id', 'nunique'),
    cites=('cites', 'sum'),
    xi_real=('xi_real', 'sum'),
    avg_novelty=('novelty_score_group', 'mean'),
    backward_cites=('backward_citations', 'sum'),
    self_cites=('self_citations', 'sum'),
    top1_patents=('top1_forward_citations', 'sum'),
    top10_to_2_patents=('top10_to_2_forward_citations', 'sum'), 
    cited_patents=('cited_patent_forward_citations', 'sum'),   
    uncited_patents=('Uncited_patent_forward_citations', 'sum'), 
    exploration_firm=('exploration_firm', 'mean'),
    exploitation_firm=('exploitation_firm', 'mean'),
    num_inventors=('inventor_id', 'nunique')
).reset_index().rename(columns={'filing_year': 'data_year'})

firm_year_panel_enriched = pd.merge(
    firm_year_patent_metrics,
    firm_year_similarity_panel,
    how='left',
    left_on=['permco', 'data_year'],
    right_on=['permco', 'year']
)

# --- Adding Inventor Mobility Metrics to the Firm-Year Panel ---
print("\nAggregating inventor mobility data to the firm-year level...")
# --- 1. Aggregate metrics for DEPARTING inventors for each firm-year ---
# We group by the firm they left ('from_permco') and the year of the move.
# We aggregate their pre-move performance to measure the talent that was lost.
departures_by_firm_year = inventor_perf_wide.groupby(['from_permco', 'movement_year']).agg(
    departing_inventors_count=('inventor_id', 'nunique'),
    sum_patents_pre_move_departures=('patent_id_pre_move', 'sum'),
    sum_cites_pre_move_departures=('cites_pre_move', 'sum'),
    sum_xi_real_pre_move_departures=('xi_real_pre_move', 'sum'),
    avg_novelty_pre_move_departures=('novelty_score_group_pre_move', 'mean'),
    avg_exploration_pre_move_departures=('exploration_inv_pre_move', 'mean')
).reset_index()

# --- 2. Aggregate metrics for ARRIVING inventors for each firm-year ---
# We group by the firm they joined ('to_permco') and the year of the move.
# We aggregate their pre-move stats (talent gained) and their change in performance (integration success).
arrivals_by_firm_year = inventor_perf_wide.groupby(['to_permco', 'movement_year']).agg(
    arriving_inventors_count=('inventor_id', 'nunique'),
    sum_patents_pre_move_arrivals=('patent_id_pre_move', 'sum'),
    sum_cites_pre_move_arrivals=('cites_pre_move', 'sum'),
    avg_change_in_patents_arrivals=('change_patent_id', 'mean'),
    avg_change_in_cites_arrivals=('change_cites', 'mean'),
    avg_change_in_novelty_arrivals=('change_novelty_score_group', 'mean'),
    avg_change_in_exploration_arrivals=('change_exploration_inv', 'mean')
).reset_index()

# --- 3. Merge Compustat Financial Data ---
# This create the full firm-year panel.
print("\tMerging Compustat data...")
firm_year_panel_enriched = compustat_core_linked.merge(
    firm_year_panel_enriched,
    on=['permco', 'data_year'], 
    how='left'
)

# --- 4. Merge DEPARTURES data into the firm-year panel ---
# We use a left merge to keep all firm-years, even those with no departures.
firm_year_panel_enriched = firm_year_panel_enriched.merge(
    departures_by_firm_year.rename(columns={
        'from_permco': 'permco', 
        'movement_year': 'data_year'
    }),
    on=['permco', 'data_year'], 
    how='left'
)

# --- 5. Merge ARRIVALS data into the firm-year panel ---
# Again, use a left merge.
firm_year_panel_enriched = firm_year_panel_enriched.merge(
    arrivals_by_firm_year.rename(columns={
        'to_permco': 'permco', 
        'movement_year': 'data_year'
    }),
    on=['permco', 'data_year'], 
    how='left'
)

# --- 6. Post-merge cleanup ---
# The left merges will create NaNs for firm-years with no mobility or patenting events. 
# For counts and sums, it's appropriate to fill these with 0.
# Averages of changes can be left as NaN if no one moved.
cols_to_fill_zero = [
    'total_patents', 'cites', 'xi_real', 'backward_cites', 'self_cites', 
    'top1_patents', 'top10_to_2_patents', 'cited_patents', 'uncited_patents', 'num_inventors'
    'departing_inventors_count', 'sum_patents_pre_move_departures', 'sum_cites_pre_move_departures', 'sum_xi_real_pre_move_departures',
    'arriving_inventors_count', 'sum_patents_pre_move_arrivals', 'sum_cites_pre_move_arrivals'
]
for col in cols_to_fill_zero:
    if col in firm_year_panel_enriched.columns:
        firm_year_panel_enriched[col] = firm_year_panel_enriched[col].fillna(0)

print("Inventor mobility metrics successfully merged into the firm-year panel.")
print(f"[Firm-year panel enriched] size: {firm_year_panel_enriched.shape[0]:,} rows × {firm_year_panel_enriched.shape[1]:,} cols", flush=True)


# --- Adding RELATIVE Inventor Performance to the Firm-Year Panel ---
print("\nAggregating relative inventor performance (star power) to the firm-year level...")

# --- 1. Calculate the average relative quality of DEPARTING inventors ---
# This measures the average "star power" a firm loses in a given year.
departing_talent_relative = mover_benchmark_df.groupby(['from_permco', 'movement_year'])[[
    'quality_top1_vs_origin_peers', 'quality_top10_vs_origin_peers',
    'novelty_vs_origin_peers', 'exploration_vs_origin_peers',
    'num_patents_vs_origin_peers', 'citations_total_vs_origin_peers'    
]].mean().reset_index()

# --- 2. Calculate the average relative quality of ARRIVING inventors ---
# This measures the average "star power" a firm gains in a given year.
arriving_talent_relative = mover_benchmark_df.groupby(['to_permco', 'movement_year'])[[
    'quality_top1_vs_dest_peers', 'quality_top10_vs_dest_peers',
    'novelty_vs_dest_peers', 'exploration_vs_dest_peers',
    'num_patents_vs_dest_peers', 'citations_total_vs_dest_peers'    
]].mean().reset_index()

# --- 3. Merge DEPARTING talent data into the firm-year panel ---
firm_year_panel_enriched = firm_year_panel_enriched.merge(
    departing_talent_relative.rename(columns={
        'from_permco': 'permco', 
        'movement_year': 'data_year',
        'quality_top1_vs_origin_peers': 'avg_rel_top1_quality_departures',
        'quality_top10_vs_origin_peers': 'avg_rel_top10_quality_departures',
        'novelty_vs_origin_peers': 'avg_rel_novelty_departures',
        'exploration_vs_origin_peers': 'avg_rel_exploration_departures',
        'num_patents_vs_origin_peers': 'avg_rel_patents_departures',
        'citations_total_vs_origin_peers': 'avg_rel_cites_departures'
    }),
    on=['permco', 'data_year'], 
    how='left'
)

# --- 4. Merge ARRIVING talent data into the firm-year panel ---
firm_year_panel_enriched = firm_year_panel_enriched.merge(
    arriving_talent_relative.rename(columns={
        'to_permco': 'permco', 
        'movement_year': 'data_year',
        'quality_top1_vs_dest_peers': 'avg_rel_top1_quality_arrivals',
        'quality_top10_vs_dest_peers': 'avg_rel_top10_quality_arrivals',
        'novelty_vs_dest_peers': 'avg_rel_novelty_arrivals',
        'exploration_vs_dest_peers': 'avg_rel_exploration_arrivals',
        'num_patents_vs_dest_peers': 'avg_rel_patents_arrivals',
        'citations_total_vs_dest_peers': 'avg_rel_cites_arrivals'
    }),
    on=['permco', 'data_year'], 
    how='left'
)

print("Relative inventor performance metrics successfully merged into the firm-year panel.")
print(f"[Firm-year panel enriched] size: {firm_year_panel_enriched.shape[0]:,} rows × {firm_year_panel_enriched.shape[1]:,} cols", flush=True)

# --- Assembling the Final Firm-Year Panel ---
print("\nAssembling the final firm-year panel by merging external datasets...")

# Start with the enriched patent and mobility panel we've built
final_firm_panel = firm_year_panel_enriched.copy()


# --- 1. Process and Merge M&A Event Data ---
# This adds flags for firms acting as acquirors or being targeted in a given year.
print("\tProcessing and merging M&A data...")
# a) Aggregate acquisition events with detailed metrics
acquiror_events = manda_df.groupby(['permco_acquiror', 'announcement_year']).agg(
    n_acquisitions=('permco_acquiror', 'size'),
    n_failed_acquisitions=('failed_merger', 'sum'),
    n_diversifying_sic2_acq=('diversifying_sic2', 'sum'),
    n_diversifying_sic3_acq=('diversifying_sic3', 'sum'),
    total_deal_value_acq_musd=('transaction_value_musd', 'sum'),
    avg_deal_value_acq_musd=('transaction_value_musd', 'mean')
).reset_index()

# b) Aggregate target events with detailed metrics
target_events = manda_df.groupby(['permco_target', 'announcement_year']).agg(
    n_times_targeted=('permco_target', 'size'),
    n_failed_targeted_deals=('failed_merger', 'sum'),
    total_deal_value_target_musd=('transaction_value_musd', 'sum'),
    avg_deal_value_target_musd=('transaction_value_musd', 'mean')
).reset_index()

# c) Merge acquiror counts and metrics into the main panel
final_firm_panel = final_firm_panel.merge(
    acquiror_events, 
    left_on=['permco', 'data_year'], 
    right_on=['permco_acquiror', 'announcement_year'], 
    how='left'
)

# d) Merge target counts and metrics into the main panel
final_firm_panel = final_firm_panel.merge(
    target_events, 
    left_on=['permco', 'data_year'], 
    right_on=['permco_target', 'announcement_year'], 
    how='left'
)
  
# --- 2. Final Cleaning and Type Conversion ---
print("\tPerforming final cleaning on the panel...")

# a) Drop redundant key columns created by the merges
cols_to_drop = [
    'permco_acquiror', 'announcement_year_x', 
    'permco_target', 'announcement_year_y',
    'year' # Assuming these might still exist from previous merges
]
final_firm_panel.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# b) Safely fill NaNs only for event counts
# This is crucial: filling all NaNs with 0 would incorrectly change missing
# financial data (e.g., from Compustat) into zeros. We only fill counts.
count_cols = [
    'n_acquisitions', 'n_times_targeted',
    # We can also include the mobility counts from the previous step here
    'departing_inventors_count', 'arriving_inventors_count'
]
for col in count_cols:
    if col in final_firm_panel.columns:
        final_firm_panel[col] = final_firm_panel[col].fillna(0).astype(int)

# --- 4. Save the Final Panel ---
output_filename = os.path.join(OUTPUT_PATH, 'firm_year_panel.pkl')
final_firm_panel.to_pickle(output_filename)

print(f"Panel saved to: {output_filename}")
print(f"[Final firm-year panel] size: {final_firm_panel.shape[0]:,} rows × {final_firm_panel.shape[1]:,} cols", flush=True)

# ---------------------------------------------------------------
# 15.3. Creating an M&A Event-Study Panel
# ---------------------------------------------------------------
print("\nTransforming the firm-year panel into an M&A event-study panel...")

# --- Step 1: Prepare a Unified M&A Event DataFrame ---
acquiror_events = manda_df[[
    'permco_acquiror',
    'permco_target',
    'announcement_year',
    'transaction_value_musd',
    'failed_merger',
    'predeal_tech_similarity_cpc_subclass'  
]].copy()
acquiror_events['deal_role'] = 'acquiror'
acquiror_events.rename(
    columns={'permco_acquiror': 'permco', 'permco_target': 'other_party_permco'},
    inplace=True
)

target_events = manda_df[[
    'permco_target',
    'permco_acquiror',
    'announcement_year',
    'transaction_value_musd',
    'failed_merger',
    'predeal_tech_similarity_cpc_subclass'  
]].copy()
target_events['deal_role'] = 'target'
target_events.rename(
    columns={'permco_target': 'permco', 'permco_acquiror': 'other_party_permco'},
    inplace=True
)

all_deal_events = pd.concat([acquiror_events, target_events], ignore_index=True)
all_deal_events.rename(
    columns={
        'announcement_year': 'deal_year',
        'transaction_value_musd': 'deal_value',
        'predeal_tech_similarity_cpc_subclass': 'predeal_tech_similarity'  
    },
    inplace=True
)
all_deal_events.dropna(subset=['permco', 'deal_year'], inplace=True)

# (keep sorted for deterministic behavior)
all_deal_events.sort_values(['permco', 'deal_year'], inplace=True)

# --- Step 2: Find the Closest M&A Event for Each Firm-Year (no merge_asof) ---

# Harmonize dtypes & sort
panel_sorted = final_firm_panel.copy()
panel_sorted['permco']    = pd.to_numeric(panel_sorted['permco'], errors='coerce')
panel_sorted['data_year'] = pd.to_numeric(panel_sorted['data_year'], errors='coerce')
panel_sorted = panel_sorted.dropna(subset=['permco','data_year'])
panel_sorted[['permco','data_year']] = panel_sorted[['permco','data_year']].astype('int64')
panel_sorted = panel_sorted.sort_values(['permco','data_year'])

all_deal_events['permco']    = pd.to_numeric(all_deal_events['permco'], errors='coerce')
all_deal_events['deal_year'] = pd.to_numeric(all_deal_events['deal_year'], errors='coerce')
all_deal_events = all_deal_events.dropna(subset=['permco','deal_year'])
all_deal_events[['permco','deal_year']] = all_deal_events[['permco','deal_year']].astype('int64')
all_deal_events = all_deal_events.sort_values(['permco','deal_year'])

# Deal columns we want to attach for prev/next
deal_cols = [
    'deal_year',
    'deal_value',
    'deal_role',
    'other_party_permco',
    'failed_merger',
    'predeal_tech_similarity'  
]

parts = []
for pco, g_panel in panel_sorted.groupby('permco', sort=True):
    g_deals = all_deal_events.loc[all_deal_events['permco'] == pco, deal_cols]
    if g_deals.empty:
        # No deals for this permco: add empty prev/next columns
        prev_empty = pd.DataFrame({f"{c}_prev_deal": pd.NA for c in deal_cols}, index=g_panel.index)
        next_empty = pd.DataFrame({f"{c}_next_deal": pd.NA for c in deal_cols}, index=g_panel.index)
        parts.append(pd.concat([g_panel, prev_empty, next_empty], axis=1))
        continue

    g_deals = g_deals.reset_index(drop=True)
    y_panel = g_panel['data_year'].to_numpy()
    y_deals = g_deals['deal_year'].to_numpy()

    # Index of last deal_year <= data_year (previous)
    prev_idx = np.searchsorted(y_deals, y_panel, side='right') - 1
    prev_mask = prev_idx >= 0
    prev_idx_safe = prev_idx.copy()
    prev_idx_safe[~prev_mask] = 0  # will mask invalid rows to NA later

    # Index of first deal_year >= data_year (next)
    next_idx = np.searchsorted(y_deals, y_panel, side='left')
    next_mask = next_idx < len(y_deals)
    next_idx_safe = next_idx.copy()
    next_idx_safe[~next_mask] = 0

    # Pick rows and align indices to panel rows
    prev_rows = g_deals.iloc[prev_idx_safe].set_axis(g_panel.index)
    next_rows = g_deals.iloc[next_idx_safe].set_axis(g_panel.index)

    # Mask invalids to NA (keeps dtypes flexible)
    prev_rows = prev_rows.where(np.broadcast_to(prev_mask[:, None], prev_rows.shape))
    next_rows = next_rows.where(np.broadcast_to(next_mask[:, None], next_rows.shape))

    # Suffix columns to match your Step 3 expectations
    prev_rows = prev_rows.add_suffix('_prev_deal')
    next_rows = next_rows.add_suffix('_next_deal')

    parts.append(pd.concat([g_panel, prev_rows, next_rows], axis=1))

# This replaces the output you'd get after the two merge_asof calls
panel_with_deals = pd.concat(parts, axis=0).sort_values(['permco','data_year']).reset_index(drop=True)

# --- Step 3: Determine the Closest Deal and Calculate Relative Period ---
panel_with_deals['diff_prev'] = (panel_with_deals['data_year'] - panel_with_deals['deal_year_prev_deal']).abs()
panel_with_deals['diff_next'] = (panel_with_deals['deal_year_next_deal'] - panel_with_deals['data_year']).abs()

is_next_closer = panel_with_deals['diff_next'] < panel_with_deals['diff_prev']
is_prev_null = panel_with_deals['diff_prev'].isnull()
use_next_deal = is_next_closer | is_prev_null

panel_with_deals['closest_deal_year'] = np.where(
    use_next_deal,
    panel_with_deals['deal_year_next_deal'],
    panel_with_deals['deal_year_prev_deal']
)
panel_with_deals['ma_deal_role']  = np.where(
    use_next_deal,
    panel_with_deals['deal_role_next_deal'],
    panel_with_deals['deal_role_prev_deal']
)
panel_with_deals['ma_deal_value'] = np.where(
    use_next_deal,
    panel_with_deals['deal_value_next_deal'],
    panel_with_deals['deal_value_prev_deal']
)
panel_with_deals['ma_other_party'] = np.where(
    use_next_deal,
    panel_with_deals['other_party_permco_next_deal'],
    panel_with_deals['other_party_permco_prev_deal']
)
panel_with_deals['ma_failed_merger'] = np.where(
    use_next_deal,
    panel_with_deals['failed_merger_next_deal'],
    panel_with_deals['failed_merger_prev_deal']
)
panel_with_deals['ma_predeal_tech_similarity'] = np.where(        
    use_next_deal,
    panel_with_deals['predeal_tech_similarity_next_deal'],
    panel_with_deals['predeal_tech_similarity_prev_deal']
)

panel_with_deals['years_from_ma_deal'] = panel_with_deals['data_year'] - panel_with_deals['closest_deal_year']

# --- Step 4: Apply Window and Clean Up ---
window_years = 5
outside_window = panel_with_deals['years_from_ma_deal'].abs() > window_years
ma_cols = [
    'ma_deal_role',
    'ma_deal_value',
    'ma_other_party',
    'years_from_ma_deal',
    'ma_failed_merger',
    'ma_predeal_tech_similarity'  
]
for col in ma_cols:
    panel_with_deals.loc[outside_window, col] = np.nan


cols_to_drop = [col for col in panel_with_deals.columns if '_prev_deal' in str(col) or '_next_deal' in str(col)]
cols_to_drop += ['diff_prev', 'diff_next', 'closest_deal_year']
panel_with_deals.drop(columns=cols_to_drop, inplace=True, errors='ignore')

panel_with_deals['ma_deal_role'].fillna('no_recent_MandA', inplace=True)

final_event_panel = panel_with_deals.sort_values(['permco', 'data_year']).reset_index(drop=True)

# --- Step 5: Save the Final Event-Study Panel ---
output_filename = os.path.join(OUTPUT_PATH, 'final_firm_year_ma_event_study_panel.pkl')
final_event_panel.to_pickle(output_filename)

print(f"\nFinal event-study panel successfully saved to: {output_filename}")
print("\nEvent-study panel creation complete.")
print(f"[Final M&A event-study panel] size: {final_event_panel.shape[0]:,} rows × {final_event_panel.shape[1]:,} cols", flush=True)


# ---------------------------------------------------------------

r'''
# %%
# =====================================================================
# FAST-RERUN GATE: load intermediates and skip heavy upstream sections
# =====================================================================
START_AT_SECTION = 15.5   # set to 0 for full run; set to 15.5 to rerun only from here

# Use strings + os.path.join instead of Path
INTER_DIR = os.path.join(OUTPUT_PATH, "intermediate_files")
os.makedirs(INTER_DIR, exist_ok=True)

def load_pickle(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache for {name}: {path}\nRun full pipeline once to create it.")
    return pd.read_pickle(path)

if START_AT_SECTION >= 15.5:
    pat_inv_firm_df   = load_pickle(os.path.join(INTER_DIR, "pat_inv_firm_df_fully_enriched.pkl"), "pat_inv_firm_df")
    mover_events_df   = load_pickle(os.path.join(INTER_DIR, "mover_events_df.pkl"), "mover_events_df")
    manda_df          = load_pickle(os.path.join(INTER_DIR, "manda.pkl"), "manda_df")
    final_event_panel = load_pickle(os.path.join(OUTPUT_PATH, "final_firm_year_ma_event_study_panel.pkl"), "final_event_panel")

    print("[gate] Loaded cached objects; running from Section 15.5+ only.")
r'''

# %%
#################################################################
# SECTION 15.5: INVENTOR-YEAR PANEL FOR INVENTORS EVER AT M&A FIRMS
#################################################################
print("\n--- Section 15.5: Building inventor-year panel for inventors ever at M&A firms ---")

def build_inventor_year_panel_ma_inventors(
    pat_inv_firm_df: pd.DataFrame,
    manda_df: pd.DataFrame,
    mover_events_df: pd.DataFrame,
    final_event_panel: pd.DataFrame,
    out_dir: str,
    window_fill_years: tuple[int, int] | None = None,  # optional override of year range
):
    """
    Build an inventor-year panel for inventors who have ever patented at a firm that is
    an acquiror or target in manda_df + inventors from control firms.

    Panel is complete over each inventor's observed career span (or optional override),
    and includes years with zero patenting (zero-filled for sums/counts).

    Adds:
      - permco_assigned (annual employer proxy; inferred for missing years)
      - is_move_year (binary for strict move-year from mover_events_df)
      - M&A context variables merged from final_event_panel at (permco_assigned, data_year)
    """

    # ----------------------------
    # 1) Identify study firms + inventor sample (treated + never-treated controls)
    # ----------------------------
    # Firms in analysis universe: those present in final_event_panel (Compustat-linked universe)
    study_permcos = pd.to_numeric(final_event_panel["permco"], errors="coerce")
    study_permcos = study_permcos.dropna().astype("int64").unique()
    
    # Firms that ever have an M&A deal (treated firms), from manda_df
    treated_permcos = pd.Index(pd.concat([
        manda_df["permco_target"].dropna(),
        manda_df["permco_acquiror"].dropna()
    ], ignore_index=True)).astype("int64").unique()
    treated_permcos_set = set(treated_permcos)
    
    # Inventors who ever have a patent with assignee permco in the analysis universe
    base = pat_inv_firm_df.copy()
    base["filing_year"] = pd.to_numeric(base["filing_year"], errors="coerce").astype("Int64")
    base["permco"] = pd.to_numeric(base["permco"], errors="coerce")
    
    inv_ids = base.loc[base["permco"].isin(study_permcos), "inventor_id"].dropna().unique()
    inv_base = base.loc[base["inventor_id"].isin(inv_ids)].copy()
    
    print(f"\tInventors with any Compustat-linked firm assignee: {len(inv_ids):,}", flush=True)
    print(f"\tFiltered pat_inv_firm_df rows for these inventors: {inv_base.shape[0]:,}", flush=True)
    
    # --- Inventor-level static features (CPC + first filing year) ---
    def _mode_or_nan(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        vc = s.value_counts()
        top = vc[vc == vc.max()].index
        return sorted(map(str, top))[0]  # deterministic tie-break
    
    print("\tBuilding inventor static features...", flush=True)
    
    inv_static = inv_base[["inventor_id"]].dropna().drop_duplicates().copy()
    
    # first_filing_year (use provided if available; else fallback to min filing_year)
    if "first_filing_year" in inv_base.columns:
        tmp_ffy = inv_base[["inventor_id", "first_filing_year"]].copy()
        tmp_ffy["first_filing_year"] = pd.to_numeric(tmp_ffy["first_filing_year"], errors="coerce")
        tmp_ffy = tmp_ffy.dropna(subset=["inventor_id", "first_filing_year"])
        tmp_ffy = tmp_ffy.groupby("inventor_id")["first_filing_year"].min().reset_index()
    else:
        tmp_ffy = (
            inv_base.dropna(subset=["inventor_id", "filing_year"])
                   .groupby("inventor_id")["filing_year"]
                   .min()
                   .rename("first_filing_year")
                   .reset_index()
        )
    
    inv_static = inv_static.merge(tmp_ffy, on="inventor_id", how="left")
    
    for c in ["first_cpc_subclass", "first_cpc_group"]:
        if c in inv_base.columns:
            tmp_c = inv_base.groupby("inventor_id")[c].apply(_mode_or_nan).rename(c).reset_index()
            inv_static = inv_static.merge(tmp_c, on="inventor_id", how="left")

    # Inventor-level ever-treated flag based on ever patenting at a treated firm (ever target/acquiror firm)
    inv_treated_flag = (
        inv_base.assign(_is_treated_firm_patent=inv_base["permco"].isin(treated_permcos_set).astype("int32"))
               .groupby("inventor_id")["_is_treated_firm_patent"].max()
               .rename("ever_treated_inv")
               .reset_index()
    )
    print(f"\tComputed inventor ever-treated flag for {inv_treated_flag.shape[0]:,} inventors", flush=True)

    # ----------------------------
    # 2) Annual aggregates (same “spirit” as your move-performance agg_metrics)
    # ----------------------------
    # sums / counts
    agg_sumcount = {
        "patent_id": "nunique",
        "cites": "sum",
        "xi_real": "sum",
        "backward_citations": "sum",
        "self_citations": "sum",
        "top1_forward_citations": "sum",
        "top10_to_2_forward_citations": "sum",
        "cited_patent_forward_citations": "sum",
        "Uncited_patent_forward_citations": "sum",
        "same_first_cpc_subclass": "sum",
    }
    # means (leave NaN for zero-patent years)
    agg_means = {
        "novelty_score_group": "mean",
        "exploration_inv": "mean",
        "exploitation_inv": "mean",
        "team_size": "mean",
        "nca_enforce_score": "mean",
    }
    agg = {**agg_sumcount, **agg_means}

    print("\tComputing inventor-year aggregates...", flush=True)
    inv_year_metrics = (
        inv_base
        .dropna(subset=["inventor_id", "filing_year"])
        .groupby(["inventor_id", "filing_year"], as_index=False)
        .agg(agg)
        .rename(columns={"filing_year": "data_year", "patent_id": "total_patents"})
    )
    print(f"\tBuilt inventor-year metrics: {inv_year_metrics.shape[0]:,} inventor-years", flush=True)

    # ----------------------------
    # 3) Annual “modal permco” from observed patents (for employer proxy)
    # ----------------------------
    # Choose permco with most patent rows for that inventor-year; tie-breaker: smallest permco
    tmp = inv_base.dropna(subset=["inventor_id", "filing_year", "permco"]).copy()
    tmp["filing_year"] = tmp["filing_year"].astype("int64")
    tmp["permco"] = tmp["permco"].astype("int64")

    counts = (
        tmp.groupby(["inventor_id", "filing_year", "permco"], as_index=False)
           .size()
           .rename(columns={"size": "n_rows"})
    )
    counts = counts.sort_values(["inventor_id", "filing_year", "n_rows", "permco"],
                                ascending=[True, True, False, True])
    permco_mode = (
        counts.drop_duplicates(["inventor_id", "filing_year"], keep="first")
              .rename(columns={"filing_year": "data_year", "permco": "permco_mode"})
              [["inventor_id", "data_year", "permco_mode"]]
    )

    inv_year = inv_year_metrics.merge(permco_mode, on=["inventor_id", "data_year"], how="left")

    # ----------------------------
    # 4) Build full inventor×year grid and merge metrics (zero-fill sums/counts)
    # ----------------------------
    # career span based on observed patents for these inventors
    span = (
        inv_base.dropna(subset=["inventor_id", "filing_year"])
                .groupby("inventor_id")["filing_year"]
                .agg(["min", "max"])
                .reset_index()
                .rename(columns={"min": "min_year", "max": "max_year"})
    )
    span["min_year"] = span["min_year"].astype("int64")
    span["max_year"] = span["max_year"].astype("int64")

    if window_fill_years is not None:
        # Override: force a common window for everyone, e.g. (1985, 2019)
        y0, y1 = window_fill_years
        span["min_year"] = int(y0)
        span["max_year"] = int(y1)

    print("\tExpanding to full inventor-year grid (may take a while)...", flush=True)
    # Expand to full grid (vectorized)
    rows = []
    
    _n = len(span)  
    for _i, r in enumerate(span.itertuples(index=False)):  # default start=0
        if (_i + 1) % 10000 == 0 or (_i + 1) == _n:
            print(f"\tGrid expansion: {_i+1:,}/{_n:,} inventors", flush=True)

        yrs = np.arange(r.min_year, r.max_year + 1, dtype="int64")
        rows.append(pd.DataFrame({"inventor_id": r.inventor_id, "data_year": yrs}))
    grid = pd.concat(rows, ignore_index=True)
    print(f"\tBuilt inventor-year grid: {grid.shape[0]:,} rows", flush=True)

    panel = grid.merge(inv_year, on=["inventor_id", "data_year"], how="left")

    # Attach inventor-level treatment status (never-treated controls have ever_treated_inv==0)
    panel = panel.merge(inv_treated_flag, on="inventor_id", how="left")
    panel["ever_treated_inv"] = panel["ever_treated_inv"].fillna(0).astype("int32")

    # attach inventor-level static features to all years (incl. zero-patent years)
    panel = panel.merge(inv_static, on="inventor_id", how="left")
    
    # inventor_age = data_year - first_filing_year
    panel["first_filing_year"] = pd.to_numeric(panel["first_filing_year"], errors="coerce").astype("Int64")
    panel["inventor_age"] = (panel["data_year"].astype("Int64") - panel["first_filing_year"]).astype("Int64")
    panel.loc[panel["inventor_age"] < 0, "inventor_age"] = np.nan

    # Zero-fill sums/counts; keep means as NaN for zero-patent years
    sum_like_cols = [
        "total_patents", "cites", "xi_real", "backward_citations", "self_citations",
        "top1_forward_citations", "top10_to_2_forward_citations",
        "cited_patent_forward_citations", "Uncited_patent_forward_citations",
        "same_first_cpc_subclass",
    ]
    for c in (col for col in sum_like_cols if col in panel.columns):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    # cumulative metrics on the full grid (for matching in 15.6)
    panel = panel.sort_values(["inventor_id", "data_year"]).reset_index(drop=True)
    
    if "total_patents" in panel.columns:
        panel["cum_patents"] = panel.groupby("inventor_id")["total_patents"].cumsum()
    else:
        panel["cum_patents"] = 0.0
    
    if "cites" in panel.columns:
        panel["cum_cites"] = panel.groupby("inventor_id")["cites"].cumsum()
    else:
        panel["cum_cites"] = 0.0

    # ----------------------------
    # 5) Assign permco in missing years:
    #    (a) forward-fill permco_mode within inventor
    #    (b) override using strict mover_events_df (to_permco from movement_year onward)
    # ----------------------------
    panel = panel.sort_values(["inventor_id", "data_year"]).reset_index(drop=True)
    panel["permco_assigned"] = panel.groupby("inventor_id")["permco_mode"].ffill()

    # mover overrides
    moves = mover_events_df.copy()
    moves["movement_year"] = pd.to_numeric(moves["movement_year"], errors="coerce").astype("Int64")
    moves["to_permco"] = pd.to_numeric(moves["to_permco"], errors="coerce")
    moves = moves.dropna(subset=["inventor_id", "movement_year", "to_permco"]).copy()
    moves["movement_year"] = moves["movement_year"].astype("int64")
    moves["to_permco"] = moves["to_permco"].astype("int64")
    moves = moves.sort_values(["inventor_id", "movement_year"])

    # apply overrides inventor-by-inventor (fast enough; number of move events is modest)
    panel["permco_assigned"] = pd.to_numeric(panel["permco_assigned"], errors="coerce")
    
    print(f"\tApplying mover overrides for {moves['inventor_id'].nunique():,} movers...", flush=True)
    _done = 0  
    _total = moves["inventor_id"].nunique()  
    for inv_id, g in moves.groupby("inventor_id", sort=False):
        idx = panel["inventor_id"] == inv_id
        if not idx.any():
            continue
        # for each move year, set employer from that year onward
        for mv in g.itertuples(index=False):
            panel.loc[idx & (panel["data_year"] >= mv.movement_year), "permco_assigned"] = mv.to_permco

        _done += 1  
        if _done % 5000 == 0 or _done == _total:  # <-- add
            print(f"\tMover overrides: {_done:,}/{_total:,} inventors", flush=True)
    print("\tFinished mover overrides", flush=True)

    # optional: keep only years where assigned permco is an M&A permco
    # If you instead want *full career* for these inventors (including non-MA firms), comment this out.
    panel["is_treated_firm_year"] = panel["permco_assigned"].isin(treated_permcos_set).astype("int32")

    # ----------------------------
    # 6) Move indicator at inventor-year level (+ from/to in move year)
    # ----------------------------
    move_year_map = (
        mover_events_df[["inventor_id", "movement_year", "from_permco", "to_permco"]]
        .dropna(subset=["inventor_id", "movement_year"])
        .drop_duplicates(subset=["inventor_id", "movement_year"])
        .copy()
    )
    move_year_map["movement_year"] = pd.to_numeric(move_year_map["movement_year"], errors="coerce").astype("Int64")
    move_year_map = move_year_map.dropna(subset=["movement_year"])
    move_year_map["movement_year"] = move_year_map["movement_year"].astype("int64")

    panel = panel.merge(
        move_year_map.rename(columns={"movement_year": "data_year"}),
        on=["inventor_id", "data_year"],
        how="left"
    )
    panel["is_move_year"] = panel["to_permco"].notna().astype("int32")

    # ----------------------------
    # 7) Merge M&A event-study context for the assigned employer firm-year
    # ----------------------------
    # final_event_panel columns exist from your Section 15.3 output
    fe = final_event_panel.copy()
    fe["permco"] = pd.to_numeric(fe["permco"], errors="coerce")
    fe["data_year"] = pd.to_numeric(fe["data_year"], errors="coerce")
    fe = fe.dropna(subset=["permco", "data_year"]).copy()
    fe["permco"] = fe["permco"].astype("int64")
    fe["data_year"] = fe["data_year"].astype("int64")

    fe_cols = [
        "permco", "data_year",
        "ma_deal_role", "years_from_ma_deal", "ma_failed_merger",
        "ma_deal_value", "ma_other_party", "ma_predeal_tech_similarity"
    ]
    fe_cols = [c for c in fe_cols if c in fe.columns]
    fe = fe[fe_cols].copy()

    print("\tMerging M&A context onto inventor-year panel...", flush=True)
    panel = panel.merge(
        fe,
        left_on=["permco_assigned", "data_year"],
        right_on=["permco", "data_year"],
        how="left"
    ).drop(columns=["permco"], errors="ignore")
    print("\tFinished M&A context merge", flush=True)

    # Make the “no deal” category explicit (matches your firm-year logic)
    if "ma_deal_role" in panel.columns:
        panel["ma_deal_role"] = panel["ma_deal_role"].fillna("no_recent_MandA")

    # "Treated firm-year" = at employer in a year that is on/after the (closest) deal
    panel["cs_treated_firmyear"] = (
        panel["years_from_ma_deal"].notna() & (panel["years_from_ma_deal"] >= 0)
    ).astype("int32")

    # ----------------------------
    # 7b) Cohort (G_i) variables for Callaway–Sant'Anna (csdid) later
    # ----------------------------
    # --- ITT exposure indicator (timed, matches years_from_ma_deal) ---
    panel["cs_treated_it"] = panel["cs_treated_firmyear"].astype("int32")
    
    # role-specific ITT exposure
    panel["cs_treated_target_it"] = (
        (panel["cs_treated_firmyear"] == 1) &
        (panel["ma_deal_role"].astype(str).str.lower() == "target")
    ).astype("int32")
    
    panel["cs_treated_acquiror_it"] = (
        (panel["cs_treated_firmyear"] == 1) &
        (panel["ma_deal_role"].astype(str).str.lower() == "acquiror")
    ).astype("int32")
    
    # cohorts = first year of timed exposure
    first_g_all = panel.loc[panel["cs_treated_it"] == 1].groupby("inventor_id")["data_year"].min()
    panel["cs_g_year_all"] = panel["inventor_id"].map(first_g_all).astype("float")
    
    first_g_tgt = panel.loc[panel["cs_treated_target_it"] == 1].groupby("inventor_id")["data_year"].min()
    panel["cs_g_year_target"] = panel["inventor_id"].map(first_g_tgt).astype("float")
    
    first_g_acq = panel.loc[panel["cs_treated_acquiror_it"] == 1].groupby("inventor_id")["data_year"].min()
    panel["cs_g_year_acquiror"] = panel["inventor_id"].map(first_g_acq).astype("float")
  
    # Indicator: treated under the csdid cohort definition (ever exposed as employee to treated firm-year)
    panel["cs_ever_treated_by_firmyear"] = panel["cs_g_year_all"].notna().astype("int32")

    # ----------------------------
    # 8) Final tidy + save
    # ----------------------------
    panel = panel.sort_values(["inventor_id", "data_year"]).reset_index(drop=True)

    chk = panel.groupby("inventor_id")["data_year"].agg(["min", "max", "nunique"])
    share_full = (chk["nunique"] == (chk["max"] - chk["min"] + 1)).mean()
    print(f"\tGrid completeness: {100*share_full:.2f}%", flush=True)

    print("\tSaving inventor-year panel...", flush=True)
    out_path = os.path.join(out_dir, "inventor_year_panel_ma_inventors.pkl")
    panel.to_pickle(out_path)

    print(f"\tSaved inventor-year M&A inventor panel: {out_path}", flush=True)
    print(f"\t[Inventor-year MA panel] {panel.shape[0]:,} rows × {panel.shape[1]:,} cols", flush=True)

    return panel


# --- Execute and save ---
inventor_year_ma_panel = build_inventor_year_panel_ma_inventors(
    pat_inv_firm_df=pat_inv_firm_df,
    manda_df=manda_df,
    mover_events_df=mover_events_df,
    final_event_panel=final_event_panel,
    out_dir=OUTPUT_PATH,
    window_fill_years=None  # e.g., (1985, 2019) if you want a common global window for all inventors
)

r'''
# %%
# =====================================================================
# FAST-RERUN GATE (ahead of 15.6): load 15.5 output + 15.3 event panel
# =====================================================================
START_AT_SECTION = 15.6   # set to 0 for full run; set to 15.6 to rerun only from here

# Use strings + os.path.join instead of Path
INTER_DIR = os.path.join(OUTPUT_PATH, "intermediate_files")
os.makedirs(INTER_DIR, exist_ok=True)

def load_pickle(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache for {name}: {path}\n"
            f"Run the full pipeline once to create it."
        )
    return pd.read_pickle(path)

if START_AT_SECTION >= 15.6:
    # (A) needed if you are *skipping* 15.5
    inventor_year_ma_panel = load_pickle(
        os.path.join(OUTPUT_PATH, "inventor_year_panel_ma_inventors.pkl"),
        "inventor_year_ma_panel"
    )

    # (B) needed only if you also skip earlier sections and references exist downstream
    #     (safe to load even if 15.6 itself doesn't use it directly)
    final_event_panel = load_pickle(
        os.path.join(OUTPUT_PATH, "final_firm_year_ma_event_study_panel.pkl"),
        "final_event_panel"
    )

    # (C) optional: if you later run the inventor-year analysis block that uses firm_lag
    #     (load only if it exists in your pipeline)
    firm_lag_path = os.path.join(OUTPUT_PATH, "firm_lag.pkl")
    if os.path.exists(firm_lag_path):
        firm_lag = load_pickle(firm_lag_path, "firm_lag")
    else:
        firm_lag = None

    print("[gate] Loaded cached objects; running from Section 15.6+ only.")
r'''

# %%
#################################################################
# SECTION 15.6: INVENTOR × M&A EVENT-STUDY PANEL (-5..+5)
#################################################################
print("\n--- Section 15.6: Building inventor M&A event-study panel (-5..+5) ---")

def filter_events_feasible_over_window(
    df0: pd.DataFrame,
    events: pd.DataFrame,
    window: tuple[int, int],
    id_col: str = "inventor_id",
    year_col: str = "data_year",
    event_year_col: str = "closest_deal_year",
):
    """
    Keep only events where the inventor is observed for ALL calendar years in
    [event_year+lo, event_year+hi].

    df0: inventor-year panel containing at least (inventor_id, data_year)
    events: event list containing at least (inventor_id, closest_deal_year)
    """
    lo, hi = window

    d = df0[[id_col, year_col]].copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d.dropna(subset=[id_col, year_col]).copy()
    d[year_col] = d[year_col].astype(int)

    span = (d.groupby(id_col)[year_col]
              .agg(min_year="min", max_year="max")
              .reset_index())

    e = events.copy()
    e[event_year_col] = pd.to_numeric(e[event_year_col], errors="coerce")
    e = e.dropna(subset=[id_col, event_year_col]).copy()
    e[event_year_col] = e[event_year_col].astype(int)

    e = e.merge(span, on=id_col, how="left")

    # observed for ALL years in window iff inventor span covers the window
    keep = (e["min_year"] <= (e[event_year_col] + lo)) & (e["max_year"] >= (e[event_year_col] + hi))
    out = e.loc[keep].copy()

    return out


def build_matched_control_events_tminus1(
    df0: pd.DataFrame,
    treated_events: pd.DataFrame,
    window: tuple[int, int] = (-5, 5),
    start_tol: int = 2,
    end_tol: int = 2,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    n_controls: int = 3,
    max_reuse: int | None = 10,   # None = no cap
    match_method: str = "ps",          # "ps" or "euclid"
    ps_covs: list[str] | None = None,  
    ps_caliper: float | None = 0.05,   
    exact_cols: list[str] | None = None, #(e.g., ["first_cpc_subclass"])

):
    """
    Create control pseudo-events matched to treated events at t=-1.

    Matching logic:
      - event_year is the treated event's closest_deal_year
      - pick control inventors (ever_treated_inv==0) with:
          * observation at (event_year-1)  [t=-1]
          * span constraints for window feasibility
          * start/end years within tolerances
      - 1-NN in feature space using sklearn (per event_year)
    """
    lo, hi = window

    df = df0.copy()
    df["data_year"] = pd.to_numeric(df["data_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["inventor_id", "data_year"]).copy()
    df["data_year"] = df["data_year"].astype("int64")

    # inventor spans
    span = (df.groupby("inventor_id")["data_year"]
              .agg(min_year="min", max_year="max")
              .reset_index())
    df = df.merge(span, on="inventor_id", how="left")

    # cumulative productivity should already exist from 15.5; compute only if missing
    df = df.sort_values(["inventor_id", "data_year"])
    if "cum_patents" not in df.columns:
        df["total_patents"] = pd.to_numeric(df.get("total_patents"), errors="coerce").fillna(0)
        df["cum_patents"] = df.groupby("inventor_id")["total_patents"].cumsum()
    if "cum_cites" not in df.columns:
        df["cites"] = pd.to_numeric(df.get("cites"), errors="coerce").fillna(0)
        df["cum_cites"] = df.groupby("inventor_id")["cites"].cumsum()

    # defaults
    if cat_cols is None:
        cat_cols = [c for c in ["first_cpc_subclass", "first_cpc_group"] if c in df.columns]
    if num_cols is None:
        num_cols = [c for c in ["inventor_age", "cum_patents", "cum_cites"] if c in df.columns]

    if exact_cols is None:
        exact_cols = [c for c in ["first_cpc_subclass"] if c in df.columns]

    # PS features default: (numeric + a couple of “safe” extra controls if present)
    if ps_covs is None:
        ps_covs = []
        ps_covs += [c for c in ["inventor_age"] if c in df.columns]  # already in your num_cols
        ps_covs += [c for c in ["cum_patents", "cum_cites"] if c in df.columns]  # you already compute these


    # treated events -> t-1 rows
    te = treated_events.copy()
    te["closest_deal_year"] = pd.to_numeric(te["closest_deal_year"], errors="coerce")
    te = te.dropna(subset=["inventor_id", "closest_deal_year"]).copy()
    te["closest_deal_year"] = te["closest_deal_year"].astype("int64")
    te = te.merge(span, on="inventor_id", how="left")

    # t-1 year for treated events
    te["data_year"] = te["closest_deal_year"].astype("int64") - 1  # t=-1 calendar year

    treated_t1 = te.merge(
        df,
        on=["inventor_id", "data_year"],
        how="left",
        suffixes=("", "_t1")
    )
    treated_t1 = treated_t1.dropna(subset=["permco_assigned"]).copy()

    # candidate controls: never-treated, use their row at (event_year-1) as feature row
    ctrl = df[df["ever_treated_inv"] == 0].copy()

    # Build permco at event_year (t=0) for event-id stability; fallback to t=-1 permco if needed
    permco_event = df[["inventor_id", "data_year", "permco_assigned"]].copy()
    permco_event = permco_event.rename(columns={"data_year": "closest_deal_year",
                                                "permco_assigned": "permco_event_year"})

    # Matching per cohort year
    out_rows = []

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    for y, g_t in treated_t1.groupby("closest_deal_year", sort=True):
        print(f"\tMatching controls for cohort year {int(y)} (treated events: {g_t.shape[0]:,})", flush=True)
        
        # controls that can serve as t=-1 for this cohort year
        g_c = ctrl[ctrl["data_year"] == (int(y) - 1)].copy()   # t=-1 year for cohort y
        if g_c.empty:
            print(f"\t  No eligible controls for cohort year {int(y)}; skipping", flush=True)
            continue

        # drop infeasible controls *before* matching (must cover full [y+lo, y+hi])
        g_c = g_c[(g_c["min_year"] <= (int(y) + lo)) & (g_c["max_year"] >= (int(y) + hi))].copy()
        if g_c.empty:
            print(f"\t  No window-feasible controls for cohort year {int(y)}; skipping", flush=True)
            continue

        # ---------- cohort-year propensity score at t=-1 ----------
        # We estimate PS on pooled (treated + control candidates) at t=-1 within this cohort year.
        # Label: 1 = treated event row, 0 = control candidate row
        pscore_map = None
        if match_method.lower() == "ps":
            pool_t = g_t.copy()
            pool_t["_ps_treated"] = 1
            pool_c = g_c.copy()
            pool_c["_ps_treated"] = 0
        
            pool = pd.concat([pool_t, pool_c], ignore_index=True, sort=False)
        
            use_num = [c for c in ps_covs if c in pool.columns]
            use_cat = [c for c in exact_cols if c in pool.columns]
        
            keep_cols = use_num + use_cat
            pool_ps = pool.dropna(subset=keep_cols + ["_ps_treated"]).copy()
        
            if pool_ps["_ps_treated"].nunique() == 2 and not pool_ps.empty:
                # numeric (scaled)
                X_num = pool_ps[use_num].apply(pd.to_numeric, errors="coerce")
                scaler = StandardScaler()
                X_num = pd.DataFrame(
                    scaler.fit_transform(X_num),
                    columns=use_num,
                    index=pool_ps.index,
                )
        
                # categorical dummies (unchanged)
                X_cat = pd.get_dummies(pool_ps[use_cat].astype(str), drop_first=True) if use_cat else None
                X_ps = X_num if X_cat is None else pd.concat([X_num, X_cat], axis=1)
        
                y_ps = pool_ps["_ps_treated"].astype(int).to_numpy()
        
                try:
                    lr = LogisticRegression(
                        solver="lbfgs",
                        max_iter=10000,          
                        class_weight="balanced"
                    )
                    # optional: silence convergence warning without hiding real errors
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        lr.fit(X_ps, y_ps)
        
                    pool_ps["pscore"] = lr.predict_proba(X_ps)[:, 1]
                    pscore_map = pool_ps[["inventor_id", "data_year", "pscore"]].drop_duplicates()
                except Exception:
                    pscore_map = None

        if pscore_map is not None:
            g_t = g_t.merge(pscore_map, on=["inventor_id", "data_year"], how="left")
            g_c = g_c.merge(pscore_map, on=["inventor_id", "data_year"], how="left")
        else:
            g_t["pscore"] = np.nan
            g_c["pscore"] = np.nan
            

        # reuse cap counter (per cohort-year)
        reuse = Counter()

        # start/end tolerance constraint vs treated inventor span (per treated row)
        # We'll do it row-wise (still usually fast enough) to keep logic exact.
        Xt_all = g_t.copy()

        # Fit once per cohort on all controls (we'll subset per treated row)
        # But because we enforce tolerance constraints per treated row, we refit NN on the subset.
        _total = Xt_all.shape[0]
        _done = 0
        for ridx, tr in Xt_all.iterrows():
            _done += 1
            if _done % 500 == 0 or _done == _total:
                 print(f"\tCohort {int(y)}: {_done:,}/{_total:,} treated processed", flush=True)

            cand = g_c[
                (g_c["min_year"].sub(tr["min_year"]).abs() <= start_tol) &
                (g_c["max_year"].sub(tr["max_year"]).abs() <= end_tol)
            ].copy()

            if cand.empty:
                continue

            # --- drop rows with missing matching features (minimal fix for NaNs) ---
            feat_cols = num_cols + cat_cols
            
            # skip this treated event if its matching features are missing
            if tr[feat_cols].isna().any():
                continue
            
            # drop candidate controls with missing matching features
            cand = cand.dropna(subset=feat_cols)
            if cand.empty:
                continue

            # --- matching ---
            if match_method.lower() == "ps" and ("pscore" in cand.columns) and pd.notna(tr.get("pscore", np.nan)):
                # exact matching (if specified)
                for ex in exact_cols:
                    if ex in cand.columns and ex in tr.index:
                        cand = cand[cand[ex].astype(str) == str(tr[ex])]
                if cand.empty:
                    continue

                # caliper
                cand = cand.dropna(subset=["pscore"])
                if cand.empty:
                    continue
                cand["ps_dist"] = (cand["pscore"] - float(tr["pscore"])).abs()
                if ps_caliper is not None:
                    cand = cand[cand["ps_dist"] <= float(ps_caliper)]
                if cand.empty:
                    continue

                # choose n_controls by smallest ps distance
                cand = cand.sort_values("ps_dist")
                chosen = cand.head(int(n_controls)).copy()

                for rank, (_, ch) in enumerate(chosen.iterrows(), start=1):
                    out_rows.append({
                        "treated_inventor_id": tr["inventor_id"],
                        "treated_permco_event": tr.get("permco_assigned", np.nan),
                        "closest_deal_year": int(y),
                        "ma_deal_role": "control",
                        "ma_other_party": np.nan,
                        "is_control_event": 1,
                        "control_inventor_id": ch["inventor_id"],
                        "tminus1_year": int(y - 1),
                        "nn_dist": float(ch["ps_dist"]),   # reuse field; now PS distance
                        "control_rank": int(rank),
                    })

                    if max_reuse is not None:
                        reuse[ch["inventor_id"]] += 1
                        if reuse[ch["inventor_id"]] >= max_reuse:
                            g_c = g_c[g_c["inventor_id"] != ch["inventor_id"]]

            else:
                # assemble feature matrices
                Xt = pd.DataFrame([tr])[num_cols + cat_cols]
                Xc = cand[num_cols + cat_cols]
    
                pipe = Pipeline([("pre", preproc)])
                Xc_t = pipe.fit_transform(Xc)
                Xt_t = pipe.transform(Xt)
    
                k = min(int(n_controls), Xc_t.shape[0])
                nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
                nn.fit(Xc_t)
                dist, idx = nn.kneighbors(Xt_t, return_distance=True)
                
                chosen_ids = set()
                for rank, j in enumerate(idx[0], start=1):
                    chosen = cand.iloc[int(j)]
    
                    if chosen["inventor_id"] in chosen_ids:
                        continue
                    chosen_ids.add(chosen["inventor_id"])
    
                    out_rows.append({
                        "treated_inventor_id": tr["inventor_id"],
                        "treated_permco_event": tr.get("permco_assigned", np.nan),
                        "closest_deal_year": int(y),
                        "ma_deal_role": "control",
                        "ma_other_party": np.nan,
                        "is_control_event": 1,
                        "control_inventor_id": chosen["inventor_id"],
                        "tminus1_year": int(y - 1),
                        "nn_dist": float(dist[0][rank - 1]),
                        "control_rank": int(rank),          # (optional but useful)
                    })
    
                    # bump reuse count (only if we are capping)
                    if max_reuse is not None:
                        reuse[chosen["inventor_id"]] += 1
                        if reuse[chosen["inventor_id"]] >= max_reuse:
                            g_c = g_c[g_c["inventor_id"] != chosen["inventor_id"]]


    matches = pd.DataFrame(out_rows)
    if matches.empty:
        return pd.DataFrame(columns=["inventor_id","permco_assigned","closest_deal_year","ma_deal_role","ma_other_party","is_control_event"])

    # Build ctrl_events rows
    ctrl_events = matches.rename(columns={"control_inventor_id": "inventor_id"}).copy()
    ctrl_events["permco_assigned"] = np.nan

    # attach permco at event_year if available
    ctrl_events = ctrl_events.merge(
        permco_event,
        on=["inventor_id", "closest_deal_year"],
        how="left"
    )


    # if missing permco_event_year, fallback to t=-1 row's permco_assigned
    fallback = ctrl[["inventor_id", "data_year", "permco_assigned"]].copy()
    fallback["closest_deal_year"] = fallback["data_year"] + 1
    fallback = fallback.rename(columns={"permco_assigned": "permco_event_year_fallback"})

    ctrl_events = ctrl_events.merge(
        fallback[["inventor_id", "closest_deal_year", "permco_event_year_fallback"]],
        on=["inventor_id", "closest_deal_year"],
        how="left"
    )

    ctrl_events["permco_assigned"] = ctrl_events["permco_event_year"].fillna(ctrl_events["permco_event_year_fallback"])

    keep = [
        "inventor_id",            # control inventor
        "treated_inventor_id",    
        "control_rank",           
        "nn_dist",                
        "permco_assigned",
        "closest_deal_year",
        "ma_deal_role",
        "ma_other_party",
        "is_control_event",
    ]
    keep = [c for c in keep if c in ctrl_events.columns]
    ctrl_events = ctrl_events[keep].dropna(subset=["inventor_id","closest_deal_year"])

    print(f"\tMatched control pseudo-events created: {len(out_rows):,}", flush=True)
    return ctrl_events


def build_inventor_ma_event_study_panel(
    inventor_year_panel: pd.DataFrame,
    out_dir: str,
    window: tuple[int, int] = (-5, 5),
    require_assigned_firm: bool = True,
    control_anchor: str = "midpoint",  # "midpoint" (default) or "first" or "last"
):
    """
    Build a balanced inventor×event window panel around M&A events using
    firm-year event context already merged from final_event_panel (via 15.5).

    Fixes two issues:
      (1) Post-move outcomes bug: merge outcomes by (inventor_id, data_year),
          not by event keys (which include permco_assigned).
      (2) Preserve never-M&A inventors as controls: add an "anchor event year"
          for inventors with ever_treated_inv==0 so they enter the event-study grid.

    Expects inventor_year_panel to contain at least:
      - inventor_id, data_year
      - permco_assigned (can be NaN if require_assigned_firm=False)
      - ever_treated_inv (0/1)
      - years_from_ma_deal + M&A context columns for treated firm-years (may be NaN for controls)

    Output:
      One row per inventor-event-relative year, balanced over window [lo, hi].
      Includes both treated events and control-anchor events.
    """
    lo, hi = window

    # ----------------------------
    # 0) Base + basic hygiene
    # ----------------------------
    df0 = inventor_year_panel.copy()

    # ensure core dtypes
    df0["data_year"] = pd.to_numeric(df0["data_year"], errors="coerce").astype("Int64")
    df0["ever_treated_inv"] = pd.to_numeric(df0.get("ever_treated_inv"), errors="coerce").fillna(0).astype("int32")

    if require_assigned_firm:
        df0 = df0.dropna(subset=["permco_assigned"])
    print(f"\tBase inventor-year rows after hygiene: {df0.shape[0]:,}", flush=True)

    # make sure permco_assigned is numeric-ish for keys
    df0["permco_assigned"] = pd.to_numeric(df0["permco_assigned"], errors="coerce").astype("Int64")

    # We'll keep an outcomes/metrics table keyed ONLY by inventor_id + data_year
    # This is the critical fix for the "after moves" merge bug.
    outcome_cols = [c for c in df0.columns if c not in [
        "years_from_ma_deal", "ma_deal_role", "ma_failed_merger", "ma_deal_value",
        "ma_other_party", "ma_predeal_tech_similarity"
    ]]
    outcomes = df0[outcome_cols].copy()

    # ----------------------------
    # 1) Treated events: rows with defined years_from_ma_deal in [-5,+5]
    # ----------------------------
    df_t = df0.copy()
    df_t["years_from_ma_deal"] = pd.to_numeric(df_t.get("years_from_ma_deal"), errors="coerce")
    df_t = df_t[df_t["years_from_ma_deal"].between(lo, hi)].copy()

    # implied closest deal year
    df_t["closest_deal_year"] = (df_t["data_year"] - df_t["years_from_ma_deal"]).astype("Int64")

    # event identity (treated)
    event_keys_t = ["inventor_id", "permco_assigned", "closest_deal_year"]
    for extra in ["ma_deal_role", "ma_other_party"]:
        if extra in df_t.columns:
            event_keys_t.append(extra)

    df_t = df_t.drop_duplicates(subset=event_keys_t + ["years_from_ma_deal"]).copy()
    treated_events = df_t[event_keys_t].drop_duplicates().copy()
    print(f"\tTreated events (pre-feasibility): {treated_events.shape[0]:,}", flush=True)

    treated_events = filter_events_feasible_over_window(
        df0=df0,
        events=treated_events,
        window=window,
    )
    treated_events["is_control_event"] = 0
    print(f"\tTreated events (feasible over window): {treated_events.shape[0]:,}", flush=True)

    # ----------------------------
    # 2) Matched control pseudo-events (t=-1 matching, no midpoint anchor)
    # ----------------------------
    print("\tBuilding matched control pseudo-events (t=-1 matching)...", flush=True)
    ctrl_events = build_matched_control_events_tminus1(
        df0=df0,
        treated_events=treated_events,
        window=window,
        start_tol=2,
        end_tol=2,
        # Optional: tighten or expand features here
        cat_cols=[c for c in ["first_cpc_subclass"] if c in df0.columns],
        num_cols=[c for c in ["inventor_age", "cum_patents", "cum_cites"] if c in df0.columns],
        n_controls=3,
        max_reuse=10,
        match_method="ps",
        ps_covs=[c for c in ["inventor_age", "cum_patents", "cum_cites"] if c in df0.columns],
        ps_caliper=0.05,
        exact_cols=[c for c in ["first_cpc_subclass"] if c in df0.columns],
    )
    
    # Harmonize labels expected downstream
    ctrl_events["ma_deal_role"] = "control"
    ctrl_events["ma_other_party"] = np.nan
    print(f"\tControl pseudo-events (pre-feasibility): {ctrl_events.shape[0]:,}", flush=True)

    n0 = ctrl_events.shape[0]
    ctrl_events = filter_events_feasible_over_window(df0=df0, events=ctrl_events, window=window)
    dropped = n0 - ctrl_events.shape[0]
    if dropped:
        print(f"\tWARNING: feasibility filter dropped {dropped:,} matched controls", flush=True)
    ctrl_events["is_control_event"] = 1
    print(f"\tControl pseudo-events (feasible over window): {ctrl_events.shape[0]:,}", flush=True)

    # ----------------------------
    # 3) Build unified event list + balanced relative-year grid
    # ----------------------------
    # Use a unified event id key that works for both treated + controls.
    # (Controls have ma_deal_role="control_anchor" and ma_other_party NaN.)
    unified_event_keys = ["inventor_id","permco_assigned","closest_deal_year","ma_deal_role","ma_other_party"]
    
    extra_keys = [c for c in ["treated_inventor_id", "control_rank", "nn_dist"] if c in ctrl_events.columns]

    # Ensure columns exist in treated_events / ctrl_events
    for col in ["ma_deal_role", "ma_other_party"]:
        if col not in treated_events.columns:
            treated_events[col] = np.nan
    for col in ["ma_deal_role", "ma_other_party"]:
        if col not in ctrl_events.columns:
            ctrl_events[col] = np.nan

    events_all = pd.concat(
        [
            treated_events.reindex(columns=unified_event_keys + ["is_control_event"]),
            ctrl_events.reindex(columns=unified_event_keys + extra_keys + ["is_control_event"]),
        ],
        ignore_index=True
    )

    # Drop any events missing core pieces
    events_all["closest_deal_year"] = pd.to_numeric(events_all["closest_deal_year"], errors="coerce")
    events_all = events_all.dropna(subset=["inventor_id", "closest_deal_year"]).copy()
    events_all["closest_deal_year"] = events_all["closest_deal_year"].astype("int64")

    # Balanced grid
    print("\tExpanding to balanced inventor-event window grid...", flush=True)
    rel = np.arange(lo, hi + 1, dtype="int64")
    grid = events_all.loc[events_all.index.repeat(len(rel))].copy()
    grid["years_from_ma_deal"] = np.tile(rel, len(events_all))

    # calendar year for each relative period
    grid["data_year"] = (grid["closest_deal_year"].astype("int64") + grid["years_from_ma_deal"].astype("int64")).astype("int64")
    print(f"\tBalanced grid rows: {grid.shape[0]:,}", flush=True)

    # ----------------------------
    # 4) Merge outcomes BY (inventor_id, data_year)  [BUG FIX]
    # ----------------------------
    outcomes2 = outcomes.copy()
    outcomes2["data_year"] = pd.to_numeric(outcomes2["data_year"], errors="coerce").astype("Int64")
    outcomes2 = outcomes2.dropna(subset=["inventor_id", "data_year"]).copy()
    outcomes2["data_year"] = outcomes2["data_year"].astype("int64")

    print("\tMerging outcomes onto grid (by inventor_id, data_year)...", flush=True)
    out = grid.merge(
        outcomes2,
        on=["inventor_id", "data_year"],
        how="left",
        suffixes=("", "_y")
    )

    # (Optional) Keep "event-firm" separate from year-by-year permco_assigned if you want:
    # - permco_assigned from grid = permco at event/anchor
    # - yearly permco_assigned from outcomes2 could conflict if inventor moves inside window
    # We'll preserve the event firm's permco as permco_event, and keep the yearly employer as permco_assigned_year.
    if "permco_assigned" in out.columns:
        out = out.rename(columns={"permco_assigned": "permco_event"})
    if "permco_assigned_y" in out.columns:
        out = out.rename(columns={"permco_assigned_y": "permco_assigned_year"})
    else:
        # if no suffix happened, create the year version explicitly
        if "permco_assigned" in outcomes2.columns:
            out["permco_assigned_year"] = out.get("permco_assigned", np.nan)

    # carry control flag
    out["is_control_event"] = pd.to_numeric(out.get("is_control_event"), errors="coerce").fillna(0).astype("int32")

    # treated year indicator (t=0)
    out["is_deal_year"] = (out["years_from_ma_deal"] == 0).astype("int32")

    # ----------------------------
    # 5) Zero-fill sum/count outcomes (leave means as NaN)
    # ----------------------------
    sum_like_cols = [
        "total_patents", "cites", "xi_real", "backward_citations", "self_citations",
        "top1_forward_citations", "top10_to_2_forward_citations",
        "cited_patent_forward_citations", "Uncited_patent_forward_citations",
        "same_first_cpc_subclass",
        "is_move_year",
    ]
    for c in (col for col in sum_like_cols if col in out.columns):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # ----------------------------
    # 6) Sort & save
    # ----------------------------
    sort_keys = ["inventor_id", "closest_deal_year", "ma_deal_role", "ma_other_party", "years_from_ma_deal"]
    sort_keys = [k for k in sort_keys if k in out.columns]

    out = out.sort_values(sort_keys).reset_index(drop=True)

    out_path = os.path.join(out_dir, "inventor_ma_event_study_panel.pkl")
    out.to_pickle(out_path)
    print(f"\tSaved inventor M&A event-study panel: {out_path}", flush=True)
    print(f"\t[Inventor M&A ES panel] {out.shape[0]:,} rows × {out.shape[1]:,} cols", flush=True)

    # quick diagnostics
    n_events = events_all.shape[0]
    n_ctrl_events = int(events_all["is_control_event"].sum()) if "is_control_event" in events_all.columns else 0
    print(f"\tEvents: {n_events:,} (control anchors: {n_ctrl_events:,})", flush=True)

    return out

inventor_ma_event_study_panel = build_inventor_ma_event_study_panel(
    inventor_year_panel=inventor_year_ma_panel,  # from Section 15.5
    out_dir=OUTPUT_PATH,
    window=(-5, 5),
    require_assigned_firm=True
)

# %%
# ---------------------------------------------------------------
# QUICK DIAGNOSTICS: Inventor × Event-study panel
# ---------------------------------------------------------------
es = inventor_ma_event_study_panel
lo, hi = -5, 5
expected = hi - lo + 1

print("\n--- Diagnostics: inventor M&A event-study panel ---", flush=True)
print(f"\tPanel: {es.shape[0]:,} rows × {es.shape[1]:,} cols", flush=True)

# 1) Balanced window check: each (inventor,event) should have exactly expected relative years
event_id_cols = ["inventor_id", "closest_deal_year", "ma_deal_role", "ma_other_party"]
if "permco_event" in es.columns:
    event_id_cols.insert(1, "permco_event")  # keep keys tight
event_id_cols = [c for c in event_id_cols if c in es.columns]

sizes = es.groupby(event_id_cols, dropna=False)["years_from_ma_deal"].nunique()
n_total = sizes.shape[0]
n_full  = int((sizes == expected).sum())
print(f"\tBalanced events: {n_full:,}/{n_total:,} (expect {expected} rel-years)", flush=True)
if n_full < n_total:
    print(f"\tWARNING: {n_total - n_full:,} events missing some rel-years", flush=True)

# 2) Treated vs control rows by relative year (should be flat within each group if balanced)
if "is_control_event" in es.columns:
    ct = (es.groupby(["is_control_event", "years_from_ma_deal"])
            .size()
            .unstack(0, fill_value=0)
            .rename(columns={0: "treated_rows", 1: "control_rows"}))
    print("\tRows per rel-year (treated vs control):", flush=True)
    print(ct.to_string(), flush=True)

# 3) Simple t=-1 similarity check (means)
if "is_control_event" in es.columns:
    t1 = es[es["years_from_ma_deal"] == -1].copy()
    for v in ["inventor_age", "cum_patents", "cum_cites", "total_patents", "cites"]:
        if v in t1.columns:
            m = t1.groupby("is_control_event")[v].mean()
            print(f"\t(t=-1) {v}: treated={m.get(0, np.nan):.3f} | control={m.get(1, np.nan):.3f}", flush=True)

# 3b) Balance check at t=-1 via standardized mean differences (SMD)
# Rule of thumb: |SMD| < 0.10 (good), < 0.20 (often acceptable)
if {"is_control_event", "years_from_ma_deal"}.issubset(es.columns):
    t1 = es[es["years_from_ma_deal"] == -1].copy()

    def _smd(df, col, treat_col="is_control_event"):
        # treat_col: 0=treated, 1=control in your construction
        a = pd.to_numeric(df.loc[df[treat_col] == 0, col], errors="coerce")
        b = pd.to_numeric(df.loc[df[treat_col] == 1, col], errors="coerce")
        a, b = a.dropna(), b.dropna()
        if len(a) < 5 or len(b) < 5:
            return np.nan
        m1, m0 = a.mean(), b.mean()
        v1, v0 = a.var(ddof=1), b.var(ddof=1)
        denom = np.sqrt(0.5 * (v1 + v0))
        return np.nan if denom == 0 or np.isnan(denom) else (m1 - m0) / denom

    for v in ["inventor_age", "cum_patents", "cum_cites", "total_patents", "cites"]:
        if v in t1.columns:
            smd = _smd(t1, v)
            print(f"\t(t=-1) SMD {v}: {smd:.3f}" if pd.notna(smd) else f"\t(t=-1) SMD {v}: NA", flush=True)

# 4) Matching reuse diagnostics (controls reused across treated events)
if "is_control_event" in es.columns and "treated_inventor_id" in es.columns:
    m = es[es["is_control_event"] == 1].copy()

    # collapse to one row per matched control-event (avoid counting the 11 rel-years)
    m_unique = m.drop_duplicates(["inventor_id", "closest_deal_year", "treated_inventor_id"])

    vc = m_unique["inventor_id"].value_counts()
    print(f"\t[reuse] unique control-event matches: {m_unique.shape[0]:,}", flush=True)
    print(f"\t[reuse] share controls reused (>1): {(vc>1).mean():.3f}", flush=True)
    print(f"\t[reuse] max reuse count: {int(vc.max()) if len(vc) else 0}", flush=True)
    print("\t[reuse] top reused controls:", flush=True)
    print(vc.head(15).to_string(), flush=True)


# %%
#################################################################
# SECTION 16: SCRIPT COMPLETION
#################################################################
print(f"\n--- Data Construction Complete ---")
