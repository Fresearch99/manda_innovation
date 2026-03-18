"""
04_mobility_and_mover_metrics.py

Inventor move identification, performance around moves, and benchmarking of movers relative to peers. This file keeps the move-identification outputs because they feed later analysis panels, but the separate standalone inventor move panel output is removed elsewhere.

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
# SECTION 8: IDENTIFYING INVENTOR MOBILITY EVENTS
#################################################################
print("--- Section 8: Identifying Inventor Mobility Events ---")
# Logic remains the same, identifying move events.
min_patents = 5
inventor_counts = pat_inv_firm_df['inventor_id'].value_counts()
prolific_inventors = inventor_counts[inventor_counts >= min_patents].index
career_df = pat_inv_firm_df[pat_inv_firm_df['inventor_id'].isin(prolific_inventors)].copy()
career_df.sort_values(['inventor_id', 'filing_date'], inplace=True)

# ±1 neighbors (as before)
career_df['next_permco'] = career_df.groupby('inventor_id')['permco'].shift(-1)
career_df['prev_permco'] = career_df.groupby('inventor_id')['permco'].shift(1)

#  ±2 neighbors to ensure stability on both sides of the move
career_df['next2_permco'] = career_df.groupby('inventor_id')['permco'].shift(-2)
career_df['prev2_permco'] = career_df.groupby('inventor_id')['permco'].shift(2)

# Strict move: first patent at new firm (permco != prev_permco),
# with 2 prior patents at the old firm and 2 following patents at the new firm
is_move = (
    (career_df['permco'] != career_df['prev_permco']) &
    career_df['prev_permco'].notna() &
    career_df['next_permco'].notna() &
    career_df['prev2_permco'].notna() &
    career_df['next2_permco'].notna() &
    (career_df['prev2_permco'] == career_df['prev_permco']) &  # two before same old firm
    (career_df['next_permco']   == career_df['permco']) &      # next one same new firm
    (career_df['next2_permco']  == career_df['next_permco'])   # two after same new firm
)

mover_events_df = career_df[is_move][['inventor_id', 'filing_date', 'filing_year', 'permco', 'prev_permco']].copy()
mover_events_df.rename(columns={'permco': 'to_permco', 'prev_permco': 'from_permco', 'filing_year': 'movement_year'}, inplace=True)
mover_events_df.drop_duplicates(subset=['inventor_id', 'to_permco', 'from_permco'], inplace=True)

mover_events_df['movement_year'] = (
    pd.to_numeric(mover_events_df['movement_year'], errors='coerce').astype('Int64')
)

mover_events_df.to_pickle(os.path.join(INTERMEDIATE_PATH, 'mover_events_df.pkl'))
print(f"Identified {len(mover_events_df)} unique inventor move events.")
print(f"[Move events] size: {mover_events_df.shape[0]:,} rows × {mover_events_df.shape[1]:,} cols", flush=True)


# %%
#################################################################
# SECTION 9: CONSTRUCT PERFORMANCE METRICS AROUND MOVES
#################################################################
print("--- Section 9: Constructing Performance Metrics Around Moves ---")

inventor_moves_patents_df = mover_events_df.merge(pat_inv_firm_df, on='inventor_id', how='left')

# Define pre/post periods
window = 5
inventor_moves_patents_df['years_from_move'] = inventor_moves_patents_df['filing_year'] - inventor_moves_patents_df['movement_year']

# Drop rows where years_from_move is NA
inventor_moves_patents_df = inventor_moves_patents_df.loc[
    inventor_moves_patents_df['years_from_move'].notna()
].copy()

period = np.select(
    [inventor_moves_patents_df['years_from_move'].between(-window, -1),
     inventor_moves_patents_df['years_from_move'].between(0, window-1)],
    ['pre_move', 'post_move'], default='other'
)
inventor_moves_patents_df['period'] = period
inventor_moves_patents_df = inventor_moves_patents_df[inventor_moves_patents_df['period'] != 'other']

# Aggregate performance
agg_metrics = {
    'patent_id': 'nunique',
    'cites': 'sum',
    'xi_real': 'sum',
    'novelty_score_group': 'mean',
    'backward_citations': 'sum',        
    'self_citations': 'sum',            
    'top1_forward_citations': 'sum',
    'top10_to_2_forward_citations': 'sum',       
    'cited_patent_forward_citations': 'sum',   
    'Uncited_patent_forward_citations': 'sum', 
    'exploration_inv': 'mean',
    'exploitation_inv': 'mean', 
    'same_first_cpc_subclass': 'sum',
    'team_size': 'mean'
}
key_cols = ['inventor_id', 'from_permco', 'to_permco', 'movement_year', 'period']
inventor_performance_panel = inventor_moves_patents_df.groupby(key_cols).agg(agg_metrics).reset_index()

# Optional: Pivot to wide format
inventor_perf_wide = inventor_performance_panel.pivot_table(
    index=[c for c in key_cols if c != 'period'],
    columns='period',
    values=list(agg_metrics.keys())
)
inventor_perf_wide.columns = ['_'.join(col).strip() for col in inventor_perf_wide.columns.values]
inventor_perf_wide.reset_index(inplace=True)

# Calculate change in performance for all metrics
for col in agg_metrics.keys():
    pre_col, post_col = f'{col}_pre_move', f'{col}_post_move'
    if pre_col in inventor_perf_wide and post_col in inventor_perf_wide:
        inventor_perf_wide[f'change_{col}'] = inventor_perf_wide[post_col] - inventor_perf_wide[pre_col]

inventor_performance_panel.to_pickle(os.path.join(INTERMEDIATE_PATH, 'inventor_performance_around_moves_enriched.pkl'))
inventor_perf_wide.to_pickle(os.path.join(INTERMEDIATE_PATH, 'inventor_performance_wide.pkl'))
print("\tInventor pre/post move performance calculated with new granular quality metrics.")

print(f"\t[Inventor perf panel] size: {inventor_performance_panel.shape[0]:,} rows × {inventor_performance_panel.shape[1]:,} cols", flush=True)
print(f"\t[Inventor perf wide] size: {inventor_perf_wide.shape[0]:,} rows × {inventor_perf_wide.shape[1]:,} cols", flush=True)

# ---- Annual (relative-to-move) performance by years_from_move ----
annual_key_cols = ['inventor_id', 'from_permco', 'to_permco', 'movement_year', 'years_from_move']

# Panel (one row per inventor × move × relative year)
inventor_performance_annual = (
    inventor_moves_patents_df
    .groupby(annual_key_cols)
    .agg(agg_metrics)
    .reset_index()
    .sort_values(annual_key_cols)
)

# Optional: wide view with safe, signed suffixes (e.g., _m3 for t-3, _p2 for t+2)
annual_wide = inventor_performance_annual.pivot_table(
    index=['inventor_id', 'from_permco', 'to_permco', 'movement_year'],
    columns='years_from_move',
    values=list(agg_metrics.keys())
)
annual_wide.columns = [
    f"{metric}_{('p' if int(rel) >= 0 else 'm')}{abs(int(rel))}"
    for metric, rel in annual_wide.columns.to_flat_index()
]
annual_wide = annual_wide.reset_index()

# Save
inventor_performance_annual.to_pickle(os.path.join(INTERMEDIATE_PATH, 'inventor_performance_annual_panel.pkl'))
annual_wide.to_pickle(os.path.join(INTERMEDIATE_PATH, 'inventor_performance_annual_wide.pkl'))

print(f"\t[Inventor perf annual panel] size: {inventor_performance_annual.shape[0]:,} rows × {inventor_performance_annual.shape[1]:,} cols", flush=True)
print(f"\t[Inventor perf annual wide] size: {annual_wide.shape[0]:,} rows × {annual_wide.shape[1]:,} cols", flush=True)

# %%
#################################################################
# SECTION 10: BENCHMARKING MOVERS AGAINST PEERS (ENHANCED)
#################################################################
print("--- Section 10: Benchmarking Movers Against Their Peers  ---")

results = []
# Add citations; keep existing metrics
benchmark_metrics = {
    'quality_top1': 'top1_forward_citations',
    'quality_top10': 'top10_to_2_forward_citations',
    'novelty': 'novelty_score_group',
    'exploration': 'exploration_inv'   
}

for _, move in tqdm(mover_events_df.iterrows(), total=len(mover_events_df), desc="Benchmarking movers"):
    mover_id, from_permco, to_permco, move_year = move['inventor_id'], move['from_permco'], move['to_permco'], move['movement_year']

    # 5y pre-move window
    window_start_year, window_end_year = move_year - 5, move_year - 1
    # 5y post-move window (includes move year)
    post_window_start_year, post_window_end_year = move_year, move_year + 4  

    # Pre window: origin/destination firm patents
    origin_patents = pat_inv_firm_df[
        (pat_inv_firm_df['permco'] == from_permco) &
        (pat_inv_firm_df['filing_year'].between(window_start_year, window_end_year))
    ]
    dest_patents = pat_inv_firm_df[
        (pat_inv_firm_df['permco'] == to_permco) &
        (pat_inv_firm_df['filing_year'].between(window_start_year, window_end_year))
    ]

    # Post window: destination firm patents 
    dest_patents_post = pat_inv_firm_df[
        (pat_inv_firm_df['permco'] == to_permco) &
        (pat_inv_firm_df['filing_year'].between(post_window_start_year, post_window_end_year))
    ]

    # Isolate mover vs. peers
    mover_patents_at_origin = origin_patents[origin_patents['inventor_id'] == mover_id]
    peer_patents_at_origin  = origin_patents[origin_patents['inventor_id'] != mover_id]

    # Destination: peers in pre window (exclude mover just in case)
    peer_patents_at_dest_pre = dest_patents[dest_patents['inventor_id'] != mover_id]

    # Destination: mover & peers in post window  
    mover_patents_at_dest_post = dest_patents_post[dest_patents_post['inventor_id'] == mover_id]
    peer_patents_at_dest_post  = dest_patents_post[dest_patents_post['inventor_id'] != mover_id]

    move_context = move.to_dict()
    epsilon = 1e-9  # avoid division by zero

    # Metric ratios (means) — pre vs origin peers, pre vs dest peers, and post vs dest peers
    for short_name, metric_col in benchmark_metrics.items():
        mover_avg_pre_origin = mover_patents_at_origin[metric_col].mean()
        peer_avg_origin      = peer_patents_at_origin[metric_col].mean()
        peer_avg_dest_pre    = peer_patents_at_dest_pre[metric_col].mean()
        mover_avg_post_dest  = mover_patents_at_dest_post[metric_col].mean()
        peer_avg_dest_post   = peer_patents_at_dest_post[metric_col].mean()

        move_context[f'{short_name}_vs_origin_peers']      = (mover_avg_pre_origin / (peer_avg_origin + epsilon)
                                                              if pd.notna(mover_avg_pre_origin) else np.nan)
        move_context[f'{short_name}_vs_dest_peers']        = (mover_avg_pre_origin / (peer_avg_dest_pre + epsilon)
                                                              if pd.notna(mover_avg_pre_origin) else np.nan)
        move_context[f'{short_name}_vs_post_dest_peers']   = (mover_avg_post_dest / (peer_avg_dest_post + epsilon)
                                                              if pd.notna(mover_avg_post_dest) else np.nan)  

    # ---- count of patents ratios (pre & post)
    move_context['num_patents_vs_origin_peers']    = (
        mover_patents_at_origin['patent_id'].nunique() /
        (peer_patents_at_origin['patent_id'].nunique() + epsilon)
    )
    move_context['num_patents_vs_dest_peers']      = (
        mover_patents_at_origin['patent_id'].nunique() /
        (peer_patents_at_dest_pre['patent_id'].nunique() + epsilon)
    )
    move_context['num_patents_vs_post_dest_peers'] = (
        mover_patents_at_dest_post['patent_id'].nunique() /
        (peer_patents_at_dest_post['patent_id'].nunique() + epsilon)
    )

    # ---- total citations ratios (pre & post)
    move_context['citations_total_vs_origin_peers']    = (
        mover_patents_at_origin['cites'].sum() /
        (peer_patents_at_origin['cites'].sum() + epsilon)
    )
    move_context['citations_total_vs_dest_peers']      = (
        mover_patents_at_origin['cites'].sum() /
        (peer_patents_at_dest_pre['cites'].sum() + epsilon)
    )
    move_context['citations_total_vs_post_dest_peers'] = (
        mover_patents_at_dest_post['cites'].sum() /
        (peer_patents_at_dest_post['cites'].sum() + epsilon)
    )

    results.append(move_context)

mover_benchmark_df = pd.DataFrame(results)
mover_benchmark_df.to_pickle(os.path.join(INTERMEDIATE_PATH, 'mover_peer_benchmarks.pkl'))
print("\tPeer benchmarking with granular quality metrics complete.")
print(f"[Mover peer benchmarks] size: {mover_benchmark_df.shape[0]:,} rows × {mover_benchmark_df.shape[1]:,} cols", flush=True)


