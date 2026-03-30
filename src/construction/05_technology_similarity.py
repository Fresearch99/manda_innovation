"""
05_technology_similarity.py

Technology similarity measures around inventor mobility events, including event-based and rolling measures.
"""

# %%
#################################################################
# SECTION 11: CALCULATE TECHNOLOGY SIMILARITY FOR MOVES
#################################################################
print("--- Section 11: Building Event-Based Technology Proximity Measures ---")

# --- Helper Function (Used by both main functions) ---
def _calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors represented as Counters."""
    if not isinstance(vec1, Counter) or not isinstance(vec2, Counter):
        raise TypeError("Input vectors must be of type collections.Counter")
        
    # Check for empty vectors which would result in division by zero
    if not vec1 or not vec2:
        return np.nan

    # Find the union of all technology classes
    all_keys = vec1.keys() | vec2.keys()

    # Calculate dot product
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)

    # Calculate magnitude (norm) of each vector
    norm1 = sqrt(sum(v**2 for v in vec1.values()))
    norm2 = sqrt(sum(v**2 for v in vec2.values()))

    # Calculate cosine similarity
    if norm1 == 0 or norm2 == 0:
        return np.nan
    
    return dot_product / (norm1 * norm2)

# --- Function 1: Event-Based Proximity  ---
def build_event_based_proximity_measures(pat_inv_firm_df, inventor_moves_df, window_years=5, tech_col='cpc_subclass'):
    """Calculate technology proximity scores by comparing pre- and post-move patent portfolios."""
    print("\tPreparing data for event-based proximity analysis...", flush=True)
    
    # 1. Prepare dataframes
    cols = ['patent_id', 'filing_year', 'permco', 'inventor_id', tech_col]
    pat_df = pat_inv_firm_df.loc[:, cols].dropna(subset=[tech_col]).copy()
    moves_df = inventor_moves_df[['inventor_id', 'movement_year', 'from_permco', 'to_permco']].drop_duplicates().copy()

    # 2. Pre-calculate annual technology vectors for all entities
    print("\tPre-calculating annual technology vectors for firms and inventors...", flush=True)
    # Using nunique to count unique patents per class
    firm_vectors = pat_df.groupby(['permco', 'filing_year', tech_col])['patent_id'].nunique().to_dict()
    inv_vectors = pat_df.groupby(['inventor_id', 'filing_year', tech_col])['patent_id'].nunique().to_dict()
    
    # Convert dictionary to a more usable format: {(entity, year): Counter}
    def _restructure_vectors(vector_dict):
        restructured = defaultdict(Counter)
        for (entity, year, tech), count in vector_dict.items():
            restructured[(entity, year)][tech] += count
        return restructured
        
    firm_annual_vectors = _restructure_vectors(firm_vectors)
    inventor_annual_vectors = _restructure_vectors(inv_vectors)

    # 3. Iterate through each unique move and calculate similarities
    results = []
    unique_moves = moves_df.drop_duplicates(subset=['inventor_id', 'movement_year'])
    print("\tCalculating proximity scores for each inventor move...", flush=True)
    
    for row in tqdm(unique_moves.itertuples(index=False), total=len(unique_moves)):
        inv_id, move_year, from_firm, to_firm = row.inventor_id, row.movement_year, row.from_permco, row.to_permco
        
        # --- Define time windows ---
        pre_years = range(move_year - window_years + 1, move_year + 1)
        post_years = range(move_year + 1, move_year + window_years + 1)

        # --- Aggregate vectors for pre and post periods ---
        def _aggregate_vector(entity_id, years, annual_vectors):
            agg_vec = Counter()
            if pd.notna(entity_id):
                for year in years:
                    agg_vec.update(annual_vectors.get((entity_id, year), Counter()))
            return agg_vec

        inv_pre_vec = _aggregate_vector(inv_id, pre_years, inventor_annual_vectors)
        inv_post_vec = _aggregate_vector(inv_id, post_years, inventor_annual_vectors)
        
        from_firm_pre_vec = _aggregate_vector(from_firm, pre_years, firm_annual_vectors)
        from_firm_post_vec = _aggregate_vector(from_firm, post_years, firm_annual_vectors)

        to_firm_pre_vec = _aggregate_vector(to_firm, pre_years, firm_annual_vectors)
        to_firm_post_vec = _aggregate_vector(to_firm, post_years, firm_annual_vectors)
        
        # --- Calculate All Similarities ---
        results.append({
            'inventor_id': inv_id,
            'movement_year': move_year,
            'sim_inv_pre_post': _calculate_cosine_similarity(inv_pre_vec, inv_post_vec),
            'sim_from_firm_pre_post': _calculate_cosine_similarity(from_firm_pre_vec, from_firm_post_vec),
            'sim_to_firm_pre_post': _calculate_cosine_similarity(to_firm_pre_vec, to_firm_post_vec),
            'sim_inv_from_firm_pre': _calculate_cosine_similarity(inv_pre_vec, from_firm_pre_vec),
            'sim_inv_to_firm_pre': _calculate_cosine_similarity(inv_pre_vec, to_firm_pre_vec),
            'sim_inv_to_firm_post': _calculate_cosine_similarity(inv_post_vec, to_firm_post_vec),
            'sim_from_to_firm_pre': _calculate_cosine_similarity(from_firm_pre_vec, to_firm_pre_vec),
        })
        
    return pd.DataFrame(results)

# --- Function 2: Rolling Annual Similarity ---
def build_rolling_annual_similarity_measures(pat_inv_firm_df, inventor_moves_df, window_years=5, tech_col='cpc_subclass'):
    """Calculate rolling annual technology similarity of a year's patents against a prior 5-year window."""
    print("\tAggregating annual patent data for rolling similarity...", flush=True)

    # 1. Pre-calculate annual technology vectors for all entities
    cols = ['filing_year', 'permco', 'inventor_id', tech_col]
    pat_df = pat_inv_firm_df.loc[:, cols].dropna(subset=[tech_col]).copy()
    
    def _create_annual_vectors(df, entity_col):
        # Group by entity, year, and tech class, counting unique patents
        counts = df.groupby([entity_col, 'filing_year', tech_col])['filing_year'].count().rename('count')
        # Convert to a dictionary of Counters: {(entity_id, year): Counter}
        vector_dict = defaultdict(Counter)
        for idx, count in counts.items():
            entity_id, year, tech = idx
            vector_dict[(entity_id, year)][tech] = count
        return vector_dict

    inventor_annual_vectors = _create_annual_vectors(pat_df, 'inventor_id')
    firm_annual_vectors = _create_annual_vectors(pat_df, 'permco')

    # 2. Iterate chronologically through years to calculate rolling similarities
    results = []
    unique_moves = inventor_moves_df.drop_duplicates(subset=['inventor_id', 'movement_year'])
    
    min_year = pat_df['filing_year'].min()
    max_year = pat_df['filing_year'].max()

    print(f"\tCalculating rolling similarities from {min_year} to {max_year}...", flush=True)

    # Loop through each unique move context
    for row in tqdm(unique_moves.itertuples(index=False), total=len(unique_moves), desc="Processing Moves"):
        inv_id, move_year, from_firm, to_firm = row.inventor_id, row.movement_year, row.from_permco, row.to_permco
        
        # History deques and aggregate counters for this specific move
        inv_hist_dq, from_firm_hist_dq, to_firm_hist_dq = deque(), deque(), deque()
        inv_hist_vec, from_firm_hist_vec, to_firm_hist_vec = Counter(), Counter(), Counter()

        # Loop through all years in the dataset for this move context
        for year in range(int(min_year), int(max_year) + 1):
            cutoff = year - window_years

            # --- Purge old data from rolling window ---
            def _purge(dq, hist_vec):
                while dq and dq[0][0] < cutoff:
                    old_year, old_vec = dq.popleft()
                    hist_vec.subtract(old_vec)
                return hist_vec
            
            inv_hist_vec = _purge(inv_hist_dq, inv_hist_vec)
            from_firm_hist_vec = _purge(from_firm_hist_dq, from_firm_hist_vec)
            to_firm_hist_vec = _purge(to_firm_hist_dq, to_firm_hist_vec)
            
            # --- Get current year's vectors ---
            inv_current_vec = inventor_annual_vectors.get((inv_id, year), Counter())
            from_current_vec = firm_annual_vectors.get((from_firm, year), Counter())
            to_current_vec = firm_annual_vectors.get((to_firm, year), Counter())
            
            # Only calculate similarity if the inventor was active this year
            if not inv_current_vec:
                continue

            # --- Calculate Similarities ---
            sim_inv_self = _calculate_cosine_similarity(inv_current_vec, inv_hist_vec)
            sim_inv_from_firm = _calculate_cosine_similarity(inv_current_vec, from_firm_hist_vec)
            sim_inv_to_firm = _calculate_cosine_similarity(inv_current_vec, to_firm_hist_vec)

            # Calculate self-similarity for the from and to firms
            sim_from_firm_self = _calculate_cosine_similarity(from_current_vec, from_firm_hist_vec)
            sim_to_firm_self = _calculate_cosine_similarity(to_current_vec, to_firm_hist_vec)

            results.append({
                'inventor_id': inv_id,
                'movement_year': move_year, # To identify the move context
                'filing_year': year,
                'rol_sim_inv_self': sim_inv_self,
                'rol_sim_inv_vs_from_firm': sim_inv_from_firm,
                'rol_sim_inv_vs_to_firm': sim_inv_to_firm,
                'rol_sim_from_firm_self': sim_from_firm_self,
                'rol_sim_to_firm_self': sim_to_firm_self,
            })

            # --- Update rolling histories ---
            from_current_vec = firm_annual_vectors.get((from_firm, year), Counter())
            to_current_vec = firm_annual_vectors.get((to_firm, year), Counter())

            inv_hist_dq.append((year, inv_current_vec))
            inv_hist_vec.update(inv_current_vec)
            
            if from_current_vec:
                from_firm_hist_dq.append((year, from_current_vec))
                from_firm_hist_vec.update(from_current_vec)

            if to_current_vec:
                to_firm_hist_dq.append((year, to_current_vec))
                to_firm_hist_vec.update(to_current_vec)

    return pd.DataFrame(results)

# --- Function 3: Firm-Year Rolling Self-Similarity ---
def build_firm_rolling_self_similarity(pat_inv_firm_df, window_years=5, tech_col='cpc_subclass'):
    """Calculate rolling annual technology self-similarity for all firms."""
    print("\tAggregating annual patent data for all firms...", flush=True)

    # 1. Prepare data and create annual technology vectors for all firms
    # Note: we can restrict here to assignees with permco as we are only going to use the measure on
    #   firm-level
    cols = ['permco', 'filing_year', tech_col]
    firm_pat_df = pat_inv_firm_df.loc[:, cols].dropna(subset=['permco', tech_col]).copy()
    
    # Create a dictionary of Counters: {(firm_id, year): Counter}
    firm_annual_vectors = defaultdict(Counter)
    counts = firm_pat_df.groupby(['permco', 'filing_year', tech_col])['filing_year'].count()
    for (firm_id, year, tech), count in counts.items():
        firm_annual_vectors[(firm_id, year)][tech] = count
        
    # 2. Iterate chronologically to calculate rolling similarities
    results = []
    min_year, max_year = int(firm_pat_df['filing_year'].min()), int(firm_pat_df['filing_year'].max())
    
    # Data structures to hold the rolling window history for each firm
    firm_histories_dq = defaultdict(deque)
    firm_histories_vec = defaultdict(Counter)

    print(f"\tCalculating rolling self-similarity from {min_year} to {max_year}...", flush=True)
    
    for year in tqdm(range(min_year, max_year + 1), desc="Processing Years"):
        cutoff = year - window_years
        
        # Identify firms active in the current year
        active_firms_this_year = {
            firm_id for (firm_id, yr) in firm_annual_vectors.keys() if yr == year
        }
        
        # Process each firm that has patents in the current year
        for firm_id in active_firms_this_year:
            
            # --- Purge old data from this firm's rolling window ---
            # This brings the history window to [year-5, year-1]
            while firm_histories_dq[firm_id] and firm_histories_dq[firm_id][0][0] < cutoff:
                old_year, old_vec = firm_histories_dq[firm_id].popleft()
                firm_histories_vec[firm_id].subtract(old_vec) # Efficiently subtract counts
            
            # --- Get current and historical vectors for calculation ---
            current_year_vector = firm_annual_vectors.get((firm_id, year), Counter())
            historical_vector = firm_histories_vec[firm_id]
            
            # --- Calculate Similarity ---
            similarity_score = _calculate_cosine_similarity(current_year_vector, historical_vector)
            
            if pd.notna(similarity_score):
                results.append({
                    'permco': firm_id,
                    'year': year,
                    'rolling_self_similarity': similarity_score,
                })
            
            # --- Update this firm's history for the *next* iteration ---
            # The current year's vector now becomes part of the history
            firm_histories_dq[firm_id].append((year, current_year_vector))
            firm_histories_vec[firm_id].update(current_year_vector)

    return pd.DataFrame(results)

# --- Execute Analysis and Save Results ---
#pat_inv_firm_df['filing_year'] = pd.to_numeric(pat_inv_firm_df['filing_year'], errors='coerce').astype('Int64')
# -> should not be needed based on construction

# 1. Calculate and save event-based proximity measures
print("Calculating event-based proximity measures...")
event_prox_df = build_event_based_proximity_measures(pat_inv_firm_df, mover_events_df)
event_prox_df.to_pickle(os.path.join(INTERMEDIATE_PATH, 'technology_proximity_event_based.pkl'))
print(f"[Event proximity] size: {event_prox_df.shape[0]:,} rows × {event_prox_df.shape[1]:,} cols", flush=True)

# 2. Calculate and save rolling annual similarity measures for inventor moves
print("Calculating rolling annual similarity measures for events...")
rolling_sim_df = build_rolling_annual_similarity_measures(pat_inv_firm_df, mover_events_df)
rolling_sim_df.to_pickle(os.path.join(INTERMEDIATE_PATH, 'technology_proximity_rolling_event_based.pkl'))
print(f"[Rolling similarity] size: {rolling_sim_df.shape[0]:,} rows × {rolling_sim_df.shape[1]:,} cols", flush=True)

# 3. Calculate and save the firm-year self-similarity panel
print("Calculating firm-year rolling self-similarity panel...")
firm_year_similarity_panel = build_firm_rolling_self_similarity(pat_inv_firm_df)
firm_year_similarity_panel.to_pickle(os.path.join(INTERMEDIATE_PATH, 'firm_year_rolling_self_similarity.pkl'))
print(f"[Firm-year self-similarity] size: {firm_year_similarity_panel.shape[0]:,} rows × {firm_year_similarity_panel.shape[1]:,} cols", flush=True)

print("\nTechnology proximity and similarity measures successfully calculated and saved.", flush=True)


