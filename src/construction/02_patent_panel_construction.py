"""
02_patent_panel_construction.py

Patent-data loading and the core patent-inventor-firm dataset construction. This file covers the early heavy-lift data preparation stage and preserves the original load/clean/build logic.

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
# SECTION 4: LOAD & PRE-PROCESS RAW PATENT DATA
#################################################################
print("--- Section 4: Loading and Pre-processing Raw Patent Data ---")

def build_or_load_pat_inv_firm(cache_path=None, rebuild=False):
    """
    Load prebuilt patent-inventor-firm DF if available; otherwise
    run Section 4 load/clean + construction, save, and return.
    """
    if cache_path is None:
        cache_path = os.path.join(INTERMEDIATE_PATH, 'pat_inv_firm_df.pkl')

    if (not rebuild) and os.path.exists(cache_path):
        print("\tLoading pre-built patent-inventor-firm dataset...", flush=True)
        df = pd.read_pickle(cache_path)
        print(f"[Patent-inventor-firm] loaded: {df.shape[0]:,} rows × {df.shape[1]:,} cols", flush=True)
        return df

    inv_df = download_and_load_patentsview_data('g_inventor_disambiguated.tsv', usecols=['patent_id', 'inventor_id', 'location_id'])
    assignee_df = download_and_load_patentsview_data('g_assignee_disambiguated.tsv', usecols=['patent_id', 'assignee_id', 'assignee_type', 'assignee_sequence'])
    location_df = download_and_load_patentsview_data('g_location_disambiguated.tsv', usecols=['location_id', 'disambig_state', 'state_fips'])
    cpc_df = download_and_load_patentsview_data('g_cpc_current.tsv', usecols=['patent_id', 'cpc_group', 'cpc_subclass', 'cpc_type', 'cpc_sequence'])
    citation_df = download_and_load_patentsview_data('g_us_patent_citation.tsv', usecols=['patent_id', 'citation_patent_id'])


    # Clean and Prep
    inv_df['patent_id'] = pd.to_numeric(inv_df['patent_id'], errors='coerce')
    inv_df = inv_df.dropna(subset=['patent_id'])
    
    cpc_df['patent_id'] = pd.to_numeric(cpc_df['patent_id'], errors='coerce')
    cpc_df = cpc_df.dropna(subset=['patent_id'])
    cpc_df = cpc_df[(cpc_df['cpc_type'] == 'inventional')].copy()  # keep full (all sequences) for later use
    
    assignee_df = assignee_df[(assignee_df['assignee_type'] == 2) & (assignee_df['assignee_sequence'] == 0)].copy()
    assignee_df['patent_id'] = pd.to_numeric(assignee_df['patent_id'], errors='coerce')
    assignee_df = assignee_df.dropna(subset=['patent_id'])
    assignee_df.drop(columns=['assignee_type', 'assignee_sequence'], inplace=True)
    
    location_df['state_fips'] = pd.to_numeric(location_df['state_fips'], errors='coerce')\
                                 .apply(lambda x: f"{int(x):02d}" if pd.notna(x) else pd.NA)

    citation_df['patent_id'] = pd.to_numeric(citation_df['patent_id'], errors='coerce')
    citation_df['citation_patent_id'] = pd.to_numeric(citation_df['citation_patent_id'], errors='coerce')
    citation_df = citation_df.dropna(subset=['patent_id', 'citation_patent_id'])
    
    # External Datasets
    # Source: https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data
    KPSS_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'Replication Data/KPSS_data__update2024')
    kpss_patents_df = pd.read_csv(os.path.join(KPSS_DATA_PATH, 'KPSS_2023.csv'), usecols=['patent_num', 'xi_real', 'cites'])
    kpss_patents_df.rename(columns={'patent_num': 'patent_id'}, inplace=True)
    kpss_patents_df['patent_id'] = pd.to_numeric(kpss_patents_df['patent_id'], errors='coerce')
    
    kpss_link_df = pd.read_csv(os.path.join(KPSS_DATA_PATH, 'Match_patent_permco_permno_2023.csv'), usecols=['patent_num', 'permco'])
    kpss_link_df.rename(columns={'patent_num': 'patent_id'}, inplace=True)
    kpss_df = kpss_patents_df.merge(kpss_link_df, on='patent_id', how='left').drop_duplicates().dropna(subset=['patent_id', 'permco'])
    
    # Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/37A0L2
    # Paper: https://www.nber.org/papers/w31929
    # see also for an application to innovation: https://www.nber.org/papers/w31487 
    # => state of residence of the inventor is relevant
    NCA_ENFORCEMENT_PATH = os.path.join(BASE_PROJECT_PATH, 'Replication Data/Johnson et al 2023 - NCA Enforcement/NCA-data-OSF')
    nca_df = pd.read_stata(os.path.join(NCA_ENFORCEMENT_PATH, 'NCA_law_panel_1991-2014.dta'))
    nca_df.rename(columns={'year': 'filing_year'}, inplace=True)
    nca_df['state_fips'] = pd.to_numeric(nca_df['state_fips'], errors='coerce')\
                            .apply(lambda x: f"{int(x):02d}" if pd.notna(x) else pd.NA)

    
    # ---- cache guard for patent_df core ----
    PATENT_DF_PATH = os.path.join(INTERMEDIATE_PATH, 'patent_df_core.pkl')
    use_patent_cache = (not rebuild) and os.path.exists(PATENT_DF_PATH)
    
    if use_patent_cache:
        print("\tLoading cached patent_df core...", flush=True)
        patent_df = pd.read_pickle(PATENT_DF_PATH)
    else:
        print("\tBuilding patent_df core...", flush=True)
        patent_df = download_and_load_patentsview_data('g_patent.tsv', usecols=['patent_id', 'patent_date'])
        application_df = download_and_load_patentsview_data('g_application.tsv', usecols=['patent_id', 'filing_date'])
    
        # Clean and Prep
        patent_df['patent_date'] = pd.to_datetime(patent_df['patent_date'], errors='coerce')
        patent_df['patent_id'] = pd.to_numeric(patent_df['patent_id'], errors='coerce')
        patent_df = patent_df.dropna(subset=['patent_id'])
    
        application_df['filing_date'] = pd.to_datetime(application_df['filing_date'], errors='coerce')
        application_df['patent_id'] = pd.to_numeric(application_df['patent_id'], errors='coerce')
        application_df = application_df.dropna(subset=['patent_id'])
    
        patent_df = patent_df.merge(application_df[['patent_id', 'filing_date']],
                                    on='patent_id', how='left', validate='one_to_one')
        patent_df['filing_year'] = patent_df['filing_date'].dt.year
        patent_df['filing_year'] = pd.to_numeric(
            patent_df['filing_year'], errors='coerce').astype('Int64')

        
        cpc_primary_df = cpc_df[(cpc_df['cpc_sequence'] == 0)].copy()
        cpc_primary_df.drop(columns=['cpc_type', 'cpc_sequence'], inplace=True)
        patent_df = patent_df.merge(cpc_primary_df, on='patent_id', how='left', validate='one_to_one')
    
        # SAVE the built core
        os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
        patent_df.to_pickle(PATENT_DF_PATH)
        print(f"\tSaved patent_df core to {PATENT_DF_PATH}", flush=True)
    
        if 'application_df' in locals(): del application_df
        if 'cpc_primary_df' in locals(): del cpc_primary_df
    
    gc.collect()


    
    # %%
    #################################################################
    # SECTION 5: BUILD PATENT-LEVEL QUALITY & NOVELTY MEASURES
    #################################################################
    print("--- Section 5: Building Patent-Level Quality & Novelty Measures ---")
    
    def build_patent_quality_measures(patent_df, citation_df, kpss_df, output_dir):
        """
        Computes a suite of patent quality measures, including unconditional
        forward citations and granular, class-year normalized citation bins.
        """
        
        # 1. Unconditional Forward Citations
        fwd_cites_path = os.path.join(INTERMEDIATE_PATH, 'forward_citations_unconditional.pkl')
        if os.path.exists(fwd_cites_path):
            forward_citations = pd.read_pickle(fwd_cites_path)
        else:
            print("\tCalculating unconditional forward citations...")
            forward_citations = (citation_df.groupby('citation_patent_id')['patent_id'].nunique()
                                 .reset_index()
                                 .rename(columns={'citation_patent_id': 'patent_id', 'patent_id': 'forward_citations'}))
            forward_citations.to_pickle(fwd_cites_path)
            
        # 2. Forward Citations Normalized by CPC Class-Year
        norm_cites_path = os.path.join(INTERMEDIATE_PATH, 'forward_citations_normalized.pkl')
        if os.path.exists(norm_cites_path):
            print("\tLoading pre-built patent quality measures...")
            patent_quality_df = pd.read_pickle(norm_cites_path)
        else:
            print("\tCalculating class-year normalized citation quality bins...")
            # Prep data
            df = patent_df.merge(kpss_df[['patent_id', 'cites']], on='patent_id', how='left')
            df.rename(columns={'cites':'kpss_cites'}, inplace=True)
            df = df.merge(forward_citations, on='patent_id', how='left')
            df['forward_citations'] = df['forward_citations'].fillna(0)
            
            # Forward-citation quantiles (as before)
            stats = df.groupby(['filing_year', 'cpc_subclass'])['forward_citations'] \
                      .quantile([0.90, 0.99]).unstack()
            stats.rename(columns={0.90: 'q90_fwd_cites', 0.99: 'q99_fwd_cites'}, inplace=True)
            df = df.merge(stats, on=['filing_year', 'cpc_subclass'], how='left')
            
            # KPSS-citation quantiles (new)
            stats_kpss = df.groupby(['filing_year', 'cpc_subclass'])['kpss_cites'] \
                           .quantile([0.90, 0.99]).unstack()
            stats_kpss.rename(columns={0.90: 'q90_kpss_cites', 0.99: 'q99_kpss_cites'}, inplace=True)
            df = df.merge(stats_kpss, on=['filing_year', 'cpc_subclass'], how='left')
            
            # Forward bins (unchanged)
            df['top1_forward_citations'] = ((df['forward_citations'] > 0) & (df['forward_citations'] >= df['q99_fwd_cites'])).astype(int)
            df['top10_to_2_forward_citations'] = ((df['forward_citations'] > 0) & 
                                                  (df['forward_citations'] >= df['q90_fwd_cites']) &
                                                  (df['forward_citations'] < df['q99_fwd_cites'])).astype(int)
            df['cited_patent_forward_citations'] = ((df['forward_citations'] > 0) &
                                                    (df['forward_citations'] < df['q90_fwd_cites'])).astype(int)
            df['Uncited_patent_forward_citations'] = (df['forward_citations'] == 0).astype(int)
            
            # KPSS bins 
            df['top1_kpss_cites'] = ((df['kpss_cites'] > 0) & (df['kpss_cites'] >= df['q99_kpss_cites'])).astype(int)
            df['top10_to_2_kpss_cites'] = ((df['kpss_cites'] >= df['q90_kpss_cites']) &
                                           (df['kpss_cites'] < df['q99_kpss_cites'])).astype(int)
            df['cited_patent_kpss_cites'] = ((df['kpss_cites'] > 0) &
                                             (df['kpss_cites'] < df['q90_kpss_cites'])).astype(int)
            df['Uncited_patent_kpss_cites'] = (df['kpss_cites'] == 0).astype(int)
            
            # set bins to NA where KPSS cites is NA
            _mask = df['kpss_cites'].isna()
            for _col in ['top1_kpss_cites','top10_to_2_kpss_cites','cited_patent_kpss_cites','Uncited_patent_kpss_cites']:
                df.loc[_mask, _col] = pd.NA
            
            # Final selection of columns (add KPSS outputs)
            quality_cols = [
                'patent_id',
                'forward_citations', 'kpss_cites',
                'top1_forward_citations', 'top10_to_2_forward_citations',
                'cited_patent_forward_citations', 'Uncited_patent_forward_citations',
                'top1_kpss_cites', 'top10_to_2_kpss_cites',
                'cited_patent_kpss_cites', 'Uncited_patent_kpss_cites'
            ]
            patent_quality_df = df[quality_cols].drop_duplicates()
            patent_quality_df.to_pickle(norm_cites_path)
            
        return patent_quality_df
    
    
    def build_patent_novelty_measures(patent_df, cpc_df, output_dir):
        """
        Computes patent novelty based on new combinations of CPC classes,
        following the methodology of Arts & Fleming (2018).
        """
        novelty_path = os.path.join(INTERMEDIATE_PATH, 'patent_novelty.pkl')
    
        if os.path.exists(novelty_path):
            print("\tLoading pre-built patent novelty measures...")
            return pd.read_pickle(novelty_path)
        
        print("\tBuilding patent novelty (new CPC combinations)...")
        
        # --- Construct the input dataframe needed by the novelty logic ---
        _cpc = patent_df[['patent_id', 'patent_date']].merge(
            cpc_df[['patent_id', 'cpc_group', 'cpc_subclass']], on='patent_id'
        )
        _cpc.rename(columns={'patent_date': 'issue_date_dt', 'cpc_group': 'group_id', 'cpc_subclass': 'subclass_id'}, inplace=True)
        _cpc['issue_year'] = _cpc['issue_date_dt'].dt.year
        _cpc = _cpc[_cpc['issue_year'] >= 1976].drop_duplicates().copy()
        _cpc['issue_date_dt'] = pd.to_datetime(_cpc['issue_date_dt'], errors='coerce')
    
        def _canonical_pair(a, b): return f'{a}_{b}' if a <= b else f'{b}_{a}'
    
        def _compute_pair_novelty_for_level(level_col: str) -> pd.DataFrame:
            lv = level_col
            g = (_cpc[['patent_id', 'issue_date_dt', lv]]
                 .dropna(subset=['patent_id', 'issue_date_dt', lv])
                 .groupby(['patent_id', 'issue_date_dt'])[lv]
                 .agg(lambda s: tuple(sorted(set(s))))
                 .reset_index(name='cpc_tuple'))
            g['total_combinations'] = g['cpc_tuple'].apply(lambda t: len(list(itertools.combinations(t, 2))))
            g = g[g['total_combinations'] > 0].copy()
    
            def _make_pairs(tup): return [_canonical_pair(a, b) for a, b in itertools.combinations(tup, 2)]
            pairs = (g[['patent_id', 'issue_date_dt', 'cpc_tuple']]
                     .assign(pair_key=lambda df: df['cpc_tuple'].apply(_make_pairs))
                     .explode('pair_key').drop(columns='cpc_tuple').dropna(subset=['pair_key']))
    
            first_date_per_pair = pairs.groupby('pair_key')['issue_date_dt'].min().reset_index(name='first_issue_date')
            pairs = pairs.merge(first_date_per_pair, on='pair_key', how='left')
            
            first_day_rows = pairs[pairs['issue_date_dt'] == pairs['first_issue_date']].copy()
            first_day_rows['is_new'] = True # all patents issued on the first_issue_date are first occurances.
            
            # Alternative: select one patent as the first occurance:
            #winner = first_day_rows.groupby('pair_key')['patent_id'].min().reset_index(name='winner_patent_id')
            #first_day_rows = first_day_rows.merge(winner, on='pair_key')
            #first_day_rows['is_new'] = (first_day_rows['patent_id'] == first_day_rows['winner_patent_id'])
            
            
            pairs = pairs.merge(first_day_rows[['patent_id', 'pair_key', 'is_new']], on=['patent_id', 'pair_key'], how='left')
            pairs['is_new'] = pairs['is_new'].fillna(False)
    
            novelty = pairs.groupby('patent_id').agg(
                new_combinations=('is_new', 'sum'),
                total_combinations=('pair_key', 'nunique')
            ).reset_index()
            
            novelty.rename(columns={
                'new_combinations': f'new_combinations_{lv}',
                'total_combinations': f'total_combinations_{lv}'
            }, inplace=True)
            
            all_patents = _cpc[['patent_id']].drop_duplicates()
            return all_patents.merge(novelty, on='patent_id', how='left').fillna(0)
    
        nov_group = _compute_pair_novelty_for_level('group_id')
        nov_subclass = _compute_pair_novelty_for_level('subclass_id')
        
        nov = nov_group.merge(nov_subclass, on='patent_id', how='outer')
        nov['novelty_score_group'] = nov['new_combinations_group_id'] / nov['total_combinations_group_id']
        nov['novelty_score_subclass'] = nov['new_combinations_subclass_id'] / nov['total_combinations_subclass_id']
        nov.fillna(0, inplace=True)
    
        final_cols = ['patent_id', 'novelty_score_group', 'novelty_score_subclass',
                      'new_combinations_group_id', 'total_combinations_group_id',
                      'new_combinations_subclass_id', 'total_combinations_subclass_id']
        novelty_df = nov[final_cols]
        novelty_df.to_pickle(novelty_path)
        print("\tPatent novelty measures successfully built.")
        return novelty_df
    
    
    def build_citation_link_measures(citation_df, assignee_df, output_dir, chunk_rows=2000000):
        """
        Computes backward citation counts and firm self-citation counts for each patent,
        memory-safely, keeping assignee_id as strings (no compression to ints).
        """
        citations_path = os.path.join(INTERMEDIATE_PATH, 'patent_citation_links.pkl')
    
        if os.path.exists(citations_path):
            print("\tLoading pre-built backward and self-citation measures...")
            return pd.read_pickle(citations_path)
    
        print("\tBuilding backward and self-citation measures...")
    
        # --- 0) Keep only required columns and dedupe edges early ---
        edges = (citation_df[['patent_id', 'citation_patent_id']]
                 .dropna(subset=['patent_id', 'citation_patent_id'])
                 .drop_duplicates()
                 .reset_index(drop=True))
    
        # --- 1) Backward citations: unique cited patents per citing patent ---
        backward_citations = (edges.groupby('patent_id', sort=False)['citation_patent_id']
                                   .nunique()
                                   .reset_index(name='backward_citations'))
    
        # --- 2) Build patent -> set(assignee_id) mapping (keeps all assignees; stays as char) ---
        pat_to_assignees = (assignee_df[['patent_id', 'assignee_id']]
                            .dropna()
                            .drop_duplicates()
                            .groupby('patent_id')['assignee_id']
                            .agg(lambda s: frozenset(s))  # compact, hashable
                           )
    
        # --- 3) Self-citations via set intersection (chunked; no big joins) ---
        self_counts = {}
        empty = frozenset()
    
        n = len(edges)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            chunk = edges.iloc[start:stop, :]
    
            # Map to assignee sets (keeps char IDs; no dtype changes)
            citing_sets = [pat_to_assignees.get(pid, empty) for pid in chunk['patent_id'].values]
            cited_sets  = [pat_to_assignees.get(pid, empty) for pid in chunk['citation_patent_id'].values]
    
            # Overlap => self-citation; dedupe (patent_id, citation_patent_id) to avoid multi-counting
            mask = [bool(a & b) for a, b in zip(citing_sets, cited_sets)]
            sub = chunk.loc[mask, ['patent_id', 'citation_patent_id']].drop_duplicates()
    
            if not sub.empty:
                counts = sub.groupby('patent_id', sort=False)['citation_patent_id'].nunique()
                for k, v in counts.items():
                    self_counts[k] = self_counts.get(k, 0) + int(v)
    
            # Free per-chunk intermediates
            del chunk, citing_sets, cited_sets, sub
            gc.collect()
    
        # Materialize self-citation frame
        if self_counts:
            self_citations = (pd.Series(self_counts, name='self_citations')
                                .astype('int32')
                                .reset_index().rename(columns={'index': 'patent_id'}))
        else:
            self_citations = pd.DataFrame({
                'patent_id': pd.Series(dtype=edges['patent_id'].dtype),
                'self_citations': pd.Series(dtype='int32')
            })
    
        # --- 4) Merge results and save (same filename as before) ---
        citation_measures_df = backward_citations.merge(self_citations, on='patent_id', how='left')
        citation_measures_df['self_citations'] = citation_measures_df['self_citations'].fillna(0).astype('int32')
    
        citation_measures_df.to_pickle(citations_path)
        print("\tBackward and self-citation measures successfully built.")
        return citation_measures_df
    
    
    # Execute all patent measure builds
    patent_quality_df = build_patent_quality_measures(patent_df, citation_df, kpss_df, OUTPUT_PATH)
    patent_novelty_df = build_patent_novelty_measures(patent_df, cpc_df, OUTPUT_PATH)
    citation_links_df = build_citation_link_measures(citation_df, assignee_df, OUTPUT_PATH)
    del citation_df, assignee_df
    gc.collect()
    
    # %%
    #################################################################
    # SECTION 6: CONSTRUCT THE CORE PATENT-INVENTOR-FIRM DATASET
    #################################################################
    print("--- Section 6: Constructing the Core Patent-Inventor-Firm Dataset ---")
    # Start with inventor-patent link
    pat_inv_df = inv_df.merge(patent_df, on='patent_id', how='inner')
    # Add firm link via KPSS link
    pat_inv_firm_df = pat_inv_df.merge(kpss_df, on=['patent_id'], how='inner')
    # Merge all patent-level characteristics
    pat_inv_firm_df = pat_inv_firm_df.merge(patent_novelty_df, on='patent_id', how='left') 
    pat_inv_firm_df = pat_inv_firm_df.merge(patent_quality_df, on='patent_id', how='left')
    pat_inv_firm_df = pat_inv_firm_df.merge(citation_links_df, on='patent_id', how='left')
    # Merge location and NCA enforceability
    pat_inv_firm_df = pat_inv_firm_df.merge(location_df, on='location_id', how='left')
    pat_inv_firm_df = pat_inv_firm_df.merge(
        nca_df[['filing_year', 'state_fips', 'std_score']],
        on=['filing_year', 'state_fips'], how='left'
    ).rename(columns={'std_score': 'nca_enforce_score'})
    # Add team size and inventor career metrics
    pat_inv_firm_df['team_size'] = pat_inv_firm_df.groupby('patent_id')['inventor_id'].transform('nunique')
    pat_inv_firm_df.sort_values(['inventor_id', 'filing_date'], inplace=True)
    first_patent_info = pat_inv_firm_df.groupby('inventor_id').agg(
        first_filing_year=('filing_year', 'first'),
        first_cpc_subclass=('cpc_subclass', 'first'),
        first_cpc_group=('cpc_group', 'first'))
    pat_inv_firm_df = pat_inv_firm_df.merge(first_patent_info, on='inventor_id', how='left')
    pat_inv_firm_df['inventor_age'] = pat_inv_firm_df['filing_year'] - pat_inv_firm_df['first_filing_year']
    pat_inv_firm_df['same_first_cpc_subclass'] = pat_inv_firm_df['cpc_subclass'].eq(pat_inv_firm_df['first_cpc_subclass'])
    pat_inv_firm_df['same_first_cpc_group'] = pat_inv_firm_df['cpc_group'].eq(pat_inv_firm_df['first_cpc_group'])
    pat_inv_firm_df = pat_inv_firm_df.reset_index(drop=True)

    print(f"Core patent-inventor-firm dataset created with {len(pat_inv_firm_df)} rows.")
    print(f"[Patent-inventor-firm] size: {pat_inv_firm_df.shape[0]:,} rows × {pat_inv_firm_df.shape[1]:,} cols", flush=True)
    
    # Save cache and free memory (locals go out of scope anyway, this is extra-safe)
    pat_inv_firm_df.to_pickle(cache_path)
    print(f"\tSaved to {cache_path}", flush=True)
    del inv_df, patent_df, cpc_df, kpss_patents_df, kpss_link_df, kpss_df, location_df, nca_df, pat_inv_df, first_patent_info
    gc.collect()
    
    return pat_inv_firm_df

# Build once or load from cache with zero extra I/O
pat_inv_firm_df = build_or_load_pat_inv_firm(
    cache_path=os.path.join(INTERMEDIATE_PATH, 'pat_inv_firm_df.pkl'),
    rebuild=False,   # flip to True to refresh cache after upstream changes
)
