#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study: The Impact of Mergers and Acquisitions on Inventor Mobility and Performance
Author: Dominik Jurek
Revised by: Skilled Tech Economist AI
Revision Date: 2024-05-18

================================================================================
SCRIPT OVERVIEW (v11 Update)
================================================================================
This script constructs a comprehensive set of panel datasets to analyze the
relationship between M&A activity and inventor mobility. It serves as the master
data construction pipeline, transforming raw data sources into analysis-ready files.
"""

# %%
#################################################################
# SECTION 1: SETUP AND CONFIGURATION
#################################################################
print("--- Section 1: Setup and Configuration ---")

# ---------------------------------------------------------------
# 1.1. Core Packages
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import csv
import gc
import re
import itertools
from io import BytesIO
from collections import Counter, defaultdict, deque
from math import sqrt
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

from sklearn.exceptions import ConvergenceWarning
import warnings


# ---------------------------------------------------------------
# 1.2. Project Configuration
# ---------------------------------------------------------------
BASE_PROJECT_PATH = r'/Users/dominikjurek/Library/CloudStorage/Dropbox/University/PhD Berkeley/Research'
VERSION = 2  # Updated version for this revision

# --- Derived Data Paths ---
OUTPUT_PATH = os.path.join(BASE_PROJECT_PATH, f'Patents/Data/Inventor_Mobility__v{VERSION}')
RAW_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'Patents/Data/Raw_Patentsview')
INTERMEDIATE_PATH = os.path.join(OUTPUT_PATH, 'intermediate_files')

FINANCIAL_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'WRDS Data')
MANDA_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'SDC Data 1993 - 2018/MandA')


# --- Source Data Paths ---
# (Assume paths are correctly set)

# Create output directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
os.chdir(OUTPUT_PATH)

# --- Pandas Display Options & tqdm Integration ---
pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 100); pd.set_option('display.float_format', '{:.3f}'.format)
tqdm.pandas()
print("Setup complete.")

# %%
#################################################################
# SECTION 2 & 3: HELPERS & FOUNDATIONAL ECONOMIC DATA
#################################################################
print("--- Section 2 & 3: Loading Helpers & Foundational Economic Data ---")
# Helper functions and loading of pre-built Compustat/M&A panels are assumed
# to be here for brevity. This is a critical step for a full run.
def download_and_load_patentsview_data(file_name, **kwargs):
    """Downloads a PatentsView TSV file if not present locally, then loads it."""
    base_url = 'https://s3.amazonaws.com/data.patentsview.org/download'
    local_file_path = os.path.join(RAW_DATA_PATH, file_name)
    if not os.path.exists(RAW_DATA_PATH): os.makedirs(RAW_DATA_PATH)
    if os.path.exists(local_file_path):
        print(f"\tLoading '{file_name}' from local directory.", flush=True)
    else:
        print(f"\tDownloading '{file_name}'...", flush=True)
        r = requests.get(f"{base_url}/{file_name}.zip", timeout=300)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as z: z.extractall(RAW_DATA_PATH)
    return pd.read_csv(local_file_path, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, low_memory=False, **kwargs)

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
# %%
#################################################################
# SECTION 7: BUILD EXPLORATION & EXPLOITATION MEASURES
#################################################################
print("--- Section 7: Building Exploration & Exploitation Measures ---")
def build_exploration_exploitation_measures(pat_inv_firm_df, window_years=5):
    """
    Calculate patent-level exploration/exploitation for inventors and firms
    (firm side keyed by KPSS permco). Uses ALL permcos per patent consistently.
    """
    print("\tAggregating patent-level data for exploration/exploitation metrics...", flush=True)

    # Canonicalize to one row per patent with: year, permcos, inventors, and CPC set
    cols = ['patent_id', 'filing_year', 'permco', 'inventor_id', 'cpc_subclass']
    df = pat_inv_firm_df.loc[:, cols].dropna(subset=['cpc_subclass']).copy()

    patent_level = (
        df.groupby('patent_id', sort=False)
          .agg(
              filing_year=('filing_year', 'first'),
              permcos=('permco', lambda s: tuple(sorted(set(s.dropna())))),
              inventors=('inventor_id', lambda s: tuple(sorted(set(s.dropna())))),
              cpcs=('cpc_subclass', lambda s: frozenset(s.dropna()))
          )
          .reset_index()
          .sort_values(['filing_year', 'patent_id'])
          .reset_index(drop=True)
    )

    inventor_hist = defaultdict(deque)  # key: inventor_id -> deque[(year, cpc_set)]
    firm_hist = defaultdict(deque)      # key: permco       -> deque[(year, cpc_set)]

    results = []

    print("\tCalculating exploration/exploitation scores patent-by-patent...", flush=True)
    for row in tqdm(patent_level.itertuples(index=False), total=len(patent_level)):
        year   = row.filing_year
        cutoff = year - window_years

        # Purge helper (left-anchored rolling window)
        def _purge(dq):
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            return dq

        # ---- Inventor prior knowledge (union of CPCs across all inventors' recent patents)
        inv_knowledge = set()
        for inv_id in row.inventors:
            dq = _purge(inventor_hist[inv_id])
            for _, hist_cpcs in dq:
                inv_knowledge.update(hist_cpcs)

        # ---- Firm prior knowledge (union across ALL permcos for this patent)
        firm_knowledge = set()
        for pc in row.permcos:  # may be empty tuple if no permco
            dq = _purge(firm_hist[pc])
            for _, hist_cpcs in dq:
                firm_knowledge.update(hist_cpcs)

        # ---- Compute exploration scores (share of CPCs NOT seen before)
        current_cpcs = row.cpcs
        if not current_cpcs:
            exp_inv  = np.nan
            exp_firm = np.nan
        else:
            denom     = len(current_cpcs)
            overlap_i = len(current_cpcs.intersection(inv_knowledge))
            overlap_f = len(current_cpcs.intersection(firm_knowledge))
            exp_inv   = 1 - (overlap_i / denom)
            exp_firm  = 1 - (overlap_f / denom)

        results.append({
            'patent_id': row.patent_id,
            'exploration_inv':  exp_inv,
            'exploitation_inv': 1 - exp_inv if np.isfinite(exp_inv) else np.nan,
            'exploration_firm': exp_firm,
            'exploitation_firm': 1 - exp_firm if np.isfinite(exp_firm) else np.nan,
        })

        # ---- Update rolling histories
        for inv_id in row.inventors:
            inventor_hist[inv_id].append((year, current_cpcs))
        for pc in row.permcos:
            firm_hist[pc].append((year, current_cpcs))

    return pd.DataFrame(results)


explore_exploit_df = build_exploration_exploitation_measures(pat_inv_firm_df)
pat_inv_firm_df = pat_inv_firm_df.merge(explore_exploit_df, on='patent_id', how='left')
pat_inv_firm_df.to_pickle(os.path.join(INTERMEDIATE_PATH, 'pat_inv_firm_df_fully_enriched.pkl'))
print("Core dataset now fully enriched with quality and exploration metrics.")
print(f"[Core dataset enriched] size: {pat_inv_firm_df.shape[0]:,} rows × {pat_inv_firm_df.shape[1]:,} cols", flush=True)

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
gc.collect()
