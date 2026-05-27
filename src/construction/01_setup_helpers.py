"""
01_setup_helpers.py

Setup, imports, project paths, helper loader, and global environment configuration. 

================================================================================
SCRIPT OVERVIEW 
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
BASE_PROJECT_PATH = os.environ.get("MANDA_PROJECT_PATH", ".")
BASE_PROJECT_PATH = os.path.abspath(os.path.expanduser(BASE_PROJECT_PATH))

VERSION = int(os.environ.get("MANDA_VERSION", "1"))

OUTPUT_PATH = os.path.join(BASE_PROJECT_PATH, f"Patents/Data/Inventor_Mobility__v{VERSION}")
RAW_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "Patents/Data/Raw_Patentsview")
INTERMEDIATE_PATH = os.path.join(OUTPUT_PATH, "intermediate_files")

FINANCIAL_DATA_PATH = os.environ.get(
    "MANDA_FINANCIAL_DATA_PATH",
    os.path.join(BASE_PROJECT_PATH, "WRDS Data")
)

MANDA_DATA_PATH = os.environ.get(
    "MANDA_DATA_PATH",
    os.path.join(BASE_PROJECT_PATH, "SDC Data 1993 - 2018/MandA")
)

LINKTABLE_CSV = os.environ.get(
    "MANDA_LINKTABLE_CSV",
    os.path.join(BASE_PROJECT_PATH, "linktable.csv")
)

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(INTERMEDIATE_PATH, exist_ok=True)

def assert_required_paths_exist():
    required_paths = [
        BASE_PROJECT_PATH,
        RAW_DATA_PATH,
        FINANCIAL_DATA_PATH,
        MANDA_DATA_PATH,
        LINKTABLE_CSV,
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")

assert_required_paths_exist()

# --- Pandas Display Options & tqdm Integration ---
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

tqdm.pandas()
print("Setup complete.")

# %%
#################################################################
# SECTION 2 & 3: HELPERS & FOUNDATIONAL ECONOMIC DATA
#################################################################
print("--- Section 2 & 3: Loading Helpers & Foundational Economic Data ---")
# Helper functions and loading of pre-built Compustat/M&A panels. This is a critical step for a full run.
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

