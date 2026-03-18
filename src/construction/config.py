"""
Central configuration for the manda_innovation construction pipeline.

This file mirrors the path logic from the original monolithic script, but puts the
editable path settings in one place so a user who clones the repository can adapt
local paths without touching the substantive construction logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# User-editable project paths
# -----------------------------------------------------------------------------
# Replace these defaults with your local confidential-data locations.
BASE_PROJECT_PATH = Path(r"/Users/dominikjurek/Library/CloudStorage/Dropbox/University/PhD Berkeley/Research")
VERSION = 2

# Derived output/cache locations
OUTPUT_PATH = BASE_PROJECT_PATH / f"Patents/Data/Inventor_Mobility__v{VERSION}"
RAW_DATA_PATH = BASE_PROJECT_PATH / "Patents/Data/Raw_Patentsview"
INTERMEDIATE_PATH = OUTPUT_PATH / "intermediate_files"
FINANCIAL_DATA_PATH = BASE_PROJECT_PATH / "WRDS Data"
MANDA_DATA_PATH = BASE_PROJECT_PATH / "SDC Data 1993 - 2018/MandA"

# Linktable path retained from the original script.
LINKTABLE_CSV = BASE_PROJECT_PATH / "Alice Project/Patent Portfolio and Economic Data/Patent Portfolio Source Data/linktable.csv"

# -----------------------------------------------------------------------------
# Convenience helper
# -----------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create output and intermediate directories if they do not already exist."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
