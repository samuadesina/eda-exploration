# ================================================================
# config.py — Module 06 Configuration
# ================================================================
# WHAT THIS FILE DOES:
#   Provides a single, shared source of truth for:
#     - Which industry we are analysing (INDUSTRY)
#     - Where the input data lives (DATA_PATH)
#     - Where to save outputs (REPORTS_DIR)
#     - How to log messages (logger)
#
# ALL OTHER FILES import from here. Change a setting once → updated everywhere.
# ================================================================

import os          # os: access environment variables and file system
import pathlib     # pathlib: cross-platform file paths (works on Windows + Mac + Linux)
import logging     # logging: Python's built-in structured output system
from dotenv import load_dotenv   # reads .env file into environment variables

# Load variables from .env into os.environ
# The .env file stores secrets (passwords, API keys) — it is never committed to Git
load_dotenv()

# ── INDUSTRY SETTING ─────────────────────────────────────────────
# The teaching project always uses bootcamp_data (GlobalTech workforce data)
# Students change this to their industry schema in their projects
INDUSTRY = os.getenv("INDUSTRY", "retail")

# ── PROJECT PATHS ─────────────────────────────────────────────────
# pathlib.Path(__file__) → path to THIS file (config.py)
# .resolve() → convert to absolute path (no relative ".." parts)
# .parent → go up one folder (from config.py's folder to the project root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Data folder: where processed-data.csv lives
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)   # create if it does not exist

# Input: processed-data.csv from Module 05
# This is the CLEAN data — Module 05 already fixed nulls, types, and duplicates
DATA_PATH = DATA_DIR / "processed-data.csv"

# Output: where analysis reports and anomaly CSVs are saved
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)   # create if it does not exist

# ── ANALYSIS SETTINGS ─────────────────────────────────────────────
# How many top groups to show in the report (e.g. top 8 departments by salary)
TOP_N_GROUPS = 8

# Minimum correlation strength to include in the report
# Below 0.3 is considered "negligible" in business analytics
CORRELATION_THRESHOLD = 0.3

# ── LOGGER SETUP ──────────────────────────────────────────────────
# Logger explains what is happening step by step.
# Much better than print() because:
#   - Includes timestamp automatically
#   - Has severity levels: INFO, WARNING, ERROR
#   - Can be redirected to files without changing any code

def _setup_logger() -> logging.Logger:
    """
    Create and return the shared project logger.

    All modules import this logger:
        from config import logger
        logger.info("Starting analysis...")
    """
    lgr = logging.getLogger("module06")   # unique name prevents duplicate handlers
    lgr.setLevel(logging.INFO)            # show INFO and above (hide DEBUG)

    if not lgr.handlers:   # only add handler once (prevents duplicate log lines)
        handler = logging.StreamHandler()   # print to terminal
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        lgr.addHandler(handler)

    return lgr

# Create the shared logger — all files import this
logger = _setup_logger()
