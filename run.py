# ================================================================
# run.py — Module 06 Entry Point
# ================================================================
# This is the file you run to execute the full EDA pipeline.
#
# HOW TO RUN (from this project folder):
#   python run.py
#
# WHAT HAPPENS:
#   1. Load processed-data.csv (from Module 05)
#   2. Profile the dataset (shape, completeness, distributions)
#   3. Group analysis (metrics by category — e.g. salary by department)
#   4. Correlation (which numeric variables move together?)
#   5. Time trends (if a time column exists)
#   6. Print and save the analysis report
#   7. Run anomaly detection (IQR + Z-score consensus)
#   8. Save confirmed anomalies to reports/anomalies.csv
# ================================================================

import sys
import pathlib

# Add project root to Python path so imports work from any directory
_root = pathlib.Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import INDUSTRY, logger
from src.eda_engine       import EDAEngine
from src.anomaly_detector import AnomalyDetector


def main() -> None:
    """
    Run the complete EDA pipeline.

    Method chaining reads like a sentence:
    "Load the data, profile it, group it, correlate it,
     find trends, then generate the report."
    """
    logger.info("=" * 60)
    logger.info(f"  MODULE 06 — DATA ANALYSIS AND VISUALISATION")
    logger.info(f"  Industry: {INDUSTRY}")
    logger.info("=" * 60)

    # ── PART 1: EDA ENGINE ────────────────────────────────────────
    # Create the engine and run all analysis steps in a chain
    engine = EDAEngine()
    logger.info(f"Created: {engine}")

    (
        engine
        .load()           # load processed-data.csv, identify column types
        .profile()        # dataset shape, null rates, descriptive statistics
        .group_analysis() # mean/median/std per categorical group
        .correlation()    # Pearson correlation between numeric pairs
        .time_trends()    # month-over-month trends if time column exists
        .report(save=True)# print to terminal + save to reports/analysis_report.txt
    )

    logger.info(f"EDA complete: {engine}")

    # ── PART 2: ANOMALY DETECTION ─────────────────────────────────
    # Run separately so anomalies are found on the SAME data
    # but stored in a separate output file
    if engine.df is not None:
        logger.info("")
        logger.info("[ANOMALY] Starting anomaly detection...")

        detector = AnomalyDetector(engine.df)
        detector.run()     # IQR + Z-score consensus
        detector.save_anomalies()   # save to reports/anomalies.csv

        print()
        print("  ANOMALY DETECTION SUMMARY:")
        print("  " + "-" * 50)
        print(detector.summary().to_string(index=False))
        print()
        print(f"  Confirmed anomaly rows: {len(detector.confirmed):,}")
        print(f"  Saved to: reports/anomalies.csv")
        print()
        print(f"  NEXT: Open notebooks/01_eda_exploration.ipynb for")
        print(f"        interactive visualisation and deeper exploration")

        logger.info(f"Anomaly detection complete: {detector}")


# ── Entry point guard ─────────────────────────────────────────────
# Only runs when you execute: python run.py
# Does NOT run when another file imports from run.py
if __name__ == "__main__":
    main()
