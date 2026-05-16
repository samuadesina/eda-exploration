# ================================================================
# src/anomaly_detector.py
# ================================================================
# CONTEXT:
#   The EDAEngine found patterns. Now we need to find EXCEPTIONS.
#   Which specific employees have salaries that are statistically unusual?
#   Which production runs had defect rates that no normal run should produce?
#
# THE BUSINESS QUESTION:
#   "We know the average salary is £92k. But are there any employees whose
#    salaries are so far from normal that we should investigate them?
#    Either they are being underpaid (retention risk) or the record is wrong."
#
# THE ANALOGY:
#   Imagine plotting all salaries on a number line.
#   Most cluster in the middle. A few sit far to the left or right.
#   The AnomalyDetector finds those outliers using statistics.
#   It uses TWO methods and only flags something as confirmed if BOTH agree.
#   This "consensus" approach reduces false alarms.
#
# TWO DETECTION METHODS:
#   Method 1 — IQR (Interquartile Range):
#     Works on any distribution. No assumptions about shape.
#     Uses Q1 - 1.5×IQR as the lower fence and Q3 + 1.5×IQR as the upper fence.
#
#   Method 2 — Z-score:
#     Assumes normally distributed data.
#     Flags values more than 3 standard deviations from the mean.
#
# WHY USE TWO METHODS?
#   Each method has blind spots. IQR is robust to skewed data but can
#   miss outliers that cluster near the fence. Z-score catches extreme
#   values precisely but is sensitive to the mean being pulled by outliers.
#   Consensus (flagged by BOTH) gives us the most reliable results.
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from scipy import stats   # scipy.stats: statistical functions including KS test

from config import INDUSTRY, REPORTS_DIR, logger


class AnomalyDetector:
    """
    Detects statistical anomalies using IQR and Z-score methods.

    Uses consensus: a row is a CONFIRMED anomaly only if flagged by BOTH methods.
    Consensus reduces false positives compared to using either method alone.

    Attributes
    ──────────
    df         pd.DataFrame   the input DataFrame to scan
    results    dict           anomaly findings per column
    confirmed  pd.DataFrame   rows confirmed as anomalies (flagged by both methods)
    _n_checked int            number of columns that were checked
    """

    # Z-score threshold: values more than this many standard deviations
    # from the mean are flagged as outliers.
    # 3.0 is the standard (covers 99.73% of normal distribution inside the fence)
    ZSCORE_THRESHOLD = 3.0

    def __init__(self, df: pd.DataFrame):
        """
        Initialise with the DataFrame to scan.

        Args:
            df   the processed DataFrame from EDAEngine
        """
        self.df        = df.copy()   # always work on a copy
        self.results   = {}          # per-column anomaly statistics
        self.confirmed = pd.DataFrame()   # rows flagged by both methods
        self._n_checked = 0

        logger.info(f"AnomalyDetector initialised — {len(df):,} rows to scan")

    # ── DETECTION METHODS ─────────────────────────────────────────────

    def _detect_iqr(self, col: str) -> pd.Series:
        """
        IQR method: flag values outside Tukey's fences.

        HOW THE IQR METHOD WORKS:
        ──────────────────────────
        Picture the data sorted from smallest to largest.
        Q1 = the value at position 25% (first quartile)
        Q3 = the value at position 75% (third quartile)
        IQR = Q3 - Q1  (the range of the middle 50% of the data)

        Tukey's fences (the standard outlier definition since 1977):
          Lower fence = Q1 - 1.5 × IQR
          Upper fence = Q3 + 1.5 × IQR

        Any value BELOW the lower fence OR ABOVE the upper fence is an outlier.

        WHY 1.5?
        ────────
        John Tukey chose 1.5 in his 1977 book "Exploratory Data Analysis".
        Under a normal distribution, 1.5 × IQR fences contain 99.3% of values.
        This has been the standard in box plots ever since.

        Args:
            col   the numeric column to check

        Returns:
            Boolean pd.Series — True where the value is an outlier
        """
        Q1 = self.df[col].quantile(0.25)   # 25th percentile
        Q3 = self.df[col].quantile(0.75)   # 75th percentile
        IQR = Q3 - Q1                      # interquartile range

        lower = Q1 - 1.5 * IQR   # Tukey lower fence
        upper = Q3 + 1.5 * IQR   # Tukey upper fence

        # (self.df[col] < lower) → True where value is below lower fence
        # (self.df[col] > upper) → True where value is above upper fence
        # The | operator is OR — True if EITHER condition is True
        return (self.df[col] < lower) | (self.df[col] > upper)

    def _detect_zscore(self, col: str) -> pd.Series:
        """
        Z-score method: flag values far from the mean.

        HOW THE Z-SCORE WORKS:
        ───────────────────────
        Z = (value - mean) / standard_deviation

        The Z-score tells us how many standard deviations a value is from the mean.
          Z = 0    → the value equals the mean (perfectly typical)
          Z = 1    → one standard deviation above the mean
          Z = -2   → two standard deviations below the mean
          Z = 3.5  → three and a half standard deviations above — likely an outlier

        For a normal distribution:
          68.3% of values fall within |Z| < 1
          95.4% of values fall within |Z| < 2
          99.7% of values fall within |Z| < 3
          Only 0.3% of values have |Z| > 3 — these are our outliers.

        LIMITATION:
        ───────────
        If the data is highly skewed (e.g. income data), the mean and std
        are themselves pulled by extreme values, making Z-scores unreliable.
        This is why we combine Z-score with IQR (which has no such assumption).

        Args:
            col   the numeric column to check

        Returns:
            Boolean pd.Series — True where |Z| > ZSCORE_THRESHOLD
        """
        mean = self.df[col].mean()   # arithmetic average
        std  = self.df[col].std()    # standard deviation

        # Guard: if std == 0, all values are identical — no outliers possible
        if std == 0:
            return pd.Series(False, index=self.df.index)

        # Compute Z-score for each value
        # (value - mean) / std → how many std deviations from the mean
        z_scores = (self.df[col] - mean).abs() / std   # .abs() for absolute value

        # Flag rows where the absolute Z-score exceeds our threshold
        return z_scores > self.ZSCORE_THRESHOLD

    def run(self, columns: list = None) -> "AnomalyDetector":
        """
        Run both detection methods on the specified columns and find consensus.

        CONSENSUS LOGIC:
        ─────────────────
        A row is flagged as a CONFIRMED anomaly only if it is flagged by
        BOTH the IQR method AND the Z-score method.

        Why consensus reduces false positives:
          IQR alone might flag a value that is just barely outside the fence
          Z-score alone might flag a value because the mean was itself skewed
          BOTH agreeing means the value is an outlier by two independent definitions

        Args:
            columns   list of numeric columns to analyse.
                      If not provided, analyses first 3 numeric columns.

        Returns self.
        """
        # Identify which columns to analyse
        num_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        target_cols = columns if columns else num_cols[:3]   # default: first 3

        # Track anomaly flags across all analysed columns
        all_flags = pd.DataFrame(index=self.df.index)

        for col in target_cols:
            if col not in self.df.columns:
                logger.warning(f"[ANOMALY] Column '{col}' not found — skipping")
                continue

            # Run both methods independently
            iqr_flags   = self._detect_iqr(col)     # IQR method → boolean Series
            z_flags     = self._detect_zscore(col)  # Z-score method → boolean Series

            # Consensus: True only where BOTH methods flag the row
            # The & operator is AND — True only if BOTH conditions are True
            consensus = iqr_flags & z_flags

            # Count flagged rows for reporting
            iqr_count  = int(iqr_flags.sum())
            z_count    = int(z_flags.sum())
            conf_count = int(consensus.sum())

            # Store per-column statistics
            self.results[col] = {
                "iqr_flagged":        iqr_count,
                "zscore_flagged":     z_count,
                "confirmed_anomalies": conf_count,
                "anomaly_pct":         round(conf_count / len(self.df) * 100, 2),
            }

            # Store flags for each method so we can identify confirmed rows
            all_flags[f"{col}_iqr"]       = iqr_flags
            all_flags[f"{col}_z"]         = z_flags
            all_flags[f"{col}_confirmed"] = consensus

            logger.info(
                f"[ANOMALY] {col}: "
                f"IQR={iqr_count} | Z-score={z_count} | "
                f"Confirmed={conf_count}"
            )

        # Find rows confirmed as anomalies in AT LEAST ONE column
        # .any(axis=1) checks across columns (axis=1 = row direction)
        # Returns True for rows where any confirmed flag is True
        confirmed_cols = [c for c in all_flags.columns if c.endswith("_confirmed")]
        if confirmed_cols:
            confirmed_mask = all_flags[confirmed_cols].any(axis=1)
            self.confirmed = self.df[confirmed_mask].copy()

        self._n_checked = len(target_cols)
        logger.info(
            f"[ANOMALY] Scan complete: "
            f"{len(self.confirmed):,} confirmed anomaly rows found"
        )

        return self

    def save_anomalies(self) -> "AnomalyDetector":
        """
        Save the confirmed anomaly rows to a CSV file.

        WHY SAVE ANOMALIES SEPARATELY?
        ────────────────────────────────
        Anomaly rows need to be reviewed by a human:
          - Is this a data entry error? → fix in the source system
          - Is this a legitimate extreme value? → keep, flag for modelling
          - Is this a test/dummy record? → remove from the dataset

        The CSV allows the business team to review without needing Python.
        It can be opened in Excel and investigated directly.

        Returns self.
        """
        if len(self.confirmed) == 0:
            logger.info("[ANOMALY] No anomalies to save")
            return self

        path = REPORTS_DIR / "anomalies.csv"
        self.confirmed.to_csv(path, index=False)
        logger.info(f"[ANOMALY] {len(self.confirmed):,} anomaly rows saved: {path}")

        return self

    def summary(self) -> pd.DataFrame:
        """
        Return anomaly findings as a formatted DataFrame.

        Useful for displaying results in a notebook or adding to a report.
        """
        if not self.results:
            return pd.DataFrame(columns=[
                "column", "iqr_flagged", "zscore_flagged",
                "confirmed_anomalies", "anomaly_pct"
            ])

        rows = []
        for col, result in self.results.items():
            rows.append({
                "column":               col,
                "iqr_flagged":          result["iqr_flagged"],
                "zscore_flagged":       result["zscore_flagged"],
                "confirmed_anomalies":  result["confirmed_anomalies"],
                "anomaly_pct":          result["anomaly_pct"],
            })

        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return (
            f"AnomalyDetector("
            f"{self._n_checked} columns checked | "
            f"{len(self.confirmed):,} confirmed anomalies)"
        )

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector("
            f"columns={self._n_checked}, "
            f"anomalies={len(self.confirmed)})"
        )
