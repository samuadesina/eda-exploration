# 📊 Retail Exploratory Data Analysis
### Samuel Adesina

---

## Overview

This is the analysis phase of the ShopSmart Retail Group data project.  
The raw data was extracted from a live Supabase PostgreSQL database and transformed through a custom ETL pipeline in **[retail-eda](https://github.com/samueladesina/retail-eda)**. This module picks up from there — loading `processed-data.csv` and answering three urgent business questions for the Head of Merchandising ahead of the quarterly board meeting.

**Company:** ShopSmart Retail Group  
**Role:** Junior Data Analyst  
**Input:** `processed-data.csv` — 50,243 cleaned retail transactions  
**Output:** `reports/analysis_report.txt` + `reports/anomalies.csv`  

---

## Business Questions

### Question 1 — Revenue Performance
Which product categories and stores generate the highest revenue per sale?  
Which stores are underperforming relative to their category average?

> Answered using `groupby` rankings and `vs_category_avg` comparisons.  
> A negative `vs_category_avg` value flags an underperforming store.

---

### Question 2 — Discount Impact
Does offering higher discounts actually drive higher sales volume?  
Compute the **Pearson correlation** between `discount_pct` and `total_amount`.

> A negative correlation means higher discounts produce lower revenue totals —  
> the company is sacrificing margin without increasing sales.

---

### Question 3 — Anomaly Detection
Which transactions are statistically unusual?  
Use **IQR + Z-score consensus** to identify outliers and save them to `reports/anomalies.csv`.

> A transaction is flagged only when both methods agree — reducing false positives.

---

## Project Structure

```
p01-eda-exploration/
│
├── config.py                        # Paths, logger, thresholds
│
├── src/
│   ├── eda_engine.py                # EDAEngine: load, profile, group_analysis, correlation, report
│   └── anomaly_detector.py         # AnomalyDetector: IQR + Z-score consensus
│
├── data/
│   └── processed-data.csv          # Output from ETL pipeline (gitignored)
│
├── reports/
│   ├── analysis_report.txt         # Revenue rankings + correlation results
│   └── anomalies.csv               # Flagged unusual transactions
│
├── notebooks/
│   └── eda_exploration.ipynb       # Full analysis walkthrough
│
├── tests/
│   └── test_eda.py                 # 8+ unit tests
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Key Classes

### `EDAEngine`
| Method | What it does |
|---|---|
| `load()` | Reads `processed-data.csv` into a DataFrame |
| `profile()` | Summary stats — shape, nulls, dtypes, value counts |
| `group_analysis()` | Revenue grouped by category and store, ranked |
| `correlation()` | Pearson correlation between `discount_pct` and `total_amount` |
| `report()` | Writes full findings to `reports/analysis_report.txt` |

### `AnomalyDetector`
| Method | What it does |
|---|---|
| `iqr_flags()` | Flags transactions outside Q1 − 1.5×IQR / Q3 + 1.5×IQR |
| `zscore_flags()` | Flags transactions with \|Z-score\| > 3 |
| `consensus()` | Returns only rows flagged by **both** methods |
| `save()` | Writes flagged rows to `reports/anomalies.csv` |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/samueladesina/p01-eda-exploration.git
cd p01-eda-exploration
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add processed data
Copy `processed-data.csv` from your ETL pipeline output into `data/`:
```bash
cp ../retail-eda/data/processed/processed-data.csv data/processed-data.csv
```

### 5. Run the analysis
```bash
python run.py
```

---

## Success Criteria

- [x] `processed-data.csv` loaded from ETL pipeline output
- [ ] `EDAEngine` with `load`, `profile`, `group_analysis`, `correlation`, `report`
- [ ] `AnomalyDetector` using IQR + Z-score consensus
- [ ] `analysis_report.txt` shows revenue ranked by category and store
- [ ] `anomalies.csv` contains flagged transactions
- [ ] 8+ unit tests passing
- [ ] Project pushed to GitHub

---

## Dependencies

```
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.0
seaborn==0.13.2
python-dotenv==1.0.1
pytest==8.0.0
```

---

## Related Projects

| Project | Description |
|---|---|
| [p01-retail-eda](https://github.com/samueladesina/retail-eda) | ETL pipeline — extracts and transforms raw Supabase data |
| p01-eda-exploration ← you are here | EDA — answers business questions from processed data |

---

## Author

**Samuel Adesina**  
Data Analyst | Python · SQL · Data Engineering  
[GitHub](https://github.com/samueladesina) · [LinkedIn](https://github.com/samuadesina)

---
