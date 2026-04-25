# Calibrated Ensemble Fraud Detection — Cross-Dataset Study

This repository contains the complete ML pipeline for our research paper:

**"Calibrated Ensemble Learning for Fraud Detection: A Multi-Dataset Study with SHAP and LIME Under Original Imbalance"**

*Parash, Raihan, Rahman, Milton — American International University-Bangladesh (AIUB)*

## What This Project Does

We built a fraud detection system that:
- Works on **two different datasets** (European Credit Card + PaySim Mobile Money)
- Uses **ensemble learning** (Soft Voting, Stacking, Hybrid Stacking with Isolation Forest)
- Keeps the **original class imbalance** (no SMOTE or resampling tricks)
- Adds **probability calibration** and **threshold optimization** for real-world deployment
- Explains decisions using **SHAP** (global) and **LIME** (local)
- Validates results with **5 random seeds** and **bootstrap confidence intervals**

## Key Results

| Dataset | AUPRC (Mean ± Std) | Recall | Precision | F1 |
|---------|---------------------|--------|-----------|-----|
| European | 0.8387 ± 0.0153 | 0.8147 | 0.8883 | 0.8491 |
| PaySim | 0.9356 ± 0.0047 | 0.8914 | 0.8286 | 0.8585 |

## Datasets

- **European Credit Card**: [Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Download `creditcard.csv`
- **PaySim**: [Kaggle Link](https://www.kaggle.com/datasets/ealaxi/paysim1) — Download `PS_20174392719_1491204439457_log.csv`

> **Note:** Datasets are NOT included in this repo due to size. Download them from Kaggle.

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run European Pipeline

```bash
python ml.py
```

### 3. Run PaySim Pipeline

```bash
python run_paysim.py
```

### 4. Run Multi-Seed Validation

```bash
# For European dataset
python run_multiseed.py  # Set DATASET_MODE = "european" on line 40

# For PaySim dataset
python run_multiseed.py  # Set DATASET_MODE = "paysim" on line 40
```

## Project Structure

```
.
├── ml.py                  # European dataset full pipeline
├── run_paysim.py          # PaySim dataset full pipeline
├── run_multiseed.py       # 5-seed statistical validation
├── requirements.txt       # Python dependencies
├── figures/               # European dataset output figures
├── paysim_figures/        # PaySim dataset output figures
├── paysim_results/        # PaySim CSV result tables
└── stats_results/         # Multi-seed summary CSVs
```

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{parash2025fraud,
  title={Calibrated Ensemble Learning for Fraud Detection: A Multi-Dataset Study with SHAP and LIME Under Original Imbalance},
  author={Parash, Symon Islam and Raihan, Md. Jahir and Rahman, Md. Anjala and Milton, Md. Morshed},
  year={2025}
}
```

## License

This project is for academic/research purposes.
