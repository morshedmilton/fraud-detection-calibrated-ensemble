# -*- coding: utf-8 -*-
"""
Multi-Seed Statistical Validation Script
Run AFTER both European and PaySim pipelines are complete.

This script:
1. Runs the full pipeline 5 times with different seeds
2. Computes mean ± std for all metrics
3. Computes bootstrap 95% confidence intervals
4. Outputs publication-ready tables

Works for BOTH datasets — just change DATASET_MODE below.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score,
    f1_score, fbeta_score, matthews_corrcoef, roc_auc_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

# ============================================================
# CONFIGURATION — CHANGE THIS
# ============================================================
DATASET_MODE = "paysim"  # "european" or "paysim"
SEEDS = [42, 123, 456, 789, 1024]
N_SPLITS = 5

os.makedirs("stats_results", exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================
if DATASET_MODE == "european":
    df = pd.read_csv("creditcard.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    df["LogAmount"] = np.log1p(df["Amount"])
    X = df.drop(columns=["Class", "Amount"])
    y = df["Class"].astype(int)
    scale_cols = ["Time", "LogAmount"]
    other_cols = [c for c in X.columns if c not in scale_cols]
    print(f"European dataset: {len(y)} rows, {y.sum()} fraud")

elif DATASET_MODE == "paysim":
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["LogAmount"] = np.log1p(df["amount"])
    df["balanceDiffOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["isCashOut"] = (df["type"] == "CASH_OUT").astype(int)
    df["isTransfer"] = (df["type"] == "TRANSFER").astype(int)

    feature_cols = [
        "step", "LogAmount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "balanceDiffOrig",
        "balanceDiffDest", "isCashOut", "isTransfer"
    ]
    X = df[feature_cols].copy()
    y = df["isFraud"].astype(int)
    scale_cols = ["step", "LogAmount"]
    other_cols = [c for c in X.columns if c not in scale_cols]
    print(f"PaySim dataset: {len(y)} rows, {y.sum()} fraud")


# ============================================================
# HELPERS
# ============================================================
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("scale", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), scale_cols),
            ("keep", "passthrough", other_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def make_models(spw):
    return {
        "LR": LogisticRegression(
            class_weight="balanced", max_iter=1000,
            solver="liblinear", random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "XGB": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", scale_pos_weight=float(spw),
            random_state=42, n_jobs=-1, verbosity=0
        )
    }


def run_single_seed(seed):
    """Run full pipeline with one seed. Returns dict of metrics."""
    # Split
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_dev = X_dev.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_dev = y_dev.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    spw = (y_dev == 0).sum() / max((y_dev == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pp = make_preprocessor()
    mdls = make_models(spw)

    # OOF baselines
    oof_probs = {}
    fitted = {}
    for name, model in mdls.items():
        oof = np.zeros(len(y_dev), dtype=float)
        for _, (tr, va) in enumerate(cv.split(X_dev, y_dev)):
            pipe = Pipeline([("prep", clone(pp)), ("model", clone(model))])
            pipe.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            oof[va] = pipe.predict_proba(X_dev.iloc[va])[:, 1]
        oof_probs[name] = oof
        fpipe = Pipeline([("prep", clone(pp)), ("model", clone(model))])
        fpipe.fit(X_dev, y_dev)
        fitted[name] = fpipe

    # Soft voting
    sv_dev = np.mean([oof_probs[n] for n in mdls], axis=0)
    sv_test = np.mean([
        fitted[n].predict_proba(X_test)[:, 1] for n in mdls
    ], axis=0)

    # Isotonic calibration (crossfitted on dev)
    calib_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cal_dev = np.zeros(len(y_dev), dtype=float)
    for _, (tr, va) in enumerate(calib_cv.split(sv_dev.reshape(-1, 1), y_dev)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(sv_dev[tr], y_dev.iloc[tr])
        cal_dev[va] = iso.predict(sv_dev[va])

    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_full.fit(sv_dev, y_dev)
    cal_test = iso_full.predict(sv_test)

    cal_dev = np.clip(cal_dev, 1e-6, 1 - 1e-6)
    cal_test = np.clip(cal_test, 1e-6, 1 - 1e-6)

    # F2-optimal threshold on dev
    best_thr, best_f2 = 0.5, 0
    for thr in np.linspace(0.001, 0.999, 999):
        yp = (cal_dev >= thr).astype(int)
        f2 = fbeta_score(y_dev, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr

    # Test metrics at optimized threshold
    y_pred = (cal_test >= best_thr).astype(int)
    results = {
        "seed": seed,
        "threshold": best_thr,
        "AUPRC": average_precision_score(y_test, cal_test),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "F2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, cal_test),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Brier": brier_score_loss(y_test, cal_test),
        "Alerts": int(y_pred.sum()),
    }
    return results, y_test.values, cal_test


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000):
    """Compute 95% bootstrap CI for a metric."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ============================================================
# RUN ALL SEEDS
# ============================================================
print(f"\nRunning {len(SEEDS)} seeds for {DATASET_MODE}...\n")
all_results = []
last_y_test = None
last_cal_test = None

for seed in SEEDS:
    print(f"  Seed {seed}...", end=" ")
    res, yt, ct = run_single_seed(seed)
    all_results.append(res)
    last_y_test = yt
    last_cal_test = ct
    print(f"AUPRC={res['AUPRC']:.4f}, Recall={res['Recall']:.4f}, "
          f"F1={res['F1']:.4f}, MCC={res['MCC']:.4f}")

results_df = pd.DataFrame(all_results)

# ============================================================
# MEAN ± STD TABLE
# ============================================================
metric_cols = ["AUPRC", "Recall", "Precision", "F1", "F2", "ROC-AUC", "MCC", "Brier"]
summary_rows = []
for col in metric_cols:
    vals = results_df[col].values
    summary_rows.append({
        "Metric": col,
        "Mean": np.mean(vals),
        "Std": np.std(vals),
        "Min": np.min(vals),
        "Max": np.max(vals),
        "Formatted": f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"stats_results/{DATASET_MODE}_multiseed_summary.csv", index=False)

print(f"\n{'=' * 50}")
print(f"MULTI-SEED RESULTS ({DATASET_MODE.upper()})")
print(f"{'=' * 50}")
print(f"Seeds: {SEEDS}")
print(f"\n{'Metric':<12} {'Mean ± Std':<20} {'Range'}")
print("-" * 50)
for _, row in summary_df.iterrows():
    print(f"{row['Metric']:<12} {row['Formatted']:<20} [{row['Min']:.4f}, {row['Max']:.4f}]")

# ============================================================
# BOOTSTRAP 95% CI (on last seed's test set)
# ============================================================
print(f"\n{'=' * 50}")
print(f"BOOTSTRAP 95% CI (seed={SEEDS[-1]})")
print(f"{'=' * 50}")

ci_rows = []
for metric_name, metric_fn in [
    ("AUPRC", average_precision_score),
    ("ROC-AUC", roc_auc_score),
]:
    lo, hi = bootstrap_ci(last_y_test, last_cal_test, metric_fn)
    mean_val = metric_fn(last_y_test, last_cal_test)
    ci_rows.append({
        "Metric": metric_name,
        "Mean": mean_val,
        "CI_Lower": lo,
        "CI_Upper": hi,
        "Formatted": f"{mean_val:.4f} [{lo:.4f}, {hi:.4f}]"
    })
    print(f"  {metric_name}: {mean_val:.4f} [{lo:.4f}, {hi:.4f}]")

# For threshold-dependent metrics, use the optimized threshold
best_thr = results_df[results_df["seed"] == SEEDS[-1]]["threshold"].values[0]
last_pred = (last_cal_test >= best_thr).astype(int)

for metric_name, metric_fn in [
    ("Recall", recall_score),
    ("Precision", precision_score),
    ("F1", f1_score),
    ("MCC", matthews_corrcoef),
]:
    def wrapped_fn(yt, ys, fn=metric_fn, thr=best_thr):
        yp = (ys >= thr).astype(int)
        return fn(yt, yp, **({'zero_division': 0} if fn != matthews_corrcoef else {}))

    lo, hi = bootstrap_ci(last_y_test, last_cal_test, wrapped_fn)
    mean_val = wrapped_fn(last_y_test, last_cal_test)
    ci_rows.append({
        "Metric": metric_name,
        "Mean": mean_val,
        "CI_Lower": lo,
        "CI_Upper": hi,
        "Formatted": f"{mean_val:.4f} [{lo:.4f}, {hi:.4f}]"
    })
    print(f"  {metric_name}: {mean_val:.4f} [{lo:.4f}, {hi:.4f}]")

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(f"stats_results/{DATASET_MODE}_bootstrap_ci.csv", index=False)

# Save raw per-seed results
results_df.to_csv(f"stats_results/{DATASET_MODE}_per_seed_results.csv", index=False)

print(f"\nAll results saved to stats_results/")
print("Done.")
