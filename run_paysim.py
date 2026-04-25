# -*- coding: utf-8 -*-
"""
PaySim Pipeline — Mirrors the European dataset pipeline from ml.py
Run this in Google Colab after uploading the PaySim CSV.

Produces:
  - paysim_results/  (all CSV tables)
  - paysim_figures/  (all plots)

This uses the EXACT same model configs, CV strategy, calibration,
and threshold optimization as the European pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, brier_score_loss, log_loss,
)

from xgboost import XGBClassifier

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD_SCAN = np.linspace(0.001, 0.999, 999)

os.makedirs("paysim_results", exist_ok=True)
os.makedirs("paysim_figures", exist_ok=True)

np.random.seed(RANDOM_STATE)

# ============================================================
# STEP 1: LOAD AND PREPROCESS PAYSIM
# ============================================================
print("=" * 60)
print("STEP 1: Loading and preprocessing PaySim dataset")
print("=" * 60)

# -- Upload in Colab --
try:
    from google.colab import files as colab_files
    uploaded = colab_files.upload()
    data_file = next(iter(uploaded))
    df = pd.read_csv(data_file)
except ImportError:
    # Running locally — adjust path as needed
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution (raw):")
print(df["isFraud"].value_counts())

# Filter to fraud-relevant transaction types only
# (fraud ONLY occurs in CASH_OUT and TRANSFER)
df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
print(f"\nAfter filtering to CASH_OUT/TRANSFER: {df.shape}")

# Remove exact duplicates
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
rows_after = len(df)
print(f"Removed {rows_before - rows_after} duplicates")

# Feature engineering
df["LogAmount"] = np.log1p(df["amount"])
df["balanceDiffOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["isCashOut"] = (df["type"] == "CASH_OUT").astype(int)
df["isTransfer"] = (df["type"] == "TRANSFER").astype(int)

# Define features — these are the PaySim equivalents
feature_cols = [
    "step",          # time proxy (analogous to "Time" in European)
    "LogAmount",     # log-transformed amount (same as European)
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balanceDiffOrig", "balanceDiffDest",
    "isCashOut", "isTransfer"
]

X = df[feature_cols].copy()
y = df["isFraud"].astype(int).copy()

print(f"\nFinal dataset: {len(y)} transactions, {y.sum()} fraud ({y.mean():.4%})")
print(f"Feature matrix shape: {X.shape}")

# Columns to scale (same logic as European pipeline)
scale_cols = ["step", "LogAmount"]
other_cols = [c for c in X.columns if c not in scale_cols]

# Train/test split (same 80:20 stratified as European)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_dev = X_dev.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_dev = y_dev.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"\nDev set: {len(y_dev)} ({y_dev.sum()} fraud)")
print(f"Test set: {len(y_test)} ({y_test.sum()} fraud)")

# Compute class weight for XGBoost
neg = (y_dev == 0).sum()
pos = (y_dev == 1).sum()
scale_pos_weight = neg / pos
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================
# STEP 2: LEAKAGE-SAFE BASELINES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Running baseline models")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "scale_selected",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            scale_cols
        ),
        (
            "keep_rest",
            "passthrough",
            other_cols
        )
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

models = {
    "Weighted Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000,
        solver="liblinear", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(model):
    return Pipeline([
        ("prep", clone(preprocessor)),
        ("model", clone(model))
    ])


def calc_metrics(y_true, y_prob, threshold=0.50):
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUPRC": average_precision_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred


# OOF cross-validation
oof_probabilities = {}
fitted_baseline_models = {}
cv_rows = []

for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    oof_prob = np.zeros(len(y_dev), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_dev, y_dev), 1):
        pipe = build_pipeline(model)
        pipe.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        oof_prob[valid_idx] = pipe.predict_proba(X_dev.iloc[valid_idx])[:, 1]
        print(f"  Fold {fold}/{N_SPLITS} done")

    oof_probabilities[model_name] = oof_prob
    metrics, _ = calc_metrics(y_dev, oof_prob)
    metrics["Model"] = model_name
    cv_rows.append(metrics)

    # Fit on full dev for test snapshot
    final_pipe = build_pipeline(model)
    final_pipe.fit(X_dev, y_dev)
    fitted_baseline_models[model_name] = final_pipe

baseline_dev_df = pd.DataFrame(cv_rows)[
    ["Model", "AUPRC", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]
].sort_values("AUPRC", ascending=False).reset_index(drop=True)

# Test snapshot
test_rows = []
for mn, fm in fitted_baseline_models.items():
    tp = fm.predict_proba(X_test)[:, 1]
    m, _ = calc_metrics(y_test, tp)
    m["Model"] = mn
    test_rows.append(m)

baseline_test_df = pd.DataFrame(test_rows)[
    ["Model", "AUPRC", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]
].sort_values("AUPRC", ascending=False).reset_index(drop=True)

baseline_dev_df.to_csv("paysim_results/step2_baseline_dev.csv", index=False)
baseline_test_df.to_csv("paysim_results/step2_baseline_test.csv", index=False)

print("\nPaySim Baseline CV Results:")
print(baseline_dev_df.round(4).to_string(index=False))
print("\nPaySim Baseline Test Snapshot:")
print(baseline_test_df.round(4).to_string(index=False))

# ============================================================
# STEP 3: ENSEMBLES (Soft Voting, Stacking, Hybrid)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Ensemble construction")
print("=" * 60)

# Meta-features from supervised baselines
meta_dev = pd.DataFrame({
    "lr_prob": oof_probabilities["Weighted Logistic Regression"],
    "rf_prob": oof_probabilities["Random Forest"],
    "xgb_prob": oof_probabilities["XGBoost"]
})

meta_test = pd.DataFrame({
    "lr_prob": fitted_baseline_models["Weighted Logistic Regression"].predict_proba(X_test)[:, 1],
    "rf_prob": fitted_baseline_models["Random Forest"].predict_proba(X_test)[:, 1],
    "xgb_prob": fitted_baseline_models["XGBoost"].predict_proba(X_test)[:, 1]
})

# Isolation Forest OOF
print("\nBuilding Isolation Forest anomaly branch...")
iforest_oof = np.zeros(len(y_dev), dtype=float)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_dev, y_dev), 1):
    contamination = float(y_dev.iloc[tr_idx].mean())
    iforest_pipe = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", IsolationForest(n_estimators=300,
                                   contamination=max(contamination, 1e-4),
                                   random_state=RANDOM_STATE, n_jobs=-1))
    ])
    iforest_pipe.fit(X_dev.iloc[tr_idx])
    iforest_oof[va_idx] = -iforest_pipe.decision_function(X_dev.iloc[va_idx])
    print(f"  IF fold {fold}/{N_SPLITS} done")

# Full IF for test
final_if = Pipeline([
    ("prep", clone(preprocessor)),
    ("model", IsolationForest(n_estimators=300,
                               contamination=max(float(y_dev.mean()), 1e-4),
                               random_state=RANDOM_STATE, n_jobs=-1))
])
final_if.fit(X_dev)
iforest_test = -final_if.decision_function(X_test)

# Hybrid meta-features
meta_dev_hybrid = meta_dev.copy()
meta_dev_hybrid["iforest_score"] = iforest_oof
meta_test_hybrid = meta_test.copy()
meta_test_hybrid["iforest_score"] = iforest_test

# Soft voting
sv_dev = meta_dev.mean(axis=1).values
sv_test = meta_test.mean(axis=1).values


def build_meta_learner():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                      solver="liblinear", random_state=RANDOM_STATE))
    ])


def oof_stacking(X_meta, y_true, label):
    oof = np.zeros(len(y_true), dtype=float)
    for fold, (tr, va) in enumerate(cv.split(X_meta, y_true), 1):
        ml = build_meta_learner()
        ml.fit(X_meta.iloc[tr], y_true.iloc[tr])
        oof[va] = ml.predict_proba(X_meta.iloc[va])[:, 1]
        print(f"  {label} fold {fold}/{N_SPLITS}")
    final_ml = build_meta_learner()
    final_ml.fit(X_meta, y_true)
    return oof, final_ml


print("\nSupervised stacking...")
ss_dev, ss_final = oof_stacking(meta_dev, y_dev, "Sup Stack")
ss_test = ss_final.predict_proba(meta_test)[:, 1]

print("\nHybrid stacking...")
hs_dev, hs_final = oof_stacking(meta_dev_hybrid, y_dev, "Hyb Stack")
hs_test = hs_final.predict_proba(meta_test_hybrid)[:, 1]

# Compare ensembles
ensemble_candidates = {
    "Soft Voting (Supervised)": {"dev": sv_dev, "test": sv_test},
    "Supervised Stacking": {"dev": ss_dev, "test": ss_test},
    "Hybrid Stacking + IsolationForest": {"dev": hs_dev, "test": hs_test},
}

ens_dev_rows, ens_test_rows = [], []
for name, probs in ensemble_candidates.items():
    dm, _ = calc_metrics(y_dev, probs["dev"])
    dm["Model"] = name
    ens_dev_rows.append(dm)
    tm, _ = calc_metrics(y_test, probs["test"])
    tm["Model"] = name
    ens_test_rows.append(tm)

ens_dev_df = pd.DataFrame(ens_dev_rows)[
    ["Model", "AUPRC", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]
].sort_values("AUPRC", ascending=False).reset_index(drop=True)

ens_test_df = pd.DataFrame(ens_test_rows)[
    ["Model", "AUPRC", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]
].sort_values("AUPRC", ascending=False).reset_index(drop=True)

ens_dev_df.to_csv("paysim_results/step3_ensemble_dev.csv", index=False)
ens_test_df.to_csv("paysim_results/step3_ensemble_test.csv", index=False)

print("\nPaySim Ensemble Dev Results:")
print(ens_dev_df.round(4).to_string(index=False))

# PR curve plot
plt.figure(figsize=(8, 6))
for name, probs in ensemble_candidates.items():
    prec, rec, _ = precision_recall_curve(y_dev, probs["dev"])
    auprc = average_precision_score(y_dev, probs["dev"])
    plt.plot(rec, prec, label=f"{name} (AUPRC={auprc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PaySim — PR Curves (OOF Development)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("paysim_figures/step3_pr_curves.png", dpi=300, bbox_inches="tight")
plt.show()

# Select best by dev AUPRC
best_ensemble = ens_dev_df.iloc[0]["Model"]
print(f"\nBest ensemble by dev AUPRC: {best_ensemble}")

# ============================================================
# STEP 4: CALIBRATION + THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Calibration + threshold optimization")
print("=" * 60)

selected_dev = ensemble_candidates[best_ensemble]["dev"]
selected_test = ensemble_candidates[best_ensemble]["test"]


def fit_platt(p_train, y_train):
    m = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    m.fit(np.asarray(p_train).reshape(-1, 1), np.asarray(y_train))
    return m


def predict_platt(m, p):
    return m.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def fit_isotonic(p_train, y_train):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    m.fit(np.asarray(p_train), np.asarray(y_train))
    return m


def crossfit_calibrate(dev_p, test_p, y_true, method):
    dev_p = np.clip(dev_p, 1e-6, 1 - 1e-6)
    test_p = np.clip(test_p, 1e-6, 1 - 1e-6)
    if method == "Raw":
        return dev_p, test_p

    calib_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_cal = np.zeros(len(y_true), dtype=float)
    for _, (tr, va) in enumerate(calib_cv.split(dev_p.reshape(-1, 1), y_true)):
        if method == "Platt":
            fm = fit_platt(dev_p[tr], y_true.iloc[tr])
            oof_cal[va] = predict_platt(fm, dev_p[va])
        elif method == "Isotonic":
            fm = fit_isotonic(dev_p[tr], y_true.iloc[tr])
            oof_cal[va] = fm.predict(dev_p[va])

    if method == "Platt":
        full_m = fit_platt(dev_p, y_true)
        test_cal = predict_platt(full_m, test_p)
    else:
        full_m = fit_isotonic(dev_p, y_true)
        test_cal = full_m.predict(test_p)

    return np.clip(oof_cal, 1e-6, 1 - 1e-6), np.clip(test_cal, 1e-6, 1 - 1e-6)


# Compare calibration methods
calib_rows = []
calib_outputs = {}
for method in ["Raw", "Platt", "Isotonic"]:
    d, t = crossfit_calibrate(selected_dev, selected_test, y_dev, method)
    calib_outputs[method] = {"dev": d, "test": t}
    calib_rows.append({
        "Calibration": method,
        "Brier": brier_score_loss(y_dev, d),
        "LogLoss": log_loss(y_dev, d, labels=[0, 1]),
        "AUPRC": average_precision_score(y_dev, d),
        "ROC-AUC": roc_auc_score(y_dev, d),
    })

calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv("paysim_results/step4_calibration_comparison.csv", index=False)
print("\nCalibration comparison:")
print(calib_df.round(6).to_string(index=False))

# Choose best calibration (lowest Brier, excluding Raw)
selectable = calib_df[calib_df["Calibration"] != "Raw"].sort_values("Brier")
chosen_calib = selectable.iloc[0]["Calibration"]
print(f"\nChosen calibration: {chosen_calib}")

final_dev_prob = calib_outputs[chosen_calib]["dev"]
final_test_prob = calib_outputs[chosen_calib]["test"]

# Threshold optimization
best_threshold = 0.50
best_f2 = 0
scan_rows = []
for thr in THRESHOLD_SCAN:
    yp = (final_dev_prob >= thr).astype(int)
    f2 = fbeta_score(y_dev, yp, beta=2, zero_division=0)
    scan_rows.append({
        "Threshold": thr,
        "Recall": recall_score(y_dev, yp, zero_division=0),
        "Precision": precision_score(y_dev, yp, zero_division=0),
        "F2": f2
    })
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = thr

scan_df = pd.DataFrame(scan_rows)
print(f"Optimized threshold: {best_threshold:.3f}")

# Threshold scan plot
plt.figure(figsize=(8, 6))
plt.plot(scan_df["Threshold"], scan_df["Recall"], label="Recall")
plt.plot(scan_df["Threshold"], scan_df["Precision"], label="Precision")
plt.plot(scan_df["Threshold"], scan_df["F2"], label="F2")
plt.axvline(best_threshold, linestyle="--", label=f"Best = {best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title(f"PaySim — Threshold Scan ({chosen_calib})")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("paysim_figures/step4_threshold_scan.png", dpi=300, bbox_inches="tight")
plt.show()

# Calibration plot
frac_raw, mean_raw = calibration_curve(y_dev, calib_outputs["Raw"]["dev"], n_bins=10, strategy="quantile")
frac_cal, mean_cal = calibration_curve(y_dev, final_dev_prob, n_bins=10, strategy="quantile")
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "--", label="Perfect")
plt.plot(mean_raw, frac_raw, "o-", label="Raw")
plt.plot(mean_cal, frac_cal, "o-", label=chosen_calib)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraud rate")
plt.title(f"PaySim — Calibration Plot ({best_ensemble})")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("paysim_figures/step4_calibration_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Final test results — ablation table
ablation_rows = []
for label, prob_d, prob_t, thr in [
    ("Raw @ 0.50", selected_dev, selected_test, 0.50),
    (f"{chosen_calib} @ 0.50", final_dev_prob, final_test_prob, 0.50),
    (f"{chosen_calib} @ optimized", final_dev_prob, final_test_prob, best_threshold),
]:
    m, yp = calc_metrics(y_test, prob_t, thr)
    m["Setting"] = label
    m["Threshold"] = thr
    m["Alerts"] = int(yp.sum())
    ablation_rows.append(m)

ablation_df = pd.DataFrame(ablation_rows)[
    ["Setting", "Threshold", "AUPRC", "Recall", "Precision", "F1", "F2",
     "ROC-AUC", "MCC", "Alerts"]
]
ablation_df.to_csv("paysim_results/step4_ablation_test.csv", index=False)
print("\nPaySim Final Test Ablation:")
print(ablation_df.round(4).to_string(index=False))

# Confusion matrix
final_pred = (final_test_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"PaySim — {best_ensemble}\n({chosen_calib}, threshold={best_threshold:.3f})")
plt.savefig("paysim_figures/step4_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# STEP 5: SHAP ON XGBOOST (PaySim — interpretable features)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: SHAP explainability (PaySim)")
print("=" * 60)

try:
    import shap

    # Get the XGBoost model and transform test data
    xgb_pipe = fitted_baseline_models["XGBoost"]
    X_test_transformed = xgb_pipe.named_steps["prep"].transform(X_test)

    # Feature names after preprocessing
    feat_names = scale_cols + other_cols

    xgb_model = xgb_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test_transformed,
        feature_names=feat_names, show=False
    )
    plt.title("PaySim — SHAP Summary (XGBoost)")
    plt.tight_layout()
    plt.savefig("paysim_figures/step5_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("SHAP summary saved.")
except Exception as e:
    print(f"SHAP failed (non-critical): {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PAYSIM PIPELINE COMPLETE")
print("=" * 60)
print(f"\nDataset: PaySim ({len(y)} transactions, {y.sum()} fraud, {y.mean():.4%})")
print(f"Best ensemble: {best_ensemble}")
print(f"Calibration: {chosen_calib}")
print(f"Threshold: {best_threshold:.3f}")
print(f"\nAll results in: paysim_results/")
print(f"All figures in: paysim_figures/")
