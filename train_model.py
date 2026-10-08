"""
Fair Loan Approval Predictor — Model Training
SDG 10: Reduced Inequalities
Trains a Random Forest classifier on the IBM Home Loan dataset,
runs a fairness audit, and saves the model + metadata to disk.
"""

import csv
import json
import pickle
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────────────────────

DATASET = "realistic_loan_dataset.csv"

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area"]
NUMERIC_COLS     = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History"]
TARGET           = "Loan_Status"

# Fill-in defaults for missing values (mode/median of dataset)
DEFAULTS = {
    "Gender":            "Male",
    "Married":           "Yes",
    "Dependents":        "0",
    "Education":         "Graduate",
    "Self_Employed":     "No",
    "ApplicantIncome":   5000,
    "CoapplicantIncome": 0,
    "LoanAmount":        150,
    "Loan_Amount_Term":  360,
    "Credit_History":    1.0,
    "Property_Area":     "Semiurban",
}

# Categorical encoders saved for inference
LABEL_ENCODERS: dict[str, LabelEncoder] = {}


def load_data(path: str):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return rows


def preprocess(rows: list[dict], fit: bool = True):
    X_cat, X_num, y, sensitive = [], [], [], []

    for r in rows:
        # --- numeric ---
        try:
            ai  = float(r["ApplicantIncome"])   if r["ApplicantIncome"]   else DEFAULTS["ApplicantIncome"]
            ca  = float(r["CoapplicantIncome"])  if r["CoapplicantIncome"] else DEFAULTS["CoapplicantIncome"]
            la  = float(r["LoanAmount"])         if r["LoanAmount"]        else DEFAULTS["LoanAmount"]
            lt  = float(r["Loan_Amount_Term"])   if r["Loan_Amount_Term"]  else DEFAULTS["Loan_Amount_Term"]
            ch  = float(r["Credit_History"])     if r["Credit_History"]    else DEFAULTS["Credit_History"]
        except ValueError:
            continue

        # derived features
        total_income   = ai + ca
        loan_to_income = la / (total_income / 12) if total_income > 0 else 0
        emi            = la / lt if lt > 0 else 0

        num_row = [ai, ca, la, lt, ch, total_income, loan_to_income, emi]

        # --- categorical ---
        gender   = r["Gender"]     or DEFAULTS["Gender"]
        married  = r["Married"]    or DEFAULTS["Married"]
        deps     = r["Dependents"] or DEFAULTS["Dependents"]
        edu      = r["Education"]  or DEFAULTS["Education"]
        self_emp = r["Self_Employed"] or DEFAULTS["Self_Employed"]
        area     = r["Property_Area"] or DEFAULTS["Property_Area"]

        cat_row = [gender, married, deps, edu, self_emp, area]

        label = r.get(TARGET, "")
        if label not in ("Y", "N"):
            continue

        X_cat.append(cat_row)
        X_num.append(num_row)
        y.append(1 if label == "Y" else 0)
        sensitive.append(gender)  # for fairness audit

    # Encode categoricals
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    X_cat_enc = []
    for i, col in enumerate(cat_cols):
        col_vals = [row[i] for row in X_cat]
        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(col_vals)
            LABEL_ENCODERS[col] = le
        else:
            le = LABEL_ENCODERS[col]
            encoded = le.transform(col_vals)
        X_cat_enc.append(encoded)

    X_cat_arr = np.array(X_cat_enc).T
    X_num_arr = np.array(X_num, dtype=float)
    X = np.hstack([X_cat_arr, X_num_arr])
    y = np.array(y)
    return X, y, sensitive


# ─────────────────────────────────────────────────────────────
# 2. Train
# ─────────────────────────────────────────────────────────────

def train():
    print("\n" + "="*55)
    print("  FAIR LOAN PREDICTOR — MODEL TRAINING")
    print("="*55)

    rows = load_data(DATASET)
    X, y, sensitive = preprocess(rows, fit=True)

    print(f"\n  Dataset: {len(y)} samples  |  Approved: {y.sum()}  |  Rejected: {(y==0).sum()}")

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric columns
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Three models compared
    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    }

    best_name, best_model, best_score = None, None, 0
    results = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc   = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
        except Exception:
            auc = 0.0
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
        results[name] = {"accuracy": round(acc, 4), "auc": round(auc, 4), "cv_mean": round(cv_scores.mean(), 4)}
        print(f"\n  [{name}]")
        print(f"    Accuracy : {acc:.4f}  |  AUC: {auc:.4f}  |  CV: {cv_scores.mean():.4f}")
        if auc > best_score:
            best_score = auc
            best_name  = name
            best_model = model

    print(f"\n  [OK] Best model: {best_name} (AUC={best_score:.4f})")

    # ── Feature importance (RF only) ──
    feature_names = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "TotalIncome", "LoanToIncome", "EMI"
    ]
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(feature_names, best_model.feature_importances_.tolist()))
        importances = dict(sorted(fi.items(), key=lambda x: -x[1]))

    # ─────────────────────────────────────────────────────────
    # 3. Fairness Audit
    # ─────────────────────────────────────────────────────────
    print("\n  — FAIRNESS AUDIT (Gender) —")
    preds_all = best_model.predict(X_test_s)
    fairness = defaultdict(lambda: {"approved": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})

    for pred, truth, gender in zip(preds_all, y_test, s_test):
        fairness[gender]["total"] += 1
        if pred == 1:
            fairness[gender]["approved"] += 1
        if pred == 1 and truth == 1:
            fairness[gender]["tp"] += 1
        if pred == 1 and truth == 0:
            fairness[gender]["fp"] += 1
        if pred == 0 and truth == 0:
            fairness[gender]["tn"] += 1
        if pred == 0 and truth == 1:
            fairness[gender]["fn"] += 1

    fairness_report = {}
    for grp, stats in fairness.items():
        t = stats["total"]
        fairness_report[grp] = {
            "approval_rate":        round(stats["approved"] / t * 100, 1) if t else 0,
            "true_positive_rate":   round(stats["tp"] / (stats["tp"] + stats["fn"]) * 100, 1) if (stats["tp"] + stats["fn"]) > 0 else 0,
            "false_positive_rate":  round(stats["fp"] / (stats["fp"] + stats["tn"]) * 100, 1) if (stats["fp"] + stats["tn"]) > 0 else 0,
            "total":                t,
        }
        print(f"    {grp:<8}: Approval={fairness_report[grp]['approval_rate']}%  TPR={fairness_report[grp]['true_positive_rate']}%  FPR={fairness_report[grp]['false_positive_rate']}%  n={t}")

    # ─────────────────────────────────────────────────────────
    # 4. Save Artifacts
    # ─────────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(LABEL_ENCODERS, f)

    metadata = {
        "best_model":      best_name,
        "model_results":   results,
        "feature_names":   feature_names,
        "feature_importances": importances,
        "fairness_report": fairness_report,
        "dataset_size":    int(len(y)),
        "approval_rate":   round(float(y.sum()) / len(y) * 100, 1),
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n  [SAVED] model/model.pkl, model/scaler.pkl, model/encoders.pkl, model/metadata.json")
    print("="*55 + "\n")


if __name__ == "__main__":
    train()
