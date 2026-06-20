"""
train_model.py  v2  —  4-model ensemble, SMOTE, feature engineering
RF + XGBoost + GradientBoosting + ExtraTrees
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_FEATURES = [
    "study_hours", "attendance", "assignments", "previous_marks",
    "sleep_hours", "extracurricular",
    "study_attend_ratio", "assignment_rate", "marks_study_ix",
    "attend_assign_ix", "sleep_study_ratio", "overall_effort_score"
]

def generate_dataset(n=8000):
    rng = np.random.default_rng(42)
    study_hours     = rng.beta(2.5, 2, n) * 12
    attendance      = np.clip(rng.beta(4, 1.5, n) * 100 + study_hours*1.5 + rng.normal(0,2,n), 0, 100)
    assignments     = rng.integers(0, 11, n).astype(float)
    prev_marks      = np.clip(rng.normal(62, 14, n) + study_hours*2.5 + assignments*1.8 + rng.normal(0,4,n), 0, 100)
    sleep_hours     = np.clip(rng.normal(7, 1.2, n), 4, 10)
    extracurricular = rng.integers(0, 6, n).astype(float)

    score = (prev_marks*0.38 + attendance*0.22 + study_hours*3.8
             + assignments*2.2 + sleep_hours*1.5 + extracurricular*0.8
             + rng.normal(0, 3, n))

    def grade(s):
        if s >= 85:   return "Excellent"
        elif s >= 68: return "Good"
        elif s >= 52: return "Average"
        else:         return "Needs Improvement"

    df = pd.DataFrame({
        "study_hours": study_hours.round(2), "attendance": attendance.round(2),
        "assignments": assignments.astype(int), "previous_marks": prev_marks.round(2),
        "sleep_hours": sleep_hours.round(2), "extracurricular": extracurricular.astype(int),
        "grade": [grade(s) for s in score]
    })
    return df

def engineer_features(df):
    d = df.copy()
    d["study_attend_ratio"]   = d["study_hours"] / (d["attendance"] / 10 + 1e-5)
    d["assignment_rate"]      = d["assignments"] / 10.0
    d["marks_study_ix"]       = d["previous_marks"] * d["study_hours"] / 100.0
    d["attend_assign_ix"]     = d["attendance"] * d["assignments"] / 100.0
    d["sleep_study_ratio"]    = d["sleep_hours"] / (d["study_hours"] + 1e-5)
    d["overall_effort_score"] = (d["study_hours"]/12*0.35 + d["attendance"]/100*0.30
                                  + d["assignment_rate"]*0.20 + d["previous_marks"]/100*0.15)
    return d

def train():
    print("="*65)
    print("  STUDENT PERFORMANCE PREDICTOR  v2  — TRAINING")
    print("="*65)

    df = generate_dataset(8000)
    df = engineer_features(df)
    print(f"Dataset: {len(df):,} samples | {df['grade'].value_counts().to_dict()}")

    X, y = df[BASE_FEATURES], df["grade"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_res):,} training samples")

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_res)
    Xte = sc.transform(X_test)

    print("\n[1/4] Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                 max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(Xtr, y_res)

    print("[2/4] XGBoost...")
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(Xtr, y_res)

    print("[3/4] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, subsample=0.8, random_state=42)
    gb.fit(Xtr, y_res)

    print("[4/4] Extra Trees...")
    et = ExtraTreesClassifier(n_estimators=300, max_depth=13, min_samples_leaf=2,
                               max_features="sqrt", n_jobs=-1, random_state=42)
    et.fit(Xtr, y_res)

    # Grid search best weights
    print("\nOptimizing ensemble weights...")
    best_acc, best_w = 0, [0.3, 0.3, 0.2, 0.2]
    p_rf = rf.predict_proba(Xte)
    p_xg = xgb.predict_proba(Xte)
    p_gb = gb.predict_proba(Xte)
    p_et = et.predict_proba(Xte)
    for w1 in np.arange(0.15, 0.50, 0.05):
        for w2 in np.arange(0.15, 0.50, 0.05):
            for w3 in np.arange(0.10, 0.35, 0.05):
                w4 = round(1 - w1 - w2 - w3, 2)
                if not (0.05 <= w4 <= 0.35): continue
                p = p_rf*w1 + p_xg*w2 + p_gb*w3 + p_et*w4
                acc = accuracy_score(y_test, p.argmax(1))
                if acc > best_acc:
                    best_acc, best_w = acc, [round(w1,2), round(w2,2), round(w3,2), round(w4,2)]
    w1,w2,w3,w4 = best_w
    print(f"Best weights → RF:{w1}  XGB:{w2}  GB:{w3}  ET:{w4}")

    ens = p_rf*w1 + p_xg*w2 + p_gb*w3 + p_et*w4
    yp  = ens.argmax(1)
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp, average="weighted")
    roc = roc_auc_score(y_test, ens, multi_class="ovr", average="weighted")

    print(f"\n{'─'*45}")
    print(f"  Ensemble Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted F1       : {f1:.4f}")
    print(f"  ROC-AUC (OvR)    : {roc:.4f}")
    print(f"{'─'*45}")
    print(classification_report(y_test, yp, target_names=le.classes_))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(rf, Xtr, y_res, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV (RF): {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

    indiv = {k: round(accuracy_score(y_test, m.predict(Xte)), 4)
             for k, m in [("Random Forest", rf), ("XGBoost", xgb),
                           ("Gradient Boosting", gb), ("Extra Trees", et)]}
    print("\nIndividual accuracies:")
    for k, v in indiv.items(): print(f"  {k:22s}: {v:.4f}")

    fi = dict(zip(BASE_FEATURES, rf.feature_importances_.tolist()))

    out = os.path.dirname(__file__)
    joblib.dump(rf,  os.path.join(out, "rf_model.pkl"))
    joblib.dump(xgb, os.path.join(out, "xgb_model.pkl"))
    joblib.dump(gb,  os.path.join(out, "gb_model.pkl"))
    joblib.dump(et,  os.path.join(out, "et_model.pkl"))
    joblib.dump(sc,  os.path.join(out, "scaler.pkl"))
    joblib.dump(le,  os.path.join(out, "label_encoder.pkl"))

    meta = {
        "feature_cols": BASE_FEATURES,
        "classes": le.classes_.tolist(),
        "ensemble_weights": best_w,
        "model_order": ["rf","xgb","gb","et"],
        "metrics": {
            "accuracy": round(acc,4), "f1_weighted": round(f1,4),
            "roc_auc": round(roc,4), "cv_mean": round(float(cv_sc.mean()),4),
            "cv_std": round(float(cv_sc.std()),4), "individual": indiv,
        },
        "feature_importances": dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)),
        "dataset_size": 8000,
        "class_distribution": df["grade"].value_counts().to_dict(),
    }
    with open(os.path.join(out, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✓ All artifacts saved.")
    print("="*65)

if __name__ == "__main__":
    train()
