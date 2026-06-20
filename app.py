"""app.py  —  AI Student Performance Predictor  v2"""

import os, json, logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_artifacts():
    try:
        arts = {
            "rf":      joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl")),
            "xgb":     joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
            "gb":      joblib.load(os.path.join(MODEL_DIR, "gb_model.pkl")),
            "et":      joblib.load(os.path.join(MODEL_DIR, "et_model.pkl")),
            "scaler":  joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            "encoder": joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl")),
        }
        with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
            arts["meta"] = json.load(f)
        logger.info("All artifacts loaded.")
        return arts
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}. Run: python model/train_model.py")
        return None

ARTIFACTS = load_artifacts()

def engineer_row(study_hours, attendance, assignments, previous_marks, sleep_hours, extracurricular):
    study_attend_ratio   = study_hours / (attendance / 10 + 1e-5)
    assignment_rate      = assignments / 10.0
    marks_study_ix       = previous_marks * study_hours / 100.0
    attend_assign_ix     = attendance * assignments / 100.0
    sleep_study_ratio    = sleep_hours / (study_hours + 1e-5)
    overall_effort_score = (study_hours/12*0.35 + attendance/100*0.30
                            + assignment_rate*0.20 + previous_marks/100*0.15)
    return [study_hours, attendance, assignments, previous_marks,
            sleep_hours, extracurricular,
            study_attend_ratio, assignment_rate, marks_study_ix,
            attend_assign_ix, sleep_study_ratio, overall_effort_score]

def predict(study_hours, attendance, assignments, previous_marks, sleep_hours, extracurricular):
    arts = ARTIFACTS
    cols = arts["meta"]["feature_cols"]
    row  = engineer_row(study_hours, attendance, assignments, previous_marks, sleep_hours, extracurricular)
    X    = pd.DataFrame([row], columns=cols)
    Xsc  = arts["scaler"].transform(X)
    w    = arts["meta"]["ensemble_weights"]
    ens  = (arts["rf"].predict_proba(Xsc)  * w[0] +
            arts["xgb"].predict_proba(Xsc) * w[1] +
            arts["gb"].predict_proba(Xsc)  * w[2] +
            arts["et"].predict_proba(Xsc)  * w[3])
    idx   = ens[0].argmax()
    label = arts["encoder"].classes_[idx]
    conf  = float(ens[0][idx])
    conf_lbl = "Very High" if conf>=0.80 else "High" if conf>=0.60 else "Moderate" if conf>=0.45 else "Low"
    class_probs = {c: round(float(p)*100,1) for c, p in zip(arts["encoder"].classes_, ens[0])}
    insights = generate_insights(study_hours, attendance, assignments, previous_marks, sleep_hours, extracurricular, label)
    return {"grade": label, "confidence": round(conf*100,1),
            "confidence_lvl": conf_lbl, "class_probs": class_probs, "insights": insights,
            "feature_importances": arts["meta"]["feature_importances"]}

def generate_insights(study, attend, assign, marks, sleep, extra, grade):
    tips = []
    if study < 3:
        tips.append({"icon":"📚","title":"Low Study Time","detail":f"{study:.1f} hrs/day is below average. Top performers study 6–8 hrs."})
    elif study >= 6:
        tips.append({"icon":"✅","title":"Strong Study Habit","detail":f"{study:.1f} hrs/day puts you in the top academic tier."})
    else:
        tips.append({"icon":"📖","title":"Moderate Study Time","detail":f"{study:.1f} hrs/day is decent — pushing to 6+ hrs can lift your grade."})
    if attend < 75:
        tips.append({"icon":"🚨","title":"Critical: Low Attendance","detail":f"{attend:.0f}% risks academic penalties. Target ≥ 85%."})
    elif attend < 85:
        tips.append({"icon":"📅","title":"Improve Attendance","detail":f"{attend:.0f}% is borderline. Consistency above 90% is ideal."})
    else:
        tips.append({"icon":"✅","title":"Excellent Attendance","detail":f"{attend:.0f}% — attendance is one of the strongest grade predictors."})
    if assign < 7:
        tips.append({"icon":"📝","title":"Submit Missing Assignments","detail":f"{10-assign} unsubmitted assignments. Each directly lowers your score."})
    else:
        tips.append({"icon":"✅","title":"Good Assignment Record","detail":f"{assign}/10 submitted. Keep the momentum going."})
    if marks < 50:
        tips.append({"icon":"📊","title":"Academic Support Recommended","detail":f"Previous score: {marks:.0f}/100. Consider tutoring or extra office hours."})
    elif marks >= 80:
        tips.append({"icon":"🌟","title":"Strong Prior Performance","detail":f"{marks:.0f}/100 shows solid academic foundation."})
    if sleep < 6:
        tips.append({"icon":"😴","title":"Sleep Deprivation Risk","detail":f"{sleep:.1f} hrs sleep harms memory consolidation. Aim for 7–8 hrs."})
    elif sleep > 9:
        tips.append({"icon":"⚠️","title":"Oversleeping May Signal Issues","detail":f"{sleep:.1f} hrs may reduce productive study hours."})
    return tips

# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════

@app.route("/")
def index():
    meta = ARTIFACTS["meta"] if ARTIFACTS else {}
    return render_template("index.html", model_meta=meta)

@app.route("/dashboard")
def dashboard():
    stats_path = os.path.join(MODEL_DIR, "dataset_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    return render_template("dashboard.html", stats=stats)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ARTIFACTS:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    data = request.get_json(force=True)
    try:
        sh   = float(data["study_hours"])
        att  = float(data["attendance"])
        asgn = int(data["assignments"])
        pm   = float(data["previous_marks"])
        slp  = float(data.get("sleep_hours", 7))
        ext  = int(data.get("extracurricular", 2))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    errs = []
    if not (0<=sh<=24):   errs.append("study_hours: 0–24")
    if not (0<=att<=100): errs.append("attendance: 0–100")
    if not (0<=asgn<=10): errs.append("assignments: 0–10")
    if not (0<=pm<=100):  errs.append("previous_marks: 0–100")
    if not (4<=slp<=12):  errs.append("sleep_hours: 4–12")
    if not (0<=ext<=5):   errs.append("extracurricular: 0–5")
    if errs:
        return jsonify({"error": "Validation: " + " | ".join(errs)}), 400
    return jsonify(predict(sh, att, asgn, pm, slp, ext))

@app.route("/api/model-info", methods=["GET"])
def model_info():
    if not ARTIFACTS: return jsonify({"error": "Model not loaded."}), 503
    m = ARTIFACTS["meta"]
    return jsonify({"metrics": m["metrics"], "feature_importances": m["feature_importances"],
                    "class_distribution": m["class_distribution"], "dataset_size": m["dataset_size"],
                    "classes": m["classes"]})

@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():
    stats_path = os.path.join(MODEL_DIR, "dataset_stats.json")
    if not os.path.exists(stats_path):
        return jsonify({"error": "Run dashboard.py first"}), 503
    with open(stats_path) as f:
        return jsonify(json.load(f))

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model":"loaded" if ARTIFACTS else "not loaded","version":"2.0.0"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
