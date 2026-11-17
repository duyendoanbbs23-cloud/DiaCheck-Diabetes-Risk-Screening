import json
import os
import copy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from i18n import LocalizationError, list_supported_languages, merge_catalog, translate_catalog

ROOT = Path(__file__).parent
CHARACTER_IMAGE = ROOT / "assets" / "diacheck_medical_team.png"

MANUAL_LANGUAGE_NAMES = {
    "en": "English", "vi": "Tiếng Việt", "fr": "Français",
    "zh": "中文（简体）", "ja": "日本語", "ko": "한국어", "es": "Español",
    "pt": "Português", "de": "Deutsch", "it": "Italiano", "ru": "Русский",
    "ar": "العربية", "hi": "हिन्दी", "th": "ไทย", "id": "Bahasa Indonesia",
    "tr": "Türkçe", "nl": "Nederlands", "pl": "Polski", "uk": "Українська",
}

BASE_EN = {
    "tagline": "A simple first step toward understanding your health.",
    "nav": "Navigation", "home": "Home", "sim": "Cost & threshold",
    "fair": "Fairness", "about": "About & model card",
    "hero": "Understand your health. Take your next step with confidence.",
    "intro": "Answer a few questions about your health and receive an educational estimate.",
    "privacy": "Your answers are processed in this app session and are not saved.",
    "screen": "Start your screening", "age": "Age group",
    "sex": "Sex (as recorded in the source survey)", "female": "Female", "male": "Male",
    "bp": "Have you ever been told you have high blood pressure?",
    "chol": "Have you ever been told you have high cholesterol?",
    "bmi": "What is your BMI?",
    "smoke": "Have you smoked at least 100 cigarettes in your lifetime?",
    "stroke": "Have you ever had a stroke?",
    "heart": "Have you had coronary heart disease or a heart attack?",
    "activity": "In the past 30 days, other than your regular job, did you participate in any physical activities?",
    "health": "In general, would you say your health is:",
    "yes": "Yes", "no": "No", "submit": "Estimate my screening result",
    "result": "Your screening result", "estimate": "Model output score (0–100)",
    "above": "This estimate is above the current decision threshold.",
    "below": "This estimate is below the current decision threshold.",
    "action_high": "Consider arranging a blood sugar test and discussing your risk with a qualified healthcare professional.",
    "action_low": "This result alone does not indicate that a test is unnecessary. If you have symptoms or concerns, contact a healthcare professional.",
    "not_diag": "This is not a diagnosis. Only an appropriate clinical assessment and testing can determine diabetes status.",
    "cost_title": "Explore the decision threshold",
    "cost_desc": "Change the relative cost assigned to each type of error.",
    "fn": "Cost of a false negative", "fp": "Cost of a false positive",
    "threshold": "Resulting decision threshold",
    "example": "The app recommends considering a test when the estimated likelihood is at or above this threshold.",
    "explain": "What influenced this estimate?",
    "up": "Factors pushing the estimate up", "down": "Factors pushing the estimate down",
    "contrib": "Relative contribution to this model's score (not causal effects).",
    "fair_title": "Fairness and subgroup performance",
    "fair_desc": "Held-out test-set metrics by demographic and socioeconomic groups.",
    "dimension": "Compare by", "n": "Sample size", "sens": "Sensitivity",
    "spec": "Specificity", "prec": "Precision", "pos": "Observed positive share",
    "predpos": "Predicted positive share",
    "about_title": "About DiaCheck", "data": "Data source",
    "metrics": "Held-out test metrics", "limitations": "Important limitations",
    "disclaimer": "Medical disclaimer", "intended": "Intended use",
    "limitations_list": [
        "The source is a 2015 US telephone survey; it may not represent today's population.",
        "All health indicators are self-reported.",
        "The model predicts a survey label, not a laboratory-confirmed diagnosis.",
        "Performance and error rates can differ across population groups.",
    ],
    "disclaimer_text": "DiaCheck is an educational student project, not a medical device and not for clinical use. It cannot diagnose, rule out, or treat diabetes.",
    "intended_text": "Adults answering questions at home who want an educational prompt to consider a blood sugar test.",
    "footer": "Educational project — not for clinical use.",
    "select_lang": "Language", "general_health": "General health",
    "excellent": "Excellent", "very_good": "Very good", "good": "Good",
    "fair": "Fair", "poor": "Poor",
    "age_options": ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49",
                     "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+"],
    "error": "Please enter a BMI between 12 and 98.",
    "start": "Please complete the screening form to see your estimate.",
    "no_result": "Complete the form on the Home page to see a case-specific explanation.",
    "metric_auc": "ROC-AUC", "metric_ap": "Average precision",
    "metric_brier": "Brier score", "metric_recall": "Sensitivity",
    "metric_acc": "Accuracy", "metric_precision": "Precision",
    "metric_f1": "F1 score", "metric_spec": "Specificity",
    "split": "Evaluation design",
    "split_text": "Stratified 80/20 train/test split; random state 42.",
    "score_note": "This is the model's output on a 0–100 scale. Not a medically validated risk score or diagnosis.",
    "result_score_explain": "Calculated by the computer from your answers. Educational reference only.",
    "factor_more": "Answers linked with a higher model score",
    "factor_less": "Answers linked with a lower model score",
    "next_steps_text": "If concerned, have symptoms, or want to know your blood sugar, contact a healthcare professional.",
    "reset_form": "Reset / start over",
    "feature_plain": {
        "HighBP": "You reported high blood pressure.",
        "HighChol": "You reported high cholesterol.",
        "CholCheck": "Cholesterol check in the past 5 years.",
        "BMI": "Your BMI is considered by the model.",
        "Smoker": "Your smoking-history answer is considered.",
        "Stroke": "You reported having had a stroke.",
        "HeartDiseaseorAttack": "You reported heart disease or heart attack.",
        "PhysActivity": "Your physical-activity answer is considered.",
        "Fruits": "Your fruit intake is considered.",
        "Veggies": "Your vegetable intake is considered.",
        "HvyAlcoholConsump": "Your alcohol intake is considered.",
        "AnyHealthcare": "Your healthcare coverage is considered.",
        "NoDocbcCost": "Cost-related inability to see a doctor is considered.",
        "GenHlth": "Your general-health rating is considered.",
        "MentHlth": "Your poor mental-health days are considered.",
        "PhysHlth": "Your poor physical-health days are considered.",
        "DiffWalk": "Your difficulty walking is considered.",
        "Age": "Your age group is considered.",
        "Sex": "This survey characteristic is a model input.",
        "Education": "Your education level is considered.",
        "Income": "Your household income is considered.",
    },
    "feature_labels": {
        "HighBP": "High blood pressure", "HighChol": "High cholesterol",
        "CholCheck": "Cholesterol check", "BMI": "Body mass index (BMI)",
        "Smoker": "Smoking history", "Stroke": "History of stroke",
        "HeartDiseaseorAttack": "Heart disease or heart attack",
        "PhysActivity": "Physical activity", "Fruits": "Fruit intake",
        "Veggies": "Vegetable intake", "HvyAlcoholConsump": "Heavy alcohol intake",
        "AnyHealthcare": "Health care coverage", "NoDocbcCost": "Unable to see a doctor due to cost",
        "GenHlth": "General health", "MentHlth": "Mental health",
        "PhysHlth": "Physical health", "DiffWalk": "Difficulty walking",
        "Age": "Age group", "Sex": "Sex (survey coding)",
        "Education": "Education level", "Income": "Household income",
    },
    "brand_caption": "DIABETES HEALTH CHECK", "home_eyebrow": "A FRIENDLY HEALTH CHECK",
    "sim_eyebrow": "DECISION ANALYSIS", "fair_eyebrow": "MODEL REVIEW",
    "about_eyebrow": "ABOUT THIS PROJECT", "basic_information": "Basic information",
    "education_levels": ["Never attended school / kindergarten", "Grades 1–8",
                         "Grades 9–11", "Grade 12 / GED", "College 1–3 years",
                         "College 4+ years"],
    "income_levels": ["Below $10,000", "$10,000–14,999", "$15,000–19,999",
                      "$20,000–24,999", "$25,000–34,999", "$35,000–49,999",
                      "$50,000–74,999", "$75,000 or more"],
    "questions": {
        "height": "Height (cm)", "weight": "Weight (kg)", "bmi_auto": "BMI is calculated automatically",
        "cholcheck": "Have you had your cholesterol checked in the past 5 years?",
        "fruits": "Do you eat fruit at least once a day?",
        "veggies": "Do you eat vegetables at least once a day?",
        "alcohol": "Do you usually have more than 14 alcoholic drinks per week (men) or 7 (women)?",
        "healthcare": "Do you have any health insurance or health care coverage?",
        "cost": "In the past 12 months, did you need to see a doctor but could not because of the cost?",
        "mental": "During the past 30 days, on how many days was your mental health not good?",
        "physical": "During the past 30 days, on how many days was your physical health not good?",
        "walk": "Do you have serious difficulty walking or climbing stairs?",
        "education": "What is your highest level of education?",
        "income": "What is your annual household income?",
        "fruit_help": "Average servings per day (0–5)",
        "veg_help": "Average servings per day (0–5)",
    },
    "dimensions": {"Age group": "Age group", "Sex group": "Sex group",
                    "Income group": "Income group", "Education group": "Education group"},
    "group_label": "Group", "metric_label": "Metric", "score_label": "Score",
    "decision_rule": "Threshold = FP cost / (FP cost + FN cost). Relative assumptions, not medical advice.",
    "subgroup_note": "Subgroup metrics are descriptive. Differences alone do not prove bias or fairness.",
    "data_source_text": "Diabetes Health Indicators, CDC BRFSS 2015 (Kaggle).",
    "rows_label": "Survey records", "original_features_label": "Original indicators",
    "model_inputs_label": "Model inputs", "model_card_title": "Model card",
    "model_label": "Model", "target_label": "Target", "evaluation_label": "Evaluation",
    "fairness_label": "Fairness review",
    "model_target_text": "Self-reported diabetes status (binary survey label)",
    "fairness_detail": "Age, sex, income, and education groups are reviewed.",
    "source_detail": "Details in README.md and models/model_meta.json.",
    "machine_translation_note": "This interface is machine-translated. Medical wording may contain errors.",
    "global_language_setup": "Add a Google Cloud Translation Basic API key to Streamlit Secrets for more languages.",
    "sidebar_source_label": "CDC BRFSS 2015 · Educational project",
    "language_api_error": "Google Cloud Translation unavailable. Showing built-in languages.",
    "training_metrics_notice": "Run `python train.py` to create the model and evaluation metrics.",
    "metric_threshold_info": "Decision threshold used for these classification metrics:",
    "plot_group_axis": "Group", "plot_score_axis": "Rate",
}

VI_PACK = {
    "select_lang": "Ngôn ngữ", "yes": "Có", "no": "Không",
    "nav": "Điều hướng", "home": "Trang chủ", "sim": "Chi phí & ngưỡng",
    "fair": "Công bằng mô hình", "about": "Giới thiệu & Model Card",
    "hero": "Cùng tìm hiểu sức khỏe và chủ động bước tiếp theo.",
    "result": "Kết quả đánh giá sức khỏe", "estimate": "Điểm đầu ra của mô hình (0–100)",
    "submit": "Ước tính kết quả sàng lọc", "screen": "Bắt đầu sàng lọc",
    "footer": "Dự án giáo dục — không dùng cho mục đích lâm sàng.",
}

LANGUAGE_PACKS = {"en": BASE_EN, "vi": VI_PACK}
for _code in MANUAL_LANGUAGE_NAMES:
    LANGUAGE_PACKS.setdefault(_code, {})


def build_locale(language_code: str) -> dict:
    return merge_catalog(copy.deepcopy(BASE_EN), LANGUAGE_PACKS.get(language_code, {}))


st.set_page_config(page_title="DiaCheck", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f6fbf8 0%,#f8fafc 48%,#f5f8f7 100%);}
.block-container {padding-top:2.0rem; padding-bottom:3rem; max-width:1200px;}
h1,h2,h3 {color:#153b32;}
.hero {background:linear-gradient(115deg,#0f513f 0%,#16745a 55%,#2b8b69 100%);padding:2.2rem 2.4rem;border-radius:24px;color:white;margin-bottom:1.25rem;}
.hero h1 {color:white;font-size:2.35rem;margin:0 0 .7rem 0;}
.hero p {color:#e3f5ec;font-size:1.05rem;}
.eyebrow {letter-spacing:.13em;text-transform:uppercase;font-size:.72rem;font-weight:700;color:#b8e4d1;margin-bottom:.7rem;}
.panel {background:white;border:1px solid #e4eee8;border-radius:18px;padding:1.25rem 1.4rem;}
.metric {background:white;border:1px solid #e4eee8;border-radius:16px;padding:1rem 1.15rem;}
.metric-label {color:#6b7f77;font-size:.82rem;font-weight:600;}
.metric-value {font-size:1.55rem;font-weight:800;color:#174b3b;}
.small-muted {color:#71827c;font-size:.86rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "models" / "pipeline.joblib")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    fair_path = ROOT / "data" / "fairness_metrics.csv"
    fair = pd.read_csv(fair_path) if fair_path.exists() else pd.DataFrame()
    return model, meta, fair


try:
    model, META, FAIR = load_assets()
except FileNotFoundError as exc:
    st.error(f"Model files missing. Run `python train.py`.\n\n{exc}")
    st.stop()

# ============================================================
# CRITICAL FIX: read feature names directly from the fitted pipeline.
# This is the ONLY source of truth. Never use META["features"] for prediction.
# ============================================================
scaler = model.named_steps["scaler"]
MODEL_FEATURES = list(scaler.feature_names_in_)
print(f"[DiaCheck] Fitted model expects {len(MODEL_FEATURES)} features: {MODEL_FEATURES}")

AVAILABLE = set(MODEL_FEATURES)


if "last_case" not in st.session_state:
    st.session_state.last_case = None
if "last_probability" not in st.session_state:
    st.session_state.last_probability = None
if "cost_fn" not in st.session_state:
    st.session_state.cost_fn = float(META["cost_assumptions"]["false_negative"])
if "cost_fp" not in st.session_state:
    st.session_state.cost_fp = float(META["cost_assumptions"]["false_positive"])


def _get_translation_key():
    try:
        return str(st.secrets.get("GOOGLE_TRANSLATE_API_KEY", "") or "").strip()
    except Exception:
        return str(os.getenv("GOOGLE_TRANSLATE_API_KEY", "")).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_supported_languages(api_key):
    return list_supported_languages(api_key)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_translation(catalog_json, language_code, api_key):
    return translate_catalog(json.loads(catalog_json), language_code, api_key)


def _build_locale(language_code, api_key, catalog_json):
    if language_code == "en":
        return copy.deepcopy(BASE_EN), False, False
    if api_key:
        try:
            translated = _cached_translation(catalog_json, language_code, api_key)
            return merge_catalog(translated, LANGUAGE_PACKS.get(language_code, {})), False, False
        except LocalizationError:
            return build_locale(language_code), False, True
    return build_locale(language_code), False, False


translation_api_key = _get_translation_key()
language_catalog_json = json.dumps(BASE_EN, ensure_ascii=False, sort_keys=True)
language_api_error = None

if translation_api_key:
    try:
        available_languages = _cached_supported_languages(translation_api_key)
    except LocalizationError:
        available_languages = list(MANUAL_LANGUAGE_NAMES.items())
        language_api_error = True
else:
    available_languages = list(MANUAL_LANGUAGE_NAMES.items())

language_names = {code: name for code, name in available_languages}
for _code, _name in MANUAL_LANGUAGE_NAMES.items():
    language_names.setdefault(_code, _name)
language_options = sorted(language_names, key=lambda code: language_names[code].casefold())

if "language_code" not in st.session_state:
    st.session_state.language_code = "en"
if st.session_state.language_code not in language_options:
    st.session_state.language_code = "en"

L, _mt, locale_translation_failed = _build_locale(
    st.session_state.language_code, translation_api_key, language_catalog_json
)


with st.sidebar:
    st.markdown("## 🩺 DiaCheck")
    st.caption(L["brand_caption"])
    st.selectbox("🌐 " + L["select_lang"], options=language_options,
                 format_func=lambda code: language_names.get(code, code),
                 key="language_code")
    if locale_translation_failed or language_api_error:
        st.warning(L["language_api_error"])
    page_key = st.radio(L["nav"], options=["home", "sim", "fair", "about"],
                        format_func=lambda key: L[key],
                        label_visibility="collapsed", key="page_navigation")
    st.divider()
    st.info("🎓 " + L["footer"])
    st.caption(L["sidebar_source_label"])


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def contribution_table(case, locale):
    sc = model.named_steps["scaler"]
    lr = model.named_steps["model"]
    arr = np.array([case[f] for f in MODEL_FEATURES], dtype=float)
    z = (arr - sc.mean_) / sc.scale_
    contributions = z * lr.coef_[0]
    labels = locale.get("feature_labels", {})
    out = pd.DataFrame({
        "key": MODEL_FEATURES,
        "feature": [labels.get(f, f) for f in MODEL_FEATURES],
        "contribution": contributions,
    })
    return out.sort_values("contribution", ascending=False)


def render_hero(eyebrow, title, description):
    a, b = st.columns([4.5, 1.25], vertical_alignment="center")
    with a:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{title}</h1><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        if CHARACTER_IMAGE.exists():
            st.image(str(CHARACTER_IMAGE), width=220)


def _reset_screening():
    for key in list(st.session_state.keys()):
        if key in {"age_group", "sex", "height_cm", "weight_kg", "high_bp", "high_chol",
                   "chol_check", "smoking_history", "stroke_history", "heart_history",
                   "physical_activity", "fruit_intake", "vegetable_intake", "alcohol_intake",
                   "healthcare_coverage", "care_access_cost", "general_health_rating",
                   "mental_health_days", "physical_health_days", "walking_difficulty",
                   "education_level", "household_income", "last_case", "last_probability"}:
            st.session_state.pop(key, None)
    st.rerun()


if page_key == "home":
    render_hero(L["home_eyebrow"], L["hero"], L["intro"])
    st.caption("🔒 " + L["privacy"])
    st.subheader(L["screen"])
    q = L["questions"]
    yes_no = [False, True]
    yes_no_label = lambda v: L["yes"] if v else L["no"]

    with st.form("screening_form"):
        st.markdown("#### " + L["basic_information"])
        c1, c2 = st.columns(2)
        with c1:
            age_labels = L["age_options"]
            age_idx = st.selectbox(L["age"], list(range(1, 14)),
                                   format_func=lambda x: age_labels[x - 1],
                                   index=6, key="age_group")
            sex_label = st.radio(L["sex"], ["female", "male"],
                                 format_func=lambda v: L[v], horizontal=True, key="sex")
            height_cm = st.number_input(q["height"], min_value=100.0, max_value=230.0,
                                        value=170.0, step=1.0, key="height_cm")
            weight_kg = st.number_input(q["weight"], min_value=25.0, max_value=300.0,
                                        value=70.0, step=0.5, key="weight_kg")
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.metric(q["bmi_auto"], f"{bmi:.1f} kg/m²")
            bp = st.radio(L["bp"], yes_no, format_func=yes_no_label, horizontal=True, key="high_bp")
            chol = st.radio(L["chol"], yes_no, format_func=yes_no_label, horizontal=True, key="high_chol")
            cholcheck = st.radio(q["cholcheck"], yes_no, format_func=yes_no_label,
                                 horizontal=True, key="chol_check") if "CholCheck" in AVAILABLE else True
            smoker = st.radio(L["smoke"], yes_no, format_func=yes_no_label, horizontal=True, key="smoking_history")
            stroke = st.radio(L["stroke"], yes_no, format_func=yes_no_label, horizontal=True, key="stroke_history")
            heart = st.radio(L["heart"], yes_no, format_func=yes_no_label, horizontal=True, key="heart_history")
        with c2:
            activity = st.radio(L["activity"], yes_no, format_func=yes_no_label, horizontal=True, key="physical_activity")
            fruits = st.radio(q["fruits"], yes_no, format_func=yes_no_label,
                              horizontal=True, key="fruit_intake") if "Fruits" in AVAILABLE else True
            veggies = st.radio(q["veggies"], yes_no, format_func=yes_no_label,
                               horizontal=True, key="vegetable_intake") if "Veggies" in AVAILABLE else True
            heavy_alcohol = st.radio(q["alcohol"], yes_no, format_func=yes_no_label,
                                     horizontal=True, key="alcohol_intake") if "HvyAlcoholConsump" in AVAILABLE else False
            healthcare = st.radio(q["healthcare"], yes_no, format_func=yes_no_label,
                                  horizontal=True, key="healthcare_coverage") if "AnyHealthcare" in AVAILABLE else True
            no_doc_cost = st.radio(q["cost"], yes_no, format_func=yes_no_label,
                                   horizontal=True, key="care_access_cost") if "NoDocbcCost" in AVAILABLE else False
            health_labels = [L["excellent"], L["very_good"], L["good"], L["fair"], L["poor"]]
            health_idx = st.select_slider(L["general_health"], options=[1, 2, 3, 4, 5],
                                          format_func=lambda x: health_labels[x - 1], value=3,
                                          key="general_health_rating")
            mental_days = st.slider(q["mental"], 0, 30, 0, key="mental_health_days") if "MentHlth" in AVAILABLE else 0
            physical_days = st.slider(q["physical"], 0, 30, 0, key="physical_health_days") if "PhysHlth" in AVAILABLE else 0
            diff_walk = st.radio(q["walk"], yes_no, format_func=yes_no_label,
                                 horizontal=True, key="walking_difficulty") if "DiffWalk" in AVAILABLE else False
            if "Education" in AVAILABLE:
                education_labels = L["education_levels"]
                education = st.selectbox(q["education"], options=list(range(1, 7)),
                                         format_func=lambda x: education_labels[x - 1],
                                         index=5, key="education_level")
            else:
                education = 5
            if "Income" in AVAILABLE:
                income_labels = L["income_levels"]
                income = st.selectbox(q["income"], options=list(range(1, 9)),
                                      format_func=lambda x: income_labels[x - 1],
                                      index=7, key="household_income")
            else:
                income = 7
        submitted = st.form_submit_button(L["submit"], use_container_width=True, key="submit_screening")

    if submitted:
        if not 12 <= bmi <= 98:
            st.error(L["error"])
        else:
            # Build ALL possible features, then filter to what model expects.
            full_case = {
                "HighBP": int(bp), "HighChol": int(chol), "CholCheck": int(cholcheck),
                "BMI": float(round(bmi)), "Smoker": int(smoker), "Stroke": int(stroke),
                "HeartDiseaseorAttack": int(heart), "PhysActivity": int(activity),
                "Fruits": int(fruits), "Veggies": int(veggies),
                "HvyAlcoholConsump": int(heavy_alcohol), "AnyHealthcare": int(healthcare),
                "NoDocbcCost": int(no_doc_cost), "GenHlth": int(health_idx),
                "MentHlth": int(mental_days), "PhysHlth": int(physical_days),
                "DiffWalk": int(diff_walk), "Sex": 0 if sex_label == "female" else 1,
                "Age": int(age_idx), "Education": int(education), "Income": int(income),
            }
            case = {f: full_case[f] for f in MODEL_FEATURES}
            try:
                proba = model.predict_proba(pd.DataFrame([case], columns=MODEL_FEATURES))[0, 1]
                st.session_state.last_case = case
                st.session_state.last_probability = float(proba)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Model expects:", MODEL_FEATURES)
                st.write("We sent:", list(case.keys()))

    if st.session_state.last_probability is not None:
        p = st.session_state.last_probability
        threshold = st.session_state.cost_fp / (st.session_state.cost_fp + st.session_state.cost_fn)
        st.markdown("---")
        st.subheader(L["result"])
        a, b = st.columns([1, 1.5])
        with a:
            st.metric(L["estimate"], f"{p * 100:.1f}/100")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(f"{L['threshold']}: {threshold:.3f} · {fmt_pct(threshold)}")
            st.caption(L["score_note"])
            st.caption(L["result_score_explain"])
        with b:
            if p >= threshold:
                st.warning("⚠️ " + L["above"] + "\n\n" + L["action_high"])
            else:
                st.success("✓ " + L["below"] + "\n\n" + L["action_low"])
            st.info(L["not_diag"])
        st.subheader(L["explain"])
        st.caption(L["contrib"])
        cdf = contribution_table(st.session_state.last_case, L)
        pos = cdf[cdf.contribution > 0].head(5)
        neg = cdf[cdf.contribution < 0].sort_values("contribution").head(5)
        x1, x2 = st.columns(2)
        plain = L.get("feature_plain", {})
        feature_labels = L.get("feature_labels", {})
        with x1:
            st.markdown("**" + L["factor_more"] + "**")
            if len(pos):
                for _, r in pos.iterrows():
                    st.write("• " + plain.get(r.key, feature_labels.get(r.key, r.feature)))
            else:
                st.caption("—")
        with x2:
            st.markdown("**" + L["factor_less"] + "**")
            if len(neg):
                for _, r in neg.iterrows():
                    st.write("• " + plain.get(r.key, feature_labels.get(r.key, r.feature)))
            else:
                st.caption("—")
        st.info(L["next_steps_text"])
        if st.button("🔄 " + L["reset_form"], key="reset_screening"):
            _reset_screening()
    else:
        st.info(L["start"])

elif page_key == "sim":
    render_hero(L["sim_eyebrow"], L["cost_title"], L["cost_desc"])
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.cost_fn = st.number_input(L["fn"], min_value=0.1, max_value=100.0,
                                                    value=float(st.session_state.cost_fn),
                                                    step=0.5, key="cost_fn_input")
    with c2:
        st.session_state.cost_fp = st.number_input(L["fp"], min_value=0.1, max_value=100.0,
                                                    value=float(st.session_state.cost_fp),
                                                    step=0.5, key="cost_fp_input")
    threshold = st.session_state.cost_fp / (st.session_state.cost_fp + st.session_state.cost_fn)
    st.markdown(
        f'<div class="panel"><div class="metric-label">{L["threshold"]}</div>'
        f'<div class="metric-value" style="font-size:2.4rem">{threshold:.3f} · {fmt_pct(threshold)}</div>'
        f'<p class="small-muted">{L["example"]}</p></div>', unsafe_allow_html=True)
    if st.session_state.last_probability is not None:
        p = st.session_state.last_probability
        st.subheader(L["result"])
        st.metric(L["estimate"], f"{p * 100:.1f}/100")
        if p >= threshold:
            st.warning(L["above"] + " " + L["action_high"])
        else:
            st.success(L["below"] + " " + L["action_low"])
    else:
        st.info(L["no_result"])
    st.caption(L["decision_rule"])

elif page_key == "fair":
    render_hero(L["fair_eyebrow"], L["fair_title"], L["fair_desc"])
    if FAIR.empty:
        st.info(L["training_metrics_notice"])
    else:
        dims = ["Age group", "Sex group", "Income group", "Education group"]
        avail = [d for d in dims if d in FAIR["dimension"].unique()]
        dim = st.selectbox(L["dimension"], avail,
                          format_func=lambda k: L["dimensions"].get(k, k),
                          key="fairness_dimension")
        sub = FAIR[FAIR.dimension == dim].copy()

        def _loc(row):
            raw = str(row["group"])
            if row["dimension"] == "Sex group":
                if raw.lower() == "female": return L["female"]
                if raw.lower() == "male": return L["male"]
            try:
                code = int(raw.split()[-1])
                if row["dimension"] == "Age group" and 1 <= code <= len(L["age_options"]):
                    return L["age_options"][code - 1]
                if row["dimension"] == "Income group" and 1 <= code <= len(L["income_levels"]):
                    return L["income_levels"][code - 1]
                if row["dimension"] == "Education group" and 1 <= code <= len(L["education_levels"]):
                    return L["education_levels"][code - 1]
            except (ValueError, IndexError):
                pass
            return raw

        if not sub.empty:
            sub["group"] = sub.apply(_loc, axis=1)
        rename = {"group": L["group_label"], "n": L["n"], "sensitivity": L["sens"],
                  "specificity": L["spec"], "precision": L["prec"],
                  "positive_rate": L["pos"], "predicted_positive_rate": L["predpos"]}
        show = sub[["group", "n", "sensitivity", "specificity", "precision",
                    "positive_rate", "predicted_positive_rate"]].rename(columns=rename)
        for c in [L["sens"], L["spec"], L["prec"], L["pos"], L["predpos"]]:
            show[c] = show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)
        plot = sub.melt(id_vars=["group"], value_vars=["sensitivity", "specificity"],
                        var_name="Metric", value_name="Score")
        plot["Metric"] = plot["Metric"].map({"sensitivity": L["sens"], "specificity": L["spec"]})
        fig = px.bar(plot, x="group", y="Score", color="Metric", barmode="group",
                     range_y=[0, 1],
                     labels={"group": L["plot_group_axis"], "Score": L["plot_score_axis"]})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(L["subgroup_note"])

else:
    render_hero(L["about_eyebrow"], L["about_title"], L["intended_text"])
    st.subheader(L["intended"])
    st.write(L["intended_text"])
    st.subheader(L["data"])
    st.write(L["data_source_text"])
    cols = st.columns(3)
    vals = [(L["rows_label"], "253,680"), (L["original_features_label"], "21"),
            (L["model_inputs_label"], str(len(MODEL_FEATURES)))]
    for c, (lb, v) in zip(cols, vals):
        with c:
            st.metric(lb, v)
    st.subheader(L["metrics"])
    m = META.get("metrics") or {}
    if not m or m.get("roc_auc") is None:
        st.info(L["training_metrics_notice"])
    else:
        mcols = st.columns(4)
        cards = [(L["metric_auc"], m["roc_auc"]), (L["metric_ap"], m["average_precision"]),
                 (L["metric_brier"], m["brier_score"]), (L["metric_recall"], m["recall_sensitivity"]),
                 (L["metric_acc"], m["accuracy"]), (L["metric_precision"], m["precision"]),
                 (L["metric_f1"], m["f1"]), (L["metric_spec"], m["specificity"])]
        for i, (lb, v) in enumerate(cards):
            with mcols[i % 4]:
                st.markdown(f'<div class="metric"><div class="metric-label">{lb}</div>'
                            f'<div class="metric-value">{v:.3f}</div></div>', unsafe_allow_html=True)
        st.write(L["split_text"])
    st.subheader(L["limitations"])
    for item in L["limitations_list"]:
        st.markdown(f"- {item}")
    st.subheader(L["disclaimer"])
    st.warning(L["disclaimer_text"])

st.markdown("---")
st.caption("DiaCheck · " + L["footer"])
