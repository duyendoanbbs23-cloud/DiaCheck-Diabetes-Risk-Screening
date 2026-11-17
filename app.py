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
    "zh": "中文（简体）", "zh-TW": "中文（繁體）", "ja": "日本語", "ko": "한국어",
    "es": "Español", "pt": "Português", "de": "Deutsch", "it": "Italiano",
    "nl": "Nederlands", "sv": "Svenska", "da": "Dansk", "no": "Norsk",
    "fi": "Suomi", "pl": "Polski", "cs": "Čeština", "sk": "Slovenčina",
    "hu": "Magyar", "ro": "Română", "el": "Ελληνικά", "bg": "Български",
    "ru": "Русский", "uk": "Українська", "tr": "Türkçe",
    "ar": "العربية", "fa": "فارسی", "he": "עברית", "ur": "اردو",
    "hi": "हिन्दी", "bn": "বাংলা", "ta": "தமிழ்", "te": "తెలుగు",
    "th": "ไทย", "id": "Bahasa Indonesia", "ms": "Bahasa Melayu",
    "tl": "Filipino", "sw": "Kiswahili", "am": "አማርኛ",
}

# ---------------------------------------------------------------------------
# BUILT-IN TRANSLATION PACKS
# Each pack overrides English on a key-by-key basis via merge_catalog().
# Missing keys fall back to English so the UI never breaks.
# ---------------------------------------------------------------------------
BASE_EN = {
    "tagline": "A simple first step toward understanding your health.",
    "nav": "Navigation", "home": "Home", "sim": "Cost & threshold",
    "fair": "Fairness", "about": "About & model card",
    "hero": "Understand your health. Take your next step with confidence.",
    "intro": "Answer a few questions about your health and receive an educational estimate to help you consider whether to ask for a blood sugar test.",
    "privacy": "Your answers are processed in this app session and are not saved by DiaCheck.",
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
    "action_low": "This result alone does not indicate that a test is unnecessary. If you have symptoms, concerns, or risk factors, contact a healthcare professional.",
    "not_diag": "This is not a diagnosis. Only an appropriate clinical assessment and testing can determine diabetes status.",
    "cost_title": "Explore the decision threshold",
    "cost_desc": "Change the relative cost assigned to each type of error. The threshold is a decision-analysis setting, not a medical cut-off.",
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
        "The source is a 2015 US telephone survey; it may not represent today's population or people in other countries.",
        "All health indicators are self-reported and are not verified clinical measurements.",
        "The model predicts a survey label, not a laboratory-confirmed diagnosis.",
        "The selected features are a subset of the original indicators; performance may differ from models using all features.",
        "Performance and error rates can differ across population groups; subgroup estimates may be affected by sample size.",
    ],
    "disclaimer_text": "DiaCheck is an educational student project, not a medical device and not for clinical use. It cannot diagnose, rule out, or treat diabetes. Do not delay care or make health decisions solely from this result. Seek professional advice and appropriate blood glucose testing.",
    "intended_text": "Adults answering questions at home who want an educational prompt to consider whether to seek a blood sugar test.",
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
    "split_text": "Stratified 80/20 train/test split; random state 42. Metrics below are test-set estimates, not external validation.",
    "score_note": "This is the model's output displayed on a 0–100 scale. It is not a medically validated risk score, a personal probability, or a diagnosis.",
    "result_score_explain": "This number is calculated by the computer from your answers. It is only an educational reference and cannot tell you whether you have diabetes.",
    "factor_more": "Answers linked with a higher model score",
    "factor_less": "Answers linked with a lower model score",
    "next_steps_text": "If you are concerned, have symptoms, or want to know your blood sugar level, contact a healthcare professional to discuss an appropriate test. Do not rely on this score alone.",
    "reset_form": "Reset / start over",
    "feature_plain": {
        "HighBP": "You reported that a health professional had told you that you have high blood pressure.",
        "HighChol": "You reported that a health professional had told you that you have high cholesterol.",
        "CholCheck": "Your cholesterol check in the past 5 years is considered by the model.",
        "BMI": "Your BMI is one of the answers considered by the model.",
        "Smoker": "Your smoking-history answer is considered by the model.",
        "Stroke": "You reported having had a stroke.",
        "HeartDiseaseorAttack": "You reported a history of heart disease or heart attack.",
        "PhysActivity": "Your physical-activity answer is considered by the model.",
        "Fruits": "Your fruit intake is considered by the model.",
        "Veggies": "Your vegetable intake is considered by the model.",
        "HvyAlcoholConsump": "Your alcohol intake is considered by the model.",
        "AnyHealthcare": "Your healthcare coverage is considered by the model.",
        "NoDocbcCost": "Your inability to see a doctor due to cost is considered by the model.",
        "GenHlth": "Your rating of your general health is considered by the model.",
        "MentHlth": "Your number of poor mental-health days is considered by the model.",
        "PhysHlth": "Your number of poor physical-health days is considered by the model.",
        "DiffWalk": "Your difficulty walking is considered by the model.",
        "Age": "Your age group is considered by the model.",
        "Sex": "This survey characteristic is one of the model inputs.",
        "Education": "Your education level is considered by the model.",
        "Income": "Your household income is considered by the model.",
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
        "alcohol": "Do you usually have more than 14 alcoholic drinks per week (men) or 7 per week (women)?",
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
    "decision_rule": "Decision rule: threshold = false-positive cost / (false-positive cost + false-negative cost). Costs are relative assumptions, not measured monetary costs or medical advice.",
    "subgroup_note": "Subgroup metrics are descriptive. Differences alone do not prove bias or fairness. Consider sample sizes, uncertainty, prevalence, and the consequences of errors.",
    "data_source_text": "Diabetes Health Indicators, CDC BRFSS 2015 (Kaggle). The project uses the original, imbalanced dataset.",
    "rows_label": "Number of survey records", "original_features_label": "Original health indicators",
    "model_inputs_label": "Model inputs", "model_card_title": "Model card",
    "model_label": "Model", "target_label": "Target", "evaluation_label": "Evaluation",
    "fairness_label": "Fairness review",
    "model_target_text": "Self-reported diabetes status (binary survey label)",
    "fairness_detail": "Age, sex, income, and education groups are reviewed. Group metrics are descriptive and do not establish fairness on their own.",
    "source_detail": "Source and evaluation details are documented in README.md and models/model_meta.json.",
    "machine_translation_note": "This interface is machine-translated. Medical wording may contain errors. Do not use this educational app to diagnose a condition or make clinical decisions.",
    "global_language_setup": "To enable the expanded language list, add a Google Cloud Translation Basic API key to Streamlit Secrets. The API may incur usage charges.",
    "sidebar_source_label": "CDC BRFSS 2015 · Educational project",
    "language_api_error": "Google Cloud Translation is unavailable. The app is showing its built-in languages instead.",
    "training_metrics_notice": "Run `python train.py` to create the model and evaluation metrics.",
    "metric_threshold_info": "Decision threshold used for these classification metrics:",
    "plot_group_axis": "Group", "plot_score_axis": "Rate",
}

VI_PACK = {
    "tagline": "Một bước đơn giản để hiểu thêm về sức khỏe của bạn.",
    "nav": "Điều hướng", "home": "Trang chủ", "sim": "Chi phí & ngưỡng",
    "fair": "Công bằng mô hình", "about": "Giới thiệu & Model Card",
    "hero": "Cùng tìm hiểu sức khỏe và chủ động bước tiếp theo.",
    "intro": "Trả lời một số câu hỏi sức khỏe để nhận ước tính mang tính giáo dục, hỗ trợ bạn cân nhắc việc trao đổi với nhân viên y tế về xét nghiệm đường huyết.",
    "privacy": "Câu trả lời chỉ được xử lý trong phiên ứng dụng và DiaCheck không lưu lại.",
    "screen": "Bắt đầu sàng lọc", "age": "Nhóm tuổi",
    "sex": "Giới tính (theo cách mã hóa của khảo sát nguồn)",
    "female": "Nữ", "male": "Nam",
    "bp": "Bạn đã từng được thông báo bị huyết áp cao chưa?",
    "chol": "Bạn đã từng được thông báo bị cholesterol cao chưa?",
    "bmi": "Chỉ số BMI của bạn là bao nhiêu?",
    "smoke": "Bạn đã từng hút ít nhất 100 điếu thuốc trong đời chưa?",
    "stroke": "Bạn đã từng bị đột quỵ chưa?",
    "heart": "Bạn đã từng mắc bệnh mạch vành hoặc nhồi máu cơ tim chưa?",
    "activity": "Trong 30 ngày qua, ngoài công việc thường ngày, bạn có tham gia hoạt động thể chất nào không?",
    "health": "Nhìn chung, bạn tự đánh giá sức khỏe của mình như thế nào?",
    "yes": "Có", "no": "Không", "submit": "Ước tính kết quả sàng lọc",
    "result": "Kết quả đánh giá sức khỏe", "estimate": "Điểm đầu ra của mô hình (0–100)",
    "above": "Theo cách tính của DiaCheck, điểm của bạn đang cao hơn mức tham khảo.",
    "below": "Theo cách tính của DiaCheck, điểm của bạn đang thấp hơn mức tham khảo.",
    "action_high": "Bạn nên đến cơ sở y tế để được tư vấn và hỏi bác sĩ về việc kiểm tra đường huyết.",
    "action_low": "Điểm thấp không có nghĩa là chắc chắn không mắc bệnh hoặc không cần kiểm tra. Nếu thấy mệt, khát nước nhiều, tiểu nhiều hoặc lo lắng về sức khỏe, bạn nên hỏi ý kiến nhân viên y tế.",
    "not_diag": "Xin lưu ý: Đây chỉ là kết quả tham khảo từ câu trả lời của bạn, không phải kết luận mắc hay không mắc bệnh.",
    "cost_title": "Khám phá ngưỡng quyết định",
    "cost_desc": "Thay đổi chi phí tương đối của từng loại sai số. Ngưỡng này phục vụ phân tích quyết định, không phải ngưỡng y khoa.",
    "fn": "Chi phí âm tính giả", "fp": "Chi phí dương tính giả",
    "threshold": "Ngưỡng quyết định tương ứng",
    "example": "Ứng dụng khuyến nghị cân nhắc xét nghiệm khi xác suất ước tính bằng hoặc cao hơn ngưỡng này.",
    "explain": "Yếu tố nào ảnh hưởng đến ước tính?",
    "up": "Yếu tố làm tăng điểm ước tính", "down": "Yếu tố làm giảm điểm ước tính",
    "contrib": "Đóng góp tương đối vào điểm của mô hình (không phải quan hệ nhân quả).",
    "fair_title": "Công bằng và hiệu suất theo nhóm",
    "fair_desc": "Chỉ số trên tập kiểm tra, phân tách theo nhóm nhân khẩu học và kinh tế-xã hội.",
    "dimension": "Phân nhóm theo", "n": "Cỡ mẫu", "sens": "Độ nhạy",
    "spec": "Độ đặc hiệu", "prec": "Precision",
    "pos": "Tỷ lệ dương tính thực tế", "predpos": "Tỷ lệ dự đoán dương tính",
    "about_title": "Về DiaCheck", "data": "Nguồn dữ liệu",
    "metrics": "Chỉ số trên tập kiểm tra", "limitations": "Các hạn chế quan trọng",
    "disclaimer": "Tuyên bố y tế", "intended": "Mục đích sử dụng",
    "limitations_list": [
        "Nguồn dữ liệu là khảo sát điện thoại tại Hoa Kỳ năm 2015; có thể không đại diện cho dân số hiện tại hoặc các quốc gia khác.",
        "Các chỉ báo sức khỏe do người tham gia tự khai báo, không được xác minh bằng phép đo lâm sàng.",
        "Mô hình dự đoán nhãn khảo sát, không phải chẩn đoán được xác nhận bằng xét nghiệm.",
        "Các biến được chọn là một tập con của bộ chỉ báo gốc; hiệu suất có thể khác nếu dùng toàn bộ biến.",
        "Hiệu suất và tỷ lệ sai số có thể khác giữa các nhóm dân số; ước tính nhóm nhỏ phụ thuộc vào cỡ mẫu.",
    ],
    "disclaimer_text": "DiaCheck là dự án học thuật mang tính giáo dục, không phải thiết bị y tế và không dùng cho mục đích lâm sàng. Ứng dụng không thể chẩn đoán, loại trừ hay điều trị đái tháo đường. Không trì hoãn chăm sóc y tế hoặc quyết định sức khỏe chỉ dựa trên kết quả này.",
    "intended_text": "Người trưởng thành trả lời câu hỏi tại nhà và muốn có thông tin giáo dục để cân nhắc việc thực hiện xét nghiệm đường huyết.",
    "footer": "Dự án giáo dục — không dùng cho mục đích lâm sàng.",
    "select_lang": "Ngôn ngữ", "general_health": "Sức khỏe tổng quát",
    "excellent": "Rất tốt", "very_good": "Tốt", "good": "Khá",
    "fair": "Trung bình", "poor": "Kém",
    "age_options": ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49",
                     "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+"],
    "error": "Vui lòng nhập BMI trong khoảng 12 đến 98.",
    "start": "Bạn vui lòng trả lời các câu hỏi để xem kết quả đánh giá sức khỏe.",
    "no_result": "Bạn hãy hoàn thành các câu hỏi ở Trang chủ để xem kết quả.",
    "metric_auc": "ROC-AUC", "metric_ap": "Average precision",
    "metric_brier": "Brier score", "metric_recall": "Độ nhạy",
    "metric_acc": "Độ chính xác", "metric_precision": "Precision",
    "metric_f1": "F1 score", "metric_spec": "Độ đặc hiệu",
    "split": "Thiết kế đánh giá",
    "split_text": "Chia tập train/test phân tầng 80/20; random state 42. Các chỉ số là ước tính trên tập kiểm tra, chưa phải xác thực bên ngoài.",
    "score_note": "Đây là đầu ra của mô hình được hiển thị trên thang điểm 0–100. Điểm này chưa được kiểm định y khoa, không phải xác suất cá nhân mắc bệnh và cũng không phải chẩn đoán.",
    "result_score_explain": "Máy tính tạo ra con số này dựa trên câu trả lời của bạn. Đây chỉ là thông tin tham khảo mang tính giáo dục, không thể xác định bạn có mắc tiểu đường hay không.",
    "factor_more": "Câu trả lời gắn với điểm mô hình cao hơn",
    "factor_less": "Câu trả lời gắn với điểm mô hình thấp hơn",
    "next_steps_text": "Nếu bạn lo lắng, có triệu chứng hoặc muốn biết chỉ số đường huyết, hãy đến cơ sở y tế để được tư vấn về xét nghiệm phù hợp. Không nên chỉ dựa vào điểm số này.",
    "reset_form": "Đặt lại / bắt đầu lại",
    "brand_caption": "THÔNG TIN SỨC KHỎE ĐƯỜNG HUYẾT",
    "home_eyebrow": "CÙNG BẠN CHĂM SÓC SỨC KHỎE",
    "sim_eyebrow": "PHÂN TÍCH QUYẾT ĐỊNH",
    "fair_eyebrow": "ĐÁNH GIÁ MÔ HÌNH",
    "about_eyebrow": "VỀ DỰ ÁN",
    "basic_information": "Thông tin cơ bản",
    "education_levels": ["Chưa đi học / mẫu giáo", "Lớp 1–8", "Lớp 9–11",
                          "Lớp 12 / tương đương", "Cao đẳng/đại học 1–3 năm",
                          "Đại học từ 4 năm trở lên"],
    "income_levels": ["Dưới 10.000 USD", "10.000–14.999 USD", "15.000–19.999 USD",
                      "20.000–24.999 USD", "25.000–34.999 USD", "35.000–49.999 USD",
                      "50.000–74.999 USD", "Từ 75.000 USD trở lên"],
    "questions": {
        "height": "Chiều cao (cm)", "weight": "Cân nặng (kg)",
        "bmi_auto": "BMI được tự động tính",
        "cholcheck": "Bạn có kiểm tra cholesterol trong 5 năm qua không?",
        "fruits": "Bạn có ăn trái cây ít nhất một lần mỗi ngày không?",
        "veggies": "Bạn có ăn rau ít nhất một lần mỗi ngày không?",
        "alcohol": "Bạn thường uống hơn 14 đơn vị đồ uống có cồn mỗi tuần (nam) hoặc 7 đơn vị (nữ) không?",
        "healthcare": "Bạn có bảo hiểm y tế hoặc hình thức chăm sóc sức khỏe nào không?",
        "cost": "Trong 12 tháng qua, có khi nào bạn cần đi khám nhưng không thể vì chi phí không?",
        "mental": "Trong 30 ngày qua, có bao nhiêu ngày sức khỏe tinh thần của bạn không tốt?",
        "physical": "Trong 30 ngày qua, có bao nhiêu ngày sức khỏe thể chất của bạn không tốt?",
        "walk": "Bạn có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không?",
        "education": "Trình độ học vấn cao nhất của bạn là gì?",
        "income": "Thu nhập hằng năm của hộ gia đình bạn là bao nhiêu?",
        "fruit_help": "Số khẩu phần trung bình mỗi ngày (0–5)",
        "veg_help": "Số khẩu phần trung bình mỗi ngày (0–5)",
    },
    "dimensions": {"Age group": "Nhóm tuổi", "Sex group": "Giới tính",
                    "Income group": "Nhóm thu nhập", "Education group": "Trình độ học vấn"},
    "group_label": "Nhóm", "metric_label": "Chỉ số", "score_label": "Giá trị",
    "decision_rule": "Quy tắc quyết định: ngưỡng = chi phí dương tính giả / (chi phí dương tính giả + chi phí âm tính giả). Đây là các giả định tương đối, không phải chi phí tiền tệ thực tế hay lời khuyên y tế.",
    "subgroup_note": "Các chỉ số theo nhóm chỉ mang tính mô tả. Khác biệt giữa các nhóm không tự chứng minh mô hình có thiên lệch hay công bằng.",
    "data_source_text": "Bộ dữ liệu Diabetes Health Indicators, khảo sát CDC BRFSS năm 2015 (Kaggle). Dự án sử dụng bộ dữ liệu gốc có phân bố không cân bằng.",
    "rows_label": "Số bản ghi khảo sát", "original_features_label": "Chỉ báo sức khỏe ban đầu",
    "model_inputs_label": "Biến đầu vào của mô hình",
    "model_card_title": "Thẻ thông tin mô hình", "model_label": "Mô hình",
    "target_label": "Biến dự đoán", "evaluation_label": "Đánh giá",
    "fairness_label": "Đánh giá công bằng",
    "model_target_text": "Tình trạng tiểu đường do người tham gia tự khai báo (nhãn khảo sát nhị phân)",
    "fairness_detail": "Mô hình được xem xét theo nhóm tuổi, giới tính, thu nhập và học vấn. Các chỉ số theo nhóm chỉ mang tính mô tả, không tự chứng minh tính công bằng.",
    "source_detail": "Thông tin về nguồn dữ liệu và đánh giá được trình bày trong README.md và models/model_meta.json.",
    "machine_translation_note": "Một số nội dung giao diện được dịch tự động. Cách diễn đạt y tế có thể chưa chính xác. Không dùng ứng dụng giáo dục này để tự chẩn đoán hoặc quyết định điều trị.",
    "global_language_setup": "Để mở danh sách ngôn ngữ rộng hơn, hãy thêm khóa Google Cloud Translation Basic API vào Streamlit Secrets.",
    "sidebar_source_label": "CDC BRFSS 2015 · Dự án giáo dục",
    "language_api_error": "Không kết nối được Google Cloud Translation. Ứng dụng đang hiển thị các ngôn ngữ có sẵn.",
    "training_metrics_notice": "Chạy `python train.py` để tạo mô hình và các chỉ số đánh giá.",
    "metric_threshold_info": "Ngưỡng quyết định được dùng để tính các chỉ số phân loại:",
    "plot_group_axis": "Nhóm", "plot_score_axis": "Tỷ lệ",
    "feature_plain": {
        "HighBP": "Bạn cho biết đã từng được nhân viên y tế thông báo bị huyết áp cao.",
        "HighChol": "Bạn cho biết đã từng được thông báo bị cholesterol cao.",
        "CholCheck": "Việc kiểm tra cholesterol trong 5 năm qua được mô hình xem xét.",
        "BMI": "Chỉ số BMI của bạn là một trong các câu trả lời được mô hình xem xét.",
        "Smoker": "Tiền sử hút thuốc của bạn được mô hình xem xét.",
        "Stroke": "Bạn cho biết đã từng bị đột quỵ.",
        "HeartDiseaseorAttack": "Bạn cho biết có tiền sử bệnh tim hoặc nhồi máu cơ tim.",
        "PhysActivity": "Câu trả lời về hoạt động thể chất được mô hình xem xét.",
        "Fruits": "Mức tiêu thụ trái cây được mô hình xem xét.",
        "Veggies": "Mức tiêu thụ rau được mô hình xem xét.",
        "HvyAlcoholConsump": "Mức tiêu thụ đồ uống có cồn được mô hình xem xét.",
        "AnyHealthcare": "Tình trạng bảo hiểm y tế được mô hình xem xét.",
        "NoDocbcCost": "Việc không thể đi khám do chi phí được mô hình xem xét.",
        "GenHlth": "Đánh giá sức khỏe tổng quát của bạn được mô hình xem xét.",
        "MentHlth": "Số ngày sức khỏe tinh thần không tốt được mô hình xem xét.",
        "PhysHlth": "Số ngày sức khỏe thể chất không tốt được mô hình xem xét.",
        "DiffWalk": "Khó khăn khi đi bộ được mô hình xem xét.",
        "Age": "Nhóm tuổi của bạn được mô hình xem xét.",
        "Sex": "Đặc điểm giới tính (theo khảo sát) là một trong các biến đầu vào.",
        "Education": "Trình độ học vấn của bạn được mô hình xem xét.",
        "Income": "Thu nhập hộ gia đình của bạn được mô hình xem xét.",
    },
    "feature_labels": {
        "HighBP": "Huyết áp cao", "HighChol": "Cholesterol cao",
        "CholCheck": "Kiểm tra cholesterol", "BMI": "Chỉ số BMI",
        "Smoker": "Tiền sử hút thuốc", "Stroke": "Tiền sử đột quỵ",
        "HeartDiseaseorAttack": "Bệnh tim hoặc nhồi máu cơ tim",
        "PhysActivity": "Hoạt động thể chất", "Fruits": "Ăn trái cây",
        "Veggies": "Ăn rau", "HvyAlcoholConsump": "Uống nhiều đồ uống có cồn",
        "AnyHealthcare": "Bảo hiểm/chăm sóc y tế", "NoDocbcCost": "Không thể đi khám do chi phí",
        "GenHlth": "Sức khỏe tổng quát", "MentHlth": "Sức khỏe tinh thần",
        "PhysHlth": "Sức khỏe thể chất", "DiffWalk": "Khó khăn khi đi bộ",
        "Age": "Nhóm tuổi", "Sex": "Giới tính theo khảo sát",
        "Education": "Trình độ học vấn", "Income": "Thu nhập hộ gia đình",
    },
}

# ---------------------------------------------------------------------------
# Packs for other built-in languages — each one only overrides what it can
# meaningfully translate; everything else falls back to English automatically.
# This keeps the app fully functional in every listed language even without an
# API key, while API translation adds full coverage when available.
# ---------------------------------------------------------------------------
LANGUAGE_PACKS = {
    "en": BASE_EN,
    "vi": VI_PACK,
    # Tiny but honest packs for the remaining built-in languages.
    # Missing keys fall back to English so nothing is ever blank.
    "fr": {"select_lang": "Langue", "yes": "Oui", "no": "Non",
           "nav": "Navigation", "home": "Accueil", "sim": "Coûts et seuil",
           "fair": "Équité", "about": "À propos et fiche du modèle",
           "footer": "Projet éducatif — pas destiné à un usage clinique."},
    "zh": {"select_lang": "语言", "yes": "是", "no": "否",
           "nav": "导航", "home": "首页", "sim": "成本与阈值",
           "fair": "公平性", "about": "关于与模型卡",
           "footer": "教育项目 — 不用于临床。"},
    "ja": {"select_lang": "言語", "yes": "はい", "no": "いいえ",
           "nav": "ナビゲーション", "home": "ホーム", "sim": "コストと閾値",
           "fair": "公平性", "about": "概要・モデルカード",
           "footer": "教育プロジェクト — 臨床用途ではありません。"},
    "ko": {"select_lang": "언어", "yes": "예", "no": "아니요",
           "nav": "탐색", "home": "홈", "sim": "비용 및 임계값",
           "fair": "공정성", "about": "소개 및 모델 카드",
           "footer": "교육 프로젝트 — 임상용이 아닙니다."},
    "es": {"select_lang": "Idioma", "yes": "Sí", "no": "No",
           "nav": "Navegación", "home": "Inicio", "sim": "Costes y umbral",
           "fair": "Equidad", "about": "Acerca de y ficha del modelo",
           "footer": "Proyecto educativo — no destinado al uso clínico."},
    "pt": {"select_lang": "Idioma", "yes": "Sim", "no": "Não",
           "nav": "Navegação", "home": "Início", "sim": "Custos e limite",
           "fair": "Justiça", "about": "Sobre e cartão do modelo",
           "footer": "Projeto educacional — não para uso clínico."},
    "de": {"select_lang": "Sprache", "yes": "Ja", "no": "Nein",
           "nav": "Navigation", "home": "Start", "sim": "Kosten & Schwelle",
           "fair": "Fairness", "about": "Über & Modellkarte",
           "footer": "Bildungsprojekt — nicht für klinische Zwecke."},
    "it": {"select_lang": "Lingua", "yes": "Sì", "no": "No",
           "nav": "Navigazione", "home": "Home", "sim": "Costi e soglia",
           "fair": "Equità", "about": "Informazioni e scheda modello",
           "footer": "Progetto educativo — non per uso clinico."},
    "ru": {"select_lang": "Язык", "yes": "Да", "no": "Нет",
           "nav": "Навигация", "home": "Главная", "sim": "Издержки и порог",
           "fair": "Справедливость", "about": "О проекте и карточка модели",
           "footer": "Образовательный проект — не для клинического использования."},
    "ar": {"select_lang": "اللغة", "yes": "نعم", "no": "لا",
           "nav": "التنقل", "home": "الرئيسية", "sim": "التكلفة والحد",
           "fair": "الإنصاف", "about": "حول وبطاقة النموذج",
           "footer": "مشروع تعليمي — ليس للاستخدام السريري."},
    "hi": {"select_lang": "भाषा", "yes": "हाँ", "no": "नहीं",
           "nav": "नेविगेशन", "home": "होम", "sim": "लागत और सीमा",
           "fair": "निष्पक्षता", "about": "परिचय और मॉडल कार्ड",
           "footer": "शैक्षिक परियोजना — नैदानिक उपयोग के लिए नहीं।"},
    "th": {"select_lang": "ภาษา", "yes": "ใช่", "no": "ไม่ใช่",
           "nav": "การนำทาง", "home": "หน้าแรก", "sim": "ต้นทุนและเกณฑ์",
           "fair": "ความเป็นธรรม", "about": "เกี่ยวกับและบัตรโมเดล",
           "footer": "โครงการเพื่อการศึกษา — ไม่ใช้ในทางคลินิก"},
    "id": {"select_lang": "Bahasa", "yes": "Ya", "no": "Tidak",
           "nav": "Navigasi", "home": "Beranda", "sim": "Biaya & ambang",
           "fair": "Keadilan", "about": "Tentang & kartu model",
           "footer": "Proyek edukasi — bukan untuk penggunaan klinis."},
    "tr": {"select_lang": "Dil", "yes": "Evet", "no": "Hayır",
           "nav": "Gezinme", "home": "Ana sayfa", "sim": "Maliyet ve eşik",
           "fair": "Adalet", "about": "Hakkında ve model kartı",
           "footer": "Eğitim projesi — klinik kullanım için değildir."},
}

# Ensure every manual language has at least an empty pack so lookups never fail.
for _code in MANUAL_LANGUAGE_NAMES:
    LANGUAGE_PACKS.setdefault(_code, {})


def build_locale(language_code: str) -> dict:
    """Merge the language pack on top of English so no key is ever missing."""
    base = copy.deepcopy(BASE_EN)
    pack = LANGUAGE_PACKS.get(language_code, {})
    return merge_catalog(base, pack)


# ---------------------------------------------------------------------------
# Page configuration & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DiaCheck", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f6fbf8 0%,#f8fafc 48%,#f5f8f7 100%);}
.block-container {padding-top:2.0rem; padding-bottom:3rem; max-width:1200px;}
h1,h2,h3 {font-family:'Manrope',sans-serif; color:#153b32;}
.hero {background:linear-gradient(115deg,#0f513f 0%,#16745a 55%,#2b8b69 100%);padding:2.2rem 2.4rem;border-radius:24px;color:white;margin-bottom:1.25rem;box-shadow:0 14px 36px rgba(15,81,63,.14);}
.hero h1 {color:white;font-size:2.35rem;line-height:1.15;margin:0 0 .7rem 0;}
.hero p {color:#e3f5ec;font-size:1.05rem;max-width:760px;margin:0;}
.eyebrow {letter-spacing:.13em;text-transform:uppercase;font-size:.72rem;font-weight:700;color:#b8e4d1;margin-bottom:.7rem;}
.panel {background:white;border:1px solid #e4eee8;border-radius:18px;padding:1.25rem 1.4rem;box-shadow:0 5px 18px rgba(20,60,45,.035);}
.metric {background:white;border:1px solid #e4eee8;border-radius:16px;padding:1rem 1.15rem;}
.metric-label {color:#6b7f77;font-size:.82rem;font-weight:600;}
.metric-value {font-family:'Manrope';font-size:1.55rem;font-weight:800;color:#174b3b;}
.small-muted {color:#71827c;font-size:.86rem;}
div[data-testid="stForm"] {background:white;border:1px solid #e4eee8;border-radius:18px;padding:1.4rem;}
.stButton>button, .stFormSubmitButton>button {border-radius:12px;background:#17664e;color:white;border:0;font-weight:700;padding:.65rem 1.2rem;}
.stButton>button:hover, .stFormSubmitButton>button:hover {background:#104b39;color:white;border:0;}
div[data-testid="stAlert"] {border-radius:14px;}
.stMarkdown p, [data-testid="stWidgetLabel"] p {font-size:1rem; line-height:1.5;}
div[data-testid="stNumberInput"] input {font-size:1.05rem; min-height:2.7rem;}
.stFormSubmitButton>button {font-size:1.05rem; min-height:3rem;}
[data-testid="stSidebar"] {background:#eef6f1;border-right:1px solid #e1ece5;}
[data-testid="stSidebar"] .block-container {padding-top:1.3rem;}
hr {border-color:#e4eee8;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model & metadata loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "models" / "pipeline.joblib")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    fair = pd.read_csv(ROOT / "data" / "fairness_metrics.csv")
    return model, meta, fair

try:
    model, META, FAIR = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Run `python train.py` first, then restart the app.")
    st.stop()

if "last_case" not in st.session_state:
    st.session_state.last_case = None
if "last_probability" not in st.session_state:
    st.session_state.last_probability = None
if "cost_fn" not in st.session_state:
    st.session_state.cost_fn = float(META["cost_assumptions"]["false_negative"])
if "cost_fp" not in st.session_state:
    st.session_state.cost_fp = float(META["cost_assumptions"]["false_positive"])

# ---------------------------------------------------------------------------
# Translation API helpers
# ---------------------------------------------------------------------------
def _get_translation_key():
    try:
        value = st.secrets.get("GOOGLE_TRANSLATE_API_KEY", "")
    except Exception:
        value = ""
    return str(value or os.getenv("GOOGLE_TRANSLATE_API_KEY", "")).strip()


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
            fallback = merge_catalog(copy.deepcopy(BASE_EN), LANGUAGE_PACKS.get(language_code, {}))
            return fallback, False, True
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

L, _machine_translated, locale_translation_failed = _build_locale(
    st.session_state.language_code, translation_api_key, language_catalog_json
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 DiaCheck")
    st.caption(L["brand_caption"])
    st.selectbox(
        "🌐 " + L["select_lang"],
        options=language_options,
        format_func=lambda code: language_names.get(code, code),
        key="language_code",
    )
    if locale_translation_failed or language_api_error:
        st.warning(L["language_api_error"])
    elif st.session_state.language_code not in {"en"} and not translation_api_key:
        st.caption("🌐 " + L["machine_translation_note"])
    page_key = st.radio(
        L["nav"],
        options=["home", "sim", "fair", "about"],
        format_func=lambda key: L[key],
        label_visibility="collapsed",
        key="page_navigation",
    )
    st.divider()
    st.info("🎓 " + L["footer"])
    st.caption(L["sidebar_source_label"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_pct(x):
    return f"{100 * x:.1f}%"


def contribution_table(case, locale):
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["model"]
    arr = np.array([case[f] for f in META["features"]], dtype=float)
    z = (arr - scaler.mean_) / scaler.scale_
    contributions = z * lr.coef_[0]
    labels = locale.get("feature_labels", {})
    out = pd.DataFrame({
        "key": META["features"],
        "feature": [labels.get(f, f) for f in META["features"]],
        "contribution": contributions,
    })
    return out.sort_values("contribution", ascending=False)


def render_hero(eyebrow, title, description):
    hero_text, hero_characters = st.columns([4.5, 1.25], vertical_alignment="center")
    with hero_text:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{title}</h1><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
    with hero_characters:
        if CHARACTER_IMAGE.exists():
            st.image(str(CHARACTER_IMAGE), width=220)


def _reset_screening():
    keys_to_clear = [
        "age_group", "sex", "height_cm", "weight_kg", "high_bp", "high_chol",
        "chol_check", "smoking_history", "stroke_history", "heart_history",
        "physical_activity", "fruit_intake", "vegetable_intake", "alcohol_intake",
        "healthcare_coverage", "care_access_cost", "general_health_rating",
        "mental_health_days", "physical_health_days", "walking_difficulty",
        "education_level", "household_income", "last_case", "last_probability",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.rerun()


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
if page_key == "home":
    render_hero(L["home_eyebrow"], L["hero"], L["intro"])
    st.caption("🔒 " + L["privacy"])
    st.subheader(L["screen"])
    q = L["questions"]
    yes_no = [False, True]
    yes_no_label = lambda value: L["yes"] if value else L["no"]

    with st.form("screening_form"):
        st.markdown("#### " + L["basic_information"])
        c1, c2 = st.columns(2)
        with c1:
            age_labels = L["age_options"]
            age_idx = st.selectbox(L["age"], list(range(1, 14)),
                                   format_func=lambda x: age_labels[x - 1],
                                   index=6, key="age_group")
            sex_label = st.radio(L["sex"], ["female", "male"],
                                 format_func=lambda value: L[value],
                                 horizontal=True, key="sex")
            height_cm = st.number_input(q["height"], min_value=100.0, max_value=230.0,
                                        value=170.0, step=1.0, key="height_cm")
            weight_kg = st.number_input(q["weight"], min_value=25.0, max_value=300.0,
                                        value=70.0, step=0.5, key="weight_kg")
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.metric(q["bmi_auto"], f"{bmi:.1f} kg/m²")
            bp = st.radio(L["bp"], yes_no, format_func=yes_no_label,
                          horizontal=True, key="high_bp")
            chol = st.radio(L["chol"], yes_no, format_func=yes_no_label,
                            horizontal=True, key="high_chol")
            cholcheck = st.radio(q["cholcheck"], yes_no, format_func=yes_no_label,
                                 horizontal=True, key="chol_check")
            smoker = st.radio(L["smoke"], yes_no, format_func=yes_no_label,
                              horizontal=True, key="smoking_history")
            stroke = st.radio(L["stroke"], yes_no, format_func=yes_no_label,
                              horizontal=True, key="stroke_history")
            heart = st.radio(L["heart"], yes_no, format_func=yes_no_label,
                             horizontal=True, key="heart_history")
        with c2:
            activity = st.radio(L["activity"], yes_no, format_func=yes_no_label,
                                horizontal=True, key="physical_activity")
            fruits = st.radio(q["fruits"], yes_no, format_func=yes_no_label,
                              horizontal=True, key="fruit_intake")
            veggies = st.radio(q["veggies"], yes_no, format_func=yes_no_label,
                               horizontal=True, key="vegetable_intake")
            heavy_alcohol = st.radio(q["alcohol"], yes_no, format_func=yes_no_label,
                                     horizontal=True, key="alcohol_intake")
            healthcare = st.radio(q["healthcare"], yes_no, format_func=yes_no_label,
                                  horizontal=True, key="healthcare_coverage")
            no_doc_cost = st.radio(q["cost"], yes_no, format_func=yes_no_label,
                                   horizontal=True, key="care_access_cost")
            health_labels = [L["excellent"], L["very_good"], L["good"], L["fair"], L["poor"]]
            health_idx = st.select_slider(
                L["general_health"], options=[1, 2, 3, 4, 5],
                format_func=lambda x: health_labels[x - 1], value=3,
                key="general_health_rating",
            )
            mental_days = st.slider(q["mental"], 0, 30, 0, key="mental_health_days")
            physical_days = st.slider(q["physical"], 0, 30, 0, key="physical_health_days")
            diff_walk = st.radio(q["walk"], yes_no, format_func=yes_no_label,
                                 horizontal=True, key="walking_difficulty")
            education_labels = L["education_levels"]
            education = st.selectbox(q["education"], options=list(range(1, 7)),
                                     format_func=lambda x: education_labels[x - 1],
                                     index=5, key="education_level")
            income_labels = L["income_levels"]
            income = st.selectbox(q["income"], options=list(range(1, 9)),
                                  format_func=lambda x: income_labels[x - 1],
                                  index=7, key="household_income")
        submitted = st.form_submit_button(L["submit"], use_container_width=True,
                                          key="submit_screening")

    if submitted:
        if not 12 <= bmi <= 98:
            st.error(L.get("error", "Please check your height and weight."))
            st.stop()
        case = {
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
        st.session_state.last_case = case
        st.session_state.last_probability = float(
            model.predict_proba(pd.DataFrame([case], columns=META["features"]))[0, 1]
        )

    if st.session_state.last_probability is not None:
        probability = st.session_state.last_probability
        threshold = st.session_state.cost_fp / (st.session_state.cost_fp + st.session_state.cost_fn)
        st.markdown("---")
        st.subheader(L["result"])
        a, b = st.columns([1, 1.5])
        with a:
            st.metric(L["estimate"], f"{probability * 100:.1f}/100")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(f"{L['threshold']}: {threshold:.3f} · {fmt_pct(threshold)}")
            st.caption(L["score_note"])
            st.caption(L["result_score_explain"])
        with b:
            if probability >= threshold:
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

# ---------------------------------------------------------------------------
# Cost & threshold simulator
# ---------------------------------------------------------------------------
elif page_key == "sim":
    render_hero(L["sim_eyebrow"], L["cost_title"], L["cost_desc"])
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.cost_fn = st.number_input(
            L["fn"], min_value=0.1, max_value=100.0,
            value=float(st.session_state.cost_fn), step=0.5, key="cost_fn_input")
    with col2:
        st.session_state.cost_fp = st.number_input(
            L["fp"], min_value=0.1, max_value=100.0,
            value=float(st.session_state.cost_fp), step=0.5, key="cost_fp_input")
    threshold = st.session_state.cost_fp / (st.session_state.cost_fp + st.session_state.cost_fn)
    st.markdown(
        f'<div class="panel"><div class="metric-label">{L["threshold"]}</div>'
        f'<div class="metric-value" style="font-size:2.4rem">{threshold:.3f} · {fmt_pct(threshold)}</div>'
        f'<p class="small-muted">{L["example"]}</p></div>',
        unsafe_allow_html=True,
    )
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

# ---------------------------------------------------------------------------
# Fairness page
# ---------------------------------------------------------------------------
elif page_key == "fair":
    render_hero(L["fair_eyebrow"], L["fair_title"], L["fair_desc"])
    dimension_options = ["Age group", "Sex group", "Income group", "Education group"]
    dimension = st.selectbox(L["dimension"], dimension_options,
                             format_func=lambda key: L["dimensions"].get(key, key),
                             key="fairness_dimension")
    sub = FAIR[FAIR.dimension == dimension].copy()

    def _localize_group(row):
        raw = str(row["group"])
        if row["dimension"] == "Sex group":
            if raw.lower() == "female":
                return L["female"]
            if raw.lower() == "male":
                return L["male"]
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
        sub["group"] = sub.apply(_localize_group, axis=1)
    rename = {
        "group": L["group_label"], "n": L["n"], "sensitivity": L["sens"],
        "specificity": L["spec"], "precision": L["prec"],
        "positive_rate": L["pos"], "predicted_positive_rate": L["predpos"],
    }
    show = sub[["group", "n", "sensitivity", "specificity", "precision",
                "positive_rate", "predicted_positive_rate"]].rename(columns=rename)
    for c in [L["sens"], L["spec"], L["prec"], L["pos"], L["predpos"]]:
        show[c] = show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(show, use_container_width=True, hide_index=True)
    plot = sub.melt(id_vars=["group"],
                    value_vars=["sensitivity", "specificity"],
                    var_name="Metric", value_name="Score")
    plot["Metric"] = plot["Metric"].map({"sensitivity": L["sens"], "specificity": L["spec"]})
    fig = px.bar(plot, x="group", y="Score", color="Metric", barmode="group",
                 range_y=[0, 1],
                 labels={"group": L["plot_group_axis"], "Score": L["plot_score_axis"]})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(L["subgroup_note"])

# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------
else:
    render_hero(L["about_eyebrow"], L["about_title"], L["intended_text"])
    st.subheader(L["intended"])
    st.write(L["intended_text"])
    st.subheader(L["data"])
    st.write(L["data_source_text"])
    data_cols = st.columns(3)
    data_values = [
        (L["rows_label"], f"{253680:,}"),
        (L["original_features_label"], "21"),
        (L["model_inputs_label"], str(len(META["features"]))),
    ]
    for col, (label, value) in zip(data_cols, data_values):
        with col:
            st.metric(label, value)
    st.subheader(L["metrics"])
    m = META.get("metrics") or {}
    if not m or m.get("roc_auc") is None:
        st.info(L["training_metrics_notice"])
        m = None
    if m:
        metric_cols = st.columns(4)
        cards = [
            (L["metric_auc"], m["roc_auc"]),
            (L["metric_ap"], m["average_precision"]),
            (L["metric_brier"], m["brier_score"]),
            (L["metric_recall"], m["recall_sensitivity"]),
            (L["metric_acc"], m["accuracy"]),
            (L["metric_precision"], m["precision"]),
            (L["metric_f1"], m["f1"]),
            (L["metric_spec"], m["specificity"]),
        ]
        for i, (label, value) in enumerate(cards):
            with metric_cols[i % 4]:
                st.markdown(
                    f'<div class="metric"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value:.3f}</div></div>',
                    unsafe_allow_html=True,
                )
        st.write(L["split_text"])
        st.caption(
            f"{L['metric_threshold_info']} {m['threshold']:.3f} · "
            f"{L['fn']}: {META['cost_assumptions']['false_negative']:.1f}; "
            f"{L['fp']}: {META['cost_assumptions']['false_positive']:.1f}."
        )
    st.subheader(L["limitations"])
    for item in L["limitations_list"]:
        st.markdown(f"- {item}")
    st.subheader(L["disclaimer"])
    st.warning(L["disclaimer_text"])
    st.subheader(L["model_card_title"])
    st.markdown(
        f"**{L['model_label']}:** {META['model']}  \n"
        f"**{L['target_label']}:** {L['model_target_text']}  \n"
        f"**{L['evaluation_label']}:** {L['split_text']}  \n"
        f"**{L['fairness_label']}:** {L['fairness_detail']}"
    )
    st.caption(L["source_detail"])

st.markdown("---")
st.caption("DiaCheck · " + L["footer"])
