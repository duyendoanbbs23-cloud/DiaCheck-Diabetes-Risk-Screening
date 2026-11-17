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
MANUAL_LANGUAGE_NAMES = {"en":"English", "vi":"Tiếng Việt", "fr":"Français", "zh":"中文（简体）", "ja":"日本語", "ko":"한국어", "es":"Español"}

T = {
"en": {
"tagline":"A simple first step toward understanding your health.",
"nav":"Navigation","home":"Home","sim":"Cost & threshold","fair":"Fairness","about":"About & model card",
"hero":"Know your risk. Take the next step with confidence.",
"intro":"Answer a few questions about your health and receive an educational estimate to help you consider whether to ask for a blood sugar test.",
"privacy":"Your answers are processed in this app session and are not saved by DiaCheck.",
"screen":"Start your screening","age":"Age group","sex":"Sex (as recorded in the source survey)","female":"Female","male":"Male",
"bp":"Have you ever been told you have high blood pressure?","chol":"Have you ever been told you have high cholesterol?","bmi":"What is your BMI?","smoke":"Have you smoked at least 100 cigarettes in your lifetime?","stroke":"Have you ever had a stroke?","heart":"Have you had coronary heart disease or a heart attack?","activity":"In the past 30 days, other than your regular job, did you participate in any physical activities?","health":"In general, would you say your health is:","yes":"Yes","no":"No","submit":"Estimate my screening result","result":"Your screening result","estimate":"Illustrative model score","above":"This estimate is above the current decision threshold.","below":"This estimate is below the current decision threshold.","action_high":"Consider arranging a blood sugar test and discussing your risk with a qualified healthcare professional.","action_low":"This result alone does not indicate that a test is unnecessary. If you have symptoms, concerns, or risk factors, contact a healthcare professional.","not_diag":"This is not a diagnosis. Only an appropriate clinical assessment and testing can determine diabetes status.",
"cost_title":"Explore the decision threshold","cost_desc":"Change the relative cost assigned to each type of error. The threshold is a decision-analysis setting, not a medical cut-off.",
"fn":"Cost of a false negative","fp":"Cost of a false positive","threshold":"Resulting decision threshold","example":"The app recommends considering a test when the estimated likelihood is at or above this threshold.",
"explain":"What influenced this estimate?","up":"Factors pushing the estimate up","down":"Factors pushing the estimate down","contrib":"Relative contribution to this model's score (not causal effects).",
"fair_title":"Fairness and subgroup performance","fair_desc":"Held-out test-set metrics by demographic and socioeconomic groups. This full-questionnaire model also uses education and income as predictors.",
"dimension":"Compare by","n":"Sample size","sens":"Sensitivity","spec":"Specificity","prec":"Precision","pos":"Observed positive share","predpos":"Predicted positive share",
"about_title":"About DiaCheck","data":"Data source","metrics":"Held-out test metrics","limitations":"Important limitations","disclaimer":"Medical disclaimer","intended":"Intended use",
"limitations_list":["The source is a 2015 US telephone survey; it may not represent today's population or people in other countries.","All health indicators are self-reported and are not verified clinical measurements.","The model predicts a survey label, not a laboratory-confirmed diagnosis.","The selected features are a short subset of the original indicators; performance may differ from models using all features.","Performance and error rates can differ across population groups; subgroup estimates may be affected by sample size."],
"disclaimer_text":"DiaCheck is an educational student project, not a medical device and not for clinical use. It cannot diagnose, rule out, or treat diabetes. Do not delay care or make health decisions solely from this result. Seek professional advice and appropriate blood glucose testing.",
"intended_text":"Adults answering questions at home who want an educational prompt to consider whether to seek a blood sugar test.",
"footer":"Educational project — not for clinical use.","select_lang":"Language","general_health":"General health","excellent":"Excellent","very_good":"Very good","good":"Good","fair":"Fair","poor":"Poor","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],
"error":"Please enter a BMI between 12 and 98.","start":"Please complete the screening form to see your estimate.","no_result":"Complete the form on the Home page to see a case-specific explanation.","metric_auc":"ROC-AUC","metric_ap":"Average precision","metric_brier":"Brier score","metric_recall":"Sensitivity","metric_acc":"Accuracy","metric_precision":"Precision","metric_f1":"F1 score","metric_spec":"Specificity","split":"Evaluation design","split_text":"Stratified 80/20 train/test split; random state 42. Metrics below are test-set estimates, not external validation.","score_note":"This percentage is a score produced by a model trained on survey data. It is not your personal medical probability and cannot confirm or rule out diabetes.","factor_more":"Answers linked with a higher model score","factor_less":"Answers linked with a lower model score","next_steps_text":"If you are concerned, have symptoms, or want to know your blood sugar level, contact a healthcare professional to discuss an appropriate test. Do not rely on this score alone.","feature_plain":{"HighBP":"You reported that a health professional had told you that you have high blood pressure.","HighChol":"You reported that a health professional had told you that you have high cholesterol.","BMI":"Your BMI is one of the answers considered by the model.","Smoker":"Your smoking-history answer is considered by the model.","Stroke":"You reported having had a stroke.","HeartDiseaseorAttack":"You reported a history of heart disease or heart attack.","PhysActivity":"Your physical-activity answer is considered by the model.","GenHlth":"Your rating of your general health is considered by the model.","Age":"Your age group is considered by the model.","Sex":"This survey characteristic is one of the model inputs."}
},
"vi": {
"tagline":"Một bước đơn giản để hiểu thêm về sức khỏe của bạn.",
"nav":"Điều hướng","home":"Trang chủ","sim":"Chi phí & ngưỡng","fair":"Công bằng mô hình","about":"Giới thiệu & Model Card",
"hero":"Hiểu nguy cơ. Chủ động bước tiếp theo.",
"intro":"Trả lời một số câu hỏi sức khỏe để nhận ước tính mang tính giáo dục, hỗ trợ bạn cân nhắc việc trao đổi với nhân viên y tế về xét nghiệm đường huyết.",
"privacy":"Câu trả lời chỉ được xử lý trong phiên ứng dụng và DiaCheck không lưu lại.",
"screen":"Bắt đầu sàng lọc","age":"Nhóm tuổi","sex":"Giới tính (theo cách mã hóa của khảo sát nguồn)","female":"Nữ","male":"Nam",
"bp":"Bạn đã từng được thông báo bị huyết áp cao chưa?","chol":"Bạn đã từng được thông báo bị cholesterol cao chưa?","bmi":"Chỉ số BMI của bạn là bao nhiêu?","smoke":"Bạn đã từng hút ít nhất 100 điếu thuốc trong đời chưa?","stroke":"Bạn đã từng bị đột quỵ chưa?","heart":"Bạn đã từng mắc bệnh mạch vành hoặc nhồi máu cơ tim chưa?","activity":"Trong 30 ngày qua, ngoài công việc thường ngày, bạn có tham gia hoạt động thể chất nào không?","health":"Nhìn chung, bạn tự đánh giá sức khỏe của mình như thế nào?","yes":"Có","no":"Không","submit":"Ước tính kết quả sàng lọc","result":"Kết quả đánh giá sức khỏe","estimate":"Điểm tham khảo (trên thang 100)","above":"Theo cách tính của DiaCheck, điểm của bạn đang cao hơn mức tham khảo.","below":"Theo cách tính của DiaCheck, điểm của bạn đang thấp hơn mức tham khảo.","action_high":"Bạn nên đến cơ sở y tế để được tư vấn và hỏi bác sĩ về việc kiểm tra đường huyết.","action_low":"Điểm thấp không có nghĩa là chắc chắn không mắc bệnh hoặc không cần kiểm tra. Nếu thấy mệt, khát nước nhiều, tiểu nhiều hoặc lo lắng về sức khỏe, bạn nên hỏi ý kiến nhân viên y tế.","not_diag":"Xin lưu ý: Đây chỉ là kết quả tham khảo từ câu trả lời của bạn, không phải kết luận mắc hay không mắc bệnh. Muốn biết chính xác, bạn cần được nhân viên y tế thăm khám và làm xét nghiệm phù hợp.",
"cost_title":"Khám phá ngưỡng quyết định","cost_desc":"Thay đổi chi phí tương đối của từng loại sai số. Ngưỡng này phục vụ phân tích quyết định, không phải ngưỡng y khoa.",
"fn":"Chi phí âm tính giả","fp":"Chi phí dương tính giả","threshold":"Ngưỡng quyết định tương ứng","example":"Ứng dụng khuyến nghị cân nhắc xét nghiệm khi xác suất ước tính bằng hoặc cao hơn ngưỡng này.",
"explain":"Yếu tố nào ảnh hưởng đến ước tính?","up":"Yếu tố làm tăng điểm ước tính","down":"Yếu tố làm giảm điểm ước tính","contrib":"Đóng góp tương đối vào điểm của mô hình (không phải quan hệ nhân quả).",
"fair_title":"Công bằng và hiệu suất theo nhóm","fair_desc":"Chỉ số trên tập kiểm tra, phân tách theo nhóm nhân khẩu học và kinh tế-xã hội. Phiên bản dùng toàn bộ câu hỏi này cũng đưa học vấn và thu nhập vào mô hình.",
"dimension":"Phân nhóm theo","n":"Cỡ mẫu","sens":"Độ nhạy","spec":"Độ đặc hiệu","prec":"Precision","pos":"Tỷ lệ dương tính thực tế","predpos":"Tỷ lệ dự đoán dương tính",
"about_title":"Về DiaCheck","data":"Nguồn dữ liệu","metrics":"Chỉ số trên tập kiểm tra","limitations":"Các hạn chế quan trọng","disclaimer":"Tuyên bố y tế","intended":"Mục đích sử dụng",
"limitations_list":["Nguồn dữ liệu là khảo sát điện thoại tại Hoa Kỳ năm 2015; có thể không đại diện cho dân số hiện tại hoặc các quốc gia khác.","Các chỉ báo sức khỏe do người tham gia tự khai báo, không được xác minh bằng phép đo lâm sàng.","Mô hình dự đoán nhãn khảo sát, không phải chẩn đoán được xác nhận bằng xét nghiệm.","Các biến được chọn là một tập con ngắn của bộ chỉ báo gốc; hiệu suất có thể khác nếu dùng toàn bộ biến.","Hiệu suất và tỷ lệ sai số có thể khác giữa các nhóm dân số; ước tính nhóm nhỏ phụ thuộc vào cỡ mẫu."],
"disclaimer_text":"DiaCheck là dự án học thuật mang tính giáo dục, không phải thiết bị y tế và không dùng cho mục đích lâm sàng. Ứng dụng không thể chẩn đoán, loại trừ hay điều trị đái tháo đường. Không trì hoãn chăm sóc y tế hoặc quyết định sức khỏe chỉ dựa trên kết quả này. Hãy tìm tư vấn chuyên môn và xét nghiệm đường huyết phù hợp.",
"intended_text":"Người trưởng thành trả lời câu hỏi tại nhà và muốn có thông tin giáo dục để cân nhắc việc thực hiện xét nghiệm đường huyết.",
"footer":"Dự án giáo dục — không dùng cho mục đích lâm sàng.","select_lang":"Ngôn ngữ","general_health":"Sức khỏe tổng quát","excellent":"Rất tốt","very_good":"Tốt","good":"Khá","fair":"Trung bình","poor":"Kém","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],
"error":"Vui lòng kiểm tra lại chiều cao và cân nặng.","start":"Bạn vui lòng trả lời các câu hỏi để xem kết quả đánh giá sức khỏe.","no_result":"Bạn hãy hoàn thành các câu hỏi ở Trang chủ để xem kết quả.","metric_auc":"ROC-AUC","metric_ap":"Average precision","metric_brier":"Brier score","metric_recall":"Độ nhạy","metric_acc":"Độ chính xác","metric_precision":"Precision","metric_f1":"F1 score","metric_spec":"Độ đặc hiệu","split":"Thiết kế đánh giá","split_text":"Chia tập train/test phân tầng 80/20; random state 42. Các chỉ số là ước tính trên tập kiểm tra, chưa phải xác thực bên ngoài.","score_note":"Con số này là điểm do máy tính ước tính từ những câu trả lời của bạn (tính trên thang 100). Đây KHÔNG phải phần trăm khả năng bạn mắc bệnh, cũng không thể xác định bạn có mắc bệnh hay không."
},
"zh": {
"tagline":"了解健康状况的简单第一步。","nav":"导航","home":"首页","sim":"成本与阈值","fair":"公平性","about":"关于与模型卡",
"hero":"了解风险，自信地迈出下一步。","intro":"回答几个健康问题，获取教育用途的估计，帮助您考虑是否咨询专业人员并进行血糖检测。","privacy":"您的回答仅在本次应用会话中处理，DiaCheck 不会保存。","screen":"开始筛查","age":"年龄组","sex":"性别（按原始调查编码）","female":"女性","male":"男性","bp":"您是否曾被告知患有高血压？","chol":"您是否曾被告知胆固醇偏高？","bmi":"您的 BMI 是多少？","smoke":"您一生中是否至少吸过 100 支香烟？","stroke":"您是否曾经中风？","heart":"您是否患过冠心病或心脏病发作？","activity":"过去 30 天内，除日常工作外，您是否参加过体育活动？","health":"总体而言，您如何评价自己的健康状况？","yes":"是","no":"否","submit":"估算筛查结果","result":"筛查结果","estimate":"模型估计可能性","above":"该估计值高于当前决策阈值。","below":"该估计值低于当前决策阈值。","action_high":"建议考虑安排血糖检测，并与合格的医疗专业人员讨论相关风险。","action_low":"此结果并不意味着无需检测。如有症状、担忧或风险因素，请联系医疗专业人员。","not_diag":"这不是诊断。只有适当的临床评估和检测才能确定糖尿病状况。","cost_title":"探索决策阈值","cost_desc":"调整两类错误的相对成本。该阈值用于决策分析，并非医学界值。","fn":"假阴性成本","fp":"假阳性成本","threshold":"对应决策阈值","example":"当估计可能性达到或超过该阈值时，应用建议考虑检测。","explain":"哪些因素影响了估计？","up":"提高估计值的因素","down":"降低估计值的因素","contrib":"对模型得分的相对贡献（不是因果效应）。","fair_title":"公平性与分组表现","fair_desc":"按人口和社会经济群体展示留出测试集指标。收入和教育仅用于审计，不作为模型输入。","dimension":"分组维度","n":"样本量","sens":"敏感度","spec":"特异度","prec":"精确率","pos":"实际阳性比例","predpos":"预测阳性比例","about_title":"关于 DiaCheck","data":"数据来源","metrics":"留出测试集指标","limitations":"重要局限","disclaimer":"医疗免责声明","intended":"预期用途","limitations_list":["数据来自 2015 年美国电话调查，可能无法代表当前人群或其他国家的人群。","所有健康指标均为自我报告，未经临床测量验证。","模型预测的是调查标签，而非实验室确诊结果。","模型仅使用原始指标中的一小部分特征。","不同人群的表现和错误率可能不同；小组估计受样本量影响。"],"disclaimer_text":"DiaCheck 是教育用途的学生项目，不是医疗器械，不可用于临床。它不能诊断、排除或治疗糖尿病。请勿仅凭此结果延误就医或作出健康决定。请寻求专业建议并进行适当的血糖检测。","intended_text":"在家回答问题、希望获得教育信息以考虑是否进行血糖检测的成年人。","footer":"教育项目 — 不用于临床。","select_lang":"语言","general_health":"总体健康","excellent":"非常好","very_good":"很好","good":"好","fair":"一般","poor":"差","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],"error":"请输入 12 至 98 之间的 BMI。","start":"请完成筛查表以查看估计结果。","no_result":"请在首页完成表格以查看个案解释。","metric_auc":"ROC-AUC","metric_ap":"平均精确率","metric_brier":"Brier 分数","metric_recall":"敏感度","metric_acc":"准确率","metric_precision":"精确率","metric_f1":"F1 分数","metric_spec":"特异度","split":"评估设计","split_text":"分层 80/20 训练/测试划分；随机种子 42。以下指标为测试集估计，并非外部验证。"
},
"ja": {
"tagline":"健康を理解するための、シンプルな第一歩。","nav":"ナビゲーション","home":"ホーム","sim":"コストと閾値","fair":"公平性","about":"概要・モデルカード","hero":"リスクを知り、自信を持って次の一歩へ。","intro":"いくつかの健康に関する質問に回答し、血糖検査を検討するための教育目的の推定値を確認できます。","privacy":"回答はこのアプリのセッション内で処理され、DiaCheckには保存されません。","screen":"スクリーニングを開始","age":"年齢層","sex":"性別（元調査の符号化）","female":"女性","male":"男性","bp":"高血圧と診断されたことがありますか？","chol":"コレステロールが高いと指摘されたことがありますか？","bmi":"BMIはいくつですか？","smoke":"生涯で少なくとも100本のたばこを吸ったことがありますか？","stroke":"脳卒中を経験したことがありますか？","heart":"冠動脈性心疾患または心臓発作を経験したことがありますか？","activity":"過去30日間、通常の仕事以外で身体活動をしましたか？","health":"全体的な健康状態をどのように評価しますか？","yes":"はい","no":"いいえ","submit":"推定結果を表示","result":"スクリーニング結果","estimate":"モデルによる推定可能性","above":"推定値は現在の判断閾値を上回っています。","below":"推定値は現在の判断閾値を下回っています。","action_high":"血糖検査を検討し、医療専門家にリスクについて相談してください。","action_low":"この結果だけで検査が不要とは言えません。症状や不安、リスク要因がある場合は医療専門家に相談してください。","not_diag":"これは診断ではありません。糖尿病の状態は適切な臨床評価と検査によって判断されます。","cost_title":"判断閾値を確認","cost_desc":"2種類の誤りに対する相対コストを変更します。この閾値は意思決定分析用であり、医学的基準ではありません。","fn":"偽陰性のコスト","fp":"偽陽性のコスト","threshold":"計算された判断閾値","example":"推定可能性がこの閾値以上の場合、検査の検討を推奨します。","explain":"推定に影響した要因","up":"推定値を高める要因","down":"推定値を低める要因","contrib":"モデルスコアへの相対的な寄与（因果効果ではありません）。","fair_title":"公平性とグループ別性能","fair_desc":"人口統計・社会経済グループ別のテストデータ指標。所得と教育は監査のみに使用し、モデル入力には含めません。","dimension":"比較項目","n":"サンプル数","sens":"感度","spec":"特異度","prec":"適合率","pos":"実際の陽性割合","predpos":"予測陽性割合","about_title":"DiaCheckについて","data":"データソース","metrics":"テストデータ指標","limitations":"主な限界","disclaimer":"医療に関する免責事項","intended":"想定用途","limitations_list":["2015年の米国電話調査データであり、現在や他国の人口を代表しない可能性があります。","健康指標はすべて自己申告で、臨床測定による確認はありません。","モデルは調査上のラベルを予測するもので、検査で確定した診断ではありません。","元データの指標から一部の特徴量のみを使用しています。","集団によって性能や誤り率が異なる場合があり、小規模グループの推定はサンプル数に左右されます。"],"disclaimer_text":"DiaCheckは教育目的の学生プロジェクトであり、医療機器ではなく、臨床用途ではありません。糖尿病の診断、除外、治療はできません。この結果だけを根拠に受診を遅らせたり、健康上の判断をしたりしないでください。専門家に相談し、適切な血糖検査を受けてください。","intended_text":"自宅で質問に回答し、血糖検査を受けるか検討するための教育情報を求める成人。","footer":"教育プロジェクト — 臨床用途ではありません。","select_lang":"言語","general_health":"全般的な健康状態","excellent":"非常に良い","very_good":"とても良い","good":"良い","fair":"普通","poor":"悪い","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],"error":"BMIは12〜98の範囲で入力してください。","start":"推定結果を表示するにはフォームを入力してください。","no_result":"ホームでフォームを完了すると個別の説明が表示されます。","metric_auc":"ROC-AUC","metric_ap":"平均適合率","metric_brier":"Brierスコア","metric_recall":"感度","metric_acc":"正解率","metric_precision":"適合率","metric_f1":"F1スコア","metric_spec":"特異度","split":"評価設計","split_text":"層化80/20の訓練・テスト分割、random state 42。以下はテストデータの推定値であり、外部検証ではありません。"
},
"ko": {
"tagline":"건강을 이해하기 위한 간단한 첫걸음.","nav":"탐색","home":"홈","sim":"비용 및 임계값","fair":"공정성","about":"소개 및 모델 카드","hero":"위험을 이해하고 다음 단계를 준비하세요.","intro":"몇 가지 건강 질문에 답하고 혈당 검사를 고려하는 데 도움이 되는 교육용 추정치를 확인하세요.","privacy":"답변은 앱 세션 내에서만 처리되며 DiaCheck에 저장되지 않습니다.","screen":"검사 시작","age":"연령대","sex":"성별 (원본 설문 코딩)","female":"여성","male":"남성","bp":"고혈압 진단을 받은 적이 있습니까?","chol":"콜레스테롤이 높다고 들은 적이 있습니까?","bmi":"BMI는 얼마입니까?","smoke":"평생 담배를 100개비 이상 피운 적이 있습니까?","stroke":"뇌졸중을 겪은 적이 있습니까?","heart":"관상동맥 심장질환 또는 심장마비를 겪은 적이 있습니까?","activity":"지난 30일 동안 일상 업무 외에 신체 활동을 했습니까?","health":"전반적인 건강 상태를 어떻게 평가하십니까?","yes":"예","no":"아니요","submit":"검사 결과 추정","result":"검사 결과","estimate":"모델 추정 가능성","above":"이 추정치는 현재 의사결정 임계값보다 높습니다.","below":"이 추정치는 현재 의사결정 임계값보다 낮습니다.","action_high":"혈당 검사를 고려하고 의료 전문가와 위험에 대해 상담하세요.","action_low":"이 결과만으로 검사가 불필요하다는 뜻은 아닙니다. 증상이나 우려, 위험 요인이 있다면 의료 전문가에게 문의하세요.","not_diag":"진단이 아닙니다. 당뇨병 여부는 적절한 임상 평가와 검사로만 확인할 수 있습니다.","cost_title":"의사결정 임계값 살펴보기","cost_desc":"두 가지 오류의 상대적 비용을 변경합니다. 이 임계값은 의사결정 분석용이며 의학적 기준이 아닙니다.","fn":"위음성 비용","fp":"위양성 비용","threshold":"계산된 의사결정 임계값","example":"추정 가능성이 이 임계값 이상이면 검사를 고려하도록 권고합니다.","explain":"추정에 영향을 준 요인","up":"추정치를 높이는 요인","down":"추정치를 낮추는 요인","contrib":"모델 점수에 대한 상대적 기여(인과 효과가 아님).","fair_title":"공정성 및 그룹별 성능","fair_desc":"인구통계 및 사회경제 그룹별 테스트 세트 지표입니다. 소득과 교육은 감사에만 사용되며 모델 입력값이 아닙니다.","dimension":"비교 기준","n":"표본 수","sens":"민감도","spec":"특이도","prec":"정밀도","pos":"실제 양성 비율","predpos":"예측 양성 비율","about_title":"DiaCheck 소개","data":"데이터 출처","metrics":"테스트 세트 지표","limitations":"주요 한계","disclaimer":"의료 면책 조항","intended":"사용 목적","limitations_list":["2015년 미국 전화 설문 자료로 현재 인구나 다른 국가를 대표하지 않을 수 있습니다.","건강 지표는 모두 자기보고이며 임상 측정으로 확인되지 않았습니다.","모델은 설문 라벨을 예측하며 검사로 확정된 진단이 아닙니다.","원본 지표 중 일부 특성만 사용합니다.","집단별 성능과 오류율이 다를 수 있으며 소규모 그룹 추정은 표본 크기의 영향을 받습니다."],"disclaimer_text":"DiaCheck은 교육용 학생 프로젝트이며 의료기기가 아니고 임상 목적으로 사용할 수 없습니다. 당뇨병을 진단하거나 배제 또는 치료할 수 없습니다. 이 결과만으로 진료를 미루거나 건강 결정을 내리지 마세요. 전문가의 조언과 적절한 혈당 검사를 받으세요.","intended_text":"집에서 질문에 답하고 혈당 검사를 고려하기 위한 교육 정보를 원하는 성인.","footer":"교육 프로젝트 — 임상용이 아닙니다.","select_lang":"언어","general_health":"전반적인 건강","excellent":"매우 좋음","very_good":"좋음","good":"보통","fair":"나쁨","poor":"매우 나쁨","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],"error":"BMI는 12~98 사이로 입력하세요.","start":"추정 결과를 보려면 양식을 완료하세요.","no_result":"홈 페이지에서 양식을 완료하면 사례별 설명을 볼 수 있습니다.","metric_auc":"ROC-AUC","metric_ap":"평균 정밀도","metric_brier":"Brier 점수","metric_recall":"민감도","metric_acc":"정확도","metric_precision":"정밀도","metric_f1":"F1 점수","metric_spec":"특이도","split":"평가 설계","split_text":"계층화 80/20 학습/테스트 분할, random state 42. 아래 지표는 테스트 세트 추정치이며 외부 검증이 아닙니다."
},
"es": {
"tagline":"Un primer paso sencillo para comprender tu salud.","nav":"Navegación","home":"Inicio","sim":"Costes y umbral","fair":"Equidad","about":"Acerca de y ficha del modelo","hero":"Conoce tu riesgo. Da el siguiente paso con confianza.","intro":"Responde algunas preguntas de salud y recibe una estimación educativa para ayudarte a considerar si debes consultar sobre una prueba de glucosa.","privacy":"Tus respuestas se procesan durante esta sesión y DiaCheck no las guarda.","screen":"Iniciar evaluación","age":"Grupo de edad","sex":"Sexo (según la codificación de la encuesta)","female":"Mujer","male":"Hombre","bp":"¿Alguna vez te han dicho que tienes presión arterial alta?","chol":"¿Alguna vez te han dicho que tienes colesterol alto?","bmi":"¿Cuál es tu IMC?","smoke":"¿Has fumado al menos 100 cigarrillos en toda tu vida?","stroke":"¿Has sufrido un accidente cerebrovascular?","heart":"¿Has tenido enfermedad coronaria o un ataque al corazón?","activity":"En los últimos 30 días, aparte de tu trabajo habitual, ¿realizaste actividad física?","health":"En general, ¿cómo calificarías tu salud?","yes":"Sí","no":"No","submit":"Estimar mi resultado","result":"Resultado de la evaluación","estimate":"Probabilidad estimada por el modelo","above":"Esta estimación supera el umbral de decisión actual.","below":"Esta estimación está por debajo del umbral de decisión actual.","action_high":"Considera programar una prueba de glucosa y hablar sobre tu riesgo con un profesional sanitario cualificado.","action_low":"Este resultado por sí solo no significa que no necesites una prueba. Si tienes síntomas, dudas o factores de riesgo, consulta a un profesional sanitario.","not_diag":"Esto no es un diagnóstico. Solo una evaluación clínica y pruebas adecuadas pueden determinar si tienes diabetes.","cost_title":"Explorar el umbral de decisión","cost_desc":"Cambia el coste relativo de cada tipo de error. El umbral es una herramienta de análisis de decisiones, no un límite médico.","fn":"Coste de un falso negativo","fp":"Coste de un falso positivo","threshold":"Umbral de decisión resultante","example":"La aplicación recomienda considerar una prueba cuando la probabilidad estimada alcanza o supera este umbral.","explain":"¿Qué influyó en esta estimación?","up":"Factores que aumentan la estimación","down":"Factores que reducen la estimación","contrib":"Contribución relativa a la puntuación del modelo (no son efectos causales).","fair_title":"Equidad y rendimiento por grupos","fair_desc":"Métricas del conjunto de prueba por grupos demográficos y socioeconómicos. Ingresos y educación se usan solo para auditoría, no como entradas del modelo.","dimension":"Comparar por","n":"Tamaño de muestra","sens":"Sensibilidad","spec":"Especificidad","prec":"Precisión","pos":"Proporción positiva observada","predpos":"Proporción positiva predicha","about_title":"Acerca de DiaCheck","data":"Fuente de datos","metrics":"Métricas del conjunto de prueba","limitations":"Limitaciones importantes","disclaimer":"Aviso médico","intended":"Uso previsto","limitations_list":["La fuente es una encuesta telefónica de EE. UU. de 2015; puede no representar a la población actual ni a otros países.","Todos los indicadores son autodeclarados y no se verifican con mediciones clínicas.","El modelo predice una etiqueta de encuesta, no un diagnóstico confirmado por laboratorio.","Se utiliza un subconjunto reducido de las variables originales.","El rendimiento y los errores pueden variar entre grupos; las estimaciones de grupos pequeños dependen del tamaño de muestra."],"disclaimer_text":"DiaCheck es un proyecto estudiantil educativo, no un dispositivo médico y no está destinado al uso clínico. No puede diagnosticar, descartar ni tratar la diabetes. No retrases la atención ni tomes decisiones de salud únicamente con este resultado. Busca asesoramiento profesional y una prueba de glucosa adecuada.","intended_text":"Adultos que responden preguntas en casa y desean información educativa para considerar si deben hacerse una prueba de glucosa.","footer":"Proyecto educativo — no destinado al uso clínico.","select_lang":"Idioma","general_health":"Salud general","excellent":"Excelente","very_good":"Muy buena","good":"Buena","fair":"Regular","poor":"Mala","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],"error":"Introduce un IMC entre 12 y 98.","start":"Completa el formulario para ver tu estimación.","no_result":"Completa el formulario en Inicio para ver una explicación personalizada.","metric_auc":"ROC-AUC","metric_ap":"Precisión promedio","metric_brier":"Puntuación de Brier","metric_recall":"Sensibilidad","metric_acc":"Exactitud","metric_precision":"Precisión","metric_f1":"Puntuación F1","metric_spec":"Especificidad","split":"Diseño de evaluación","split_text":"División estratificada 80/20 de entrenamiento/prueba; random state 42. Las métricas son estimaciones del conjunto de prueba, no validación externa."
}
}

T["fr"] = {**T["en"], **{
"tagline":"Un premier pas simple pour mieux comprendre votre santé.","nav":"Navigation","home":"Accueil","sim":"Coûts et seuil","fair":"Équité","about":"À propos et fiche du modèle",
"hero":"Comprenez votre résultat. Choisissez la prochaine étape.","intro":"Répondez à quelques questions sur votre santé. Cette estimation éducative peut vous aider à discuter d’un éventuel test de glycémie avec un professionnel de santé.","privacy":"Vos réponses sont traitées pendant cette session et ne sont pas enregistrées par DiaCheck.",
"screen":"Commencer le questionnaire","age":"Tranche d’âge","sex":"Sexe (selon le codage de l’enquête source)","female":"Femme","male":"Homme","bp":"Un professionnel de santé vous a-t-il déjà dit que votre tension artérielle était élevée ?","chol":"Un professionnel de santé vous a-t-il déjà dit que votre cholestérol était élevé ?","bmi":"Quel est votre indice de masse corporelle (IMC) ?","smoke":"Avez-vous fumé au moins 100 cigarettes au cours de votre vie ?","stroke":"Avez-vous déjà eu un accident vasculaire cérébral (AVC) ?","heart":"Avez-vous déjà eu une maladie coronarienne ou une crise cardiaque ?","activity":"Au cours des 30 derniers jours, en dehors de votre travail habituel, avez-vous pratiqué une activité physique ?","health":"Dans l’ensemble, comment évaluez-vous votre santé ?","yes":"Oui","no":"Non","submit":"Afficher mon résultat indicatif",
"result":"Votre résultat de dépistage","estimate":"Score indicatif du modèle","above":"Vos réponses donnent un score supérieur au seuil de décision choisi pour cet outil.","below":"Vos réponses donnent un score inférieur au seuil de décision choisi pour cet outil.","action_high":"Parlez-en à un professionnel de santé et demandez-lui si un test de glycémie est adapté à votre situation.","action_low":"Un score inférieur au seuil ne garantit pas l’absence de diabète et ne signifie pas qu’un test est inutile. En cas de symptômes ou d’inquiétude, demandez conseil à un professionnel de santé.","not_diag":"Ce résultat n’est pas un diagnostic. Seul un professionnel de santé, avec les examens appropriés, peut déterminer votre état de santé.",
"cost_title":"Comprendre le seuil de décision","cost_desc":"Modifiez le coût relatif attribué aux deux types d’erreurs. Ce seuil illustre une règle statistique ; ce n’est pas un seuil médical.","fn":"Coût d’un faux négatif","fp":"Coût d’un faux positif","threshold":"Seuil de décision calculé","example":"Dans cette simulation, le résultat est signalé lorsque le score estimé atteint ou dépasse ce seuil.","explain":"Quels éléments ont influencé le score ?","up":"Éléments qui augmentent le score du modèle","down":"Éléments qui diminuent le score du modèle","contrib":"Ces éléments montrent comment vos réponses influencent le score du modèle. Ils ne prouvent pas une cause médicale.",
"fair_title":"Équité et résultats par groupe","fair_desc":"Indicateurs calculés sur le jeu de test, par groupe démographique et socio-économique. Le revenu et le niveau d’études servent uniquement à l’audit, pas aux prédictions.","dimension":"Comparer selon","n":"Taille de l’échantillon","sens":"Sensibilité","spec":"Spécificité","prec":"Précision","pos":"Part positive observée","predpos":"Part positive prédite","about_title":"À propos de DiaCheck","data":"Source des données","metrics":"Indicateurs sur le jeu de test","limitations":"Limites importantes","disclaimer":"Avertissement médical","intended":"Utilisation prévue",
"limitations_list":["Les données proviennent d’une enquête téléphonique menée aux États-Unis en 2015 et ne représentent pas nécessairement la population actuelle ni celle d’autres pays.","Les indicateurs de santé sont autodéclarés et ne sont pas confirmés par des mesures cliniques.","Le modèle prédit une réponse à une enquête, et non un diagnostic confirmé par laboratoire.","Le modèle utilise seulement une partie des variables du jeu de données initial.","Les performances et les erreurs peuvent varier selon les groupes ; les estimations des petits groupes sont moins stables."],
"disclaimer_text":"DiaCheck est un projet étudiant à but éducatif. Ce n’est pas un dispositif médical et il ne doit pas être utilisé pour prendre une décision clinique. Il ne peut ni diagnostiquer, ni exclure, ni traiter le diabète. Ne retardez pas une consultation sur la base de ce résultat. Demandez conseil à un professionnel de santé et faites les examens appropriés.","intended_text":"Adultes répondant à des questions à domicile et souhaitant obtenir une information éducative pour réfléchir à un éventuel test de glycémie.","footer":"Projet éducatif — pas destiné à un usage clinique.","select_lang":"Langue","general_health":"État de santé général","excellent":"Excellent","very_good":"Très bon","good":"Bon","fair":"Moyen","poor":"Mauvais","age_options":["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],
"error":"Veuillez saisir un IMC compris entre 12 et 98.","start":"Veuillez remplir le questionnaire pour afficher votre estimation.","no_result":"Remplissez le questionnaire sur la page Accueil pour obtenir une explication de votre résultat.","metric_auc":"ROC-AUC","metric_ap":"Précision moyenne","metric_brier":"Score de Brier","metric_recall":"Sensibilité","metric_acc":"Exactitude","metric_precision":"Valeur prédictive positive","metric_f1":"Score F1","metric_spec":"Spécificité","split":"Méthode d’évaluation","split_text":"Séparation stratifiée 80/20 entre entraînement et test ; random_state 42. Ces indicateurs sont calculés sur le jeu de test et ne constituent pas une validation externe.",
"score_note":"Ce pourcentage est un score produit par un modèle entraîné sur des données d’enquête. Il ne représente pas votre probabilité médicale personnelle et ne confirme ni n’exclut un diabète.","next_steps_text":"Si vous êtes inquiet, si vous avez des symptômes ou si vous souhaitez connaître votre glycémie, contactez un professionnel de santé pour discuter d’un test adapté. Ne vous fiez pas uniquement à ce score.","factor_more":"Réponses associées à un score plus élevé","factor_less":"Réponses associées à un score plus faible",
"feature_labels":{"HighBP":"Tension artérielle élevée","HighChol":"Cholestérol élevé","BMI":"Indice de masse corporelle (IMC)","Smoker":"Tabagisme au cours de la vie","Stroke":"Antécédent d’AVC","HeartDiseaseorAttack":"Maladie cardiaque ou crise cardiaque","PhysActivity":"Activité physique","GenHlth":"État de santé général","Age":"Tranche d’âge","Sex":"Sexe (codage de l’enquête)"},
"feature_plain":{"HighBP":"Vous avez indiqué avoir déjà reçu un diagnostic d’hypertension.","HighChol":"Vous avez indiqué avoir déjà reçu un diagnostic de cholestérol élevé.","BMI":"Votre IMC est pris en compte dans le score du modèle.","Smoker":"Votre réponse concernant le tabagisme est prise en compte dans le score.","Stroke":"Vous avez indiqué avoir déjà eu un AVC.","HeartDiseaseorAttack":"Vous avez indiqué avoir déjà eu une maladie cardiaque ou une crise cardiaque.","PhysActivity":"Votre réponse sur l’activité physique est prise en compte.","GenHlth":"Votre appréciation de votre état de santé général est prise en compte.","Age":"Votre tranche d’âge est prise en compte dans le score.","Sex":"Cette caractéristique figure parmi les variables de l’enquête utilisées par le modèle."}
}}

# Shared interface catalog: every visible label is routed through this catalog so
# changing languages updates the complete app, not only the main headings.
EXTRA_CATALOG = {
    "en": {
        "brand_caption": "DIABETES HEALTH CHECK",
        "home_eyebrow": "A FRIENDLY HEALTH CHECK",
        "sim_eyebrow": "DECISION ANALYSIS",
        "fair_eyebrow": "MODEL REVIEW",
        "about_eyebrow": "ABOUT THIS PROJECT",
        "basic_information": "Basic information",
        "education_levels": ["Never attended school / kindergarten", "Grades 1–8", "Grades 9–11", "Grade 12 / GED", "College 1–3 years", "College 4+ years"],
        "income_levels": ["Below $10,000", "$10,000–14,999", "$15,000–19,999", "$20,000–24,999", "$25,000–34,999", "$35,000–49,999", "$50,000–74,999", "$75,000 or more"],
        "questions": {
            "height": "Height (cm)", "weight": "Weight (kg)", "bmi_auto": "BMI is calculated automatically",
            "cholcheck": "Have you had your cholesterol checked in the past 5 years?",
            "fruits": "Do you eat fruit at least once a day?", "veggies": "Do you eat vegetables at least once a day?",
            "alcohol": "Do you usually have more than 14 alcoholic drinks per week (men) or 7 per week (women)?",
            "healthcare": "Do you have any health insurance or health care coverage?",
            "cost": "In the past 12 months, did you need to see a doctor but could not because of the cost?",
            "mental": "During the past 30 days, on how many days was your mental health (including stress, depression, or emotional problems) not good?",
            "physical": "During the past 30 days, on how many days was your physical health (including illness or injury) not good?",
            "walk": "Do you have serious difficulty walking or climbing stairs?",
            "education": "What is your highest level of education?", "income": "What is your annual household income?",
            "fruit_help": "Average servings per day (0–5)", "veg_help": "Average servings per day (0–5)"
        },
        "feature_labels": {
            "HighBP":"High blood pressure", "HighChol":"High cholesterol", "CholCheck":"Cholesterol check",
            "BMI":"Body mass index (BMI)", "Smoker":"Smoking history", "Stroke":"History of stroke",
            "HeartDiseaseorAttack":"Heart disease or heart attack", "PhysActivity":"Physical activity",
            "Fruits":"Fruit intake", "Veggies":"Vegetable intake", "HvyAlcoholConsump":"Heavy alcohol intake",
            "AnyHealthcare":"Health care coverage", "NoDocbcCost":"Unable to see a doctor due to cost",
            "GenHlth":"General health", "MentHlth":"Mental health", "PhysHlth":"Physical health",
            "DiffWalk":"Difficulty walking", "Age":"Age group", "Sex":"Sex (survey coding)",
            "Education":"Education level", "Income":"Household income"
        },
        "dimensions": {"Age group":"Age group", "Sex group":"Sex group", "Income group":"Income group", "Education group":"Education group"},
        "group_label":"Group", "metric_label":"Metric", "score_label":"Score",
        "decision_rule":"Decision rule: threshold = false-positive cost / (false-positive cost + false-negative cost). Costs are relative assumptions, not measured monetary costs or medical advice.",
        "subgroup_note":"Subgroup metrics are descriptive. Differences alone do not prove bias or fairness. Consider sample sizes, uncertainty, prevalence, and the consequences of errors.",
        "data_source_text":"Diabetes Health Indicators, CDC BRFSS 2015 (Kaggle). The project uses the original, imbalanced dataset rather than a balanced 50/50 version.",
        "rows_label":"Number of survey records", "original_features_label":"Original health indicators", "model_inputs_label":"Model inputs",
        "model_card_title":"Model card", "model_label":"Model", "target_label":"Target", "evaluation_label":"Evaluation", "fairness_label":"Fairness review",
        "model_target_text":"Self-reported diabetes status (binary survey label)",
        "fairness_detail":"Age, sex, income, and education groups are reviewed. Group metrics are descriptive and do not establish fairness on their own.",
        "source_detail":"Source and evaluation details are documented in README.md and models/model_meta.json.",
        "machine_translation_note":"This interface is machine-translated. Medical wording may contain errors. Do not use this educational app to diagnose a condition or make clinical decisions.",
        "global_language_setup":"To enable the expanded language list, add a Google Cloud Translation Basic API key to Streamlit Secrets. The API may incur usage charges.",
        "score_note":"This is the model's output displayed on a 0–100 scale. It is not a medically validated risk score, a personal probability, or a diagnosis.",
        "result_score_explain":"This number is calculated by the computer from your answers. It is only an educational reference and cannot tell you whether you have diabetes.",
        "plot_group_axis":"Group", "plot_score_axis":"Rate"
    },
    "vi": {
        "brand_caption":"THÔNG TIN SỨC KHỎE ĐƯỜNG HUYẾT", "home_eyebrow":"CÙNG BẠN CHĂM SÓC SỨC KHỎE", "sim_eyebrow":"PHÂN TÍCH QUYẾT ĐỊNH", "fair_eyebrow":"ĐÁNH GIÁ MÔ HÌNH", "about_eyebrow":"VỀ DỰ ÁN",
        "basic_information":"Thông tin cơ bản",
        "education_levels":["Chưa đi học / mẫu giáo", "Lớp 1–8", "Lớp 9–11", "Lớp 12 / tương đương", "Cao đẳng/đại học 1–3 năm", "Đại học từ 4 năm trở lên"],
        "income_levels":["Dưới 10.000 USD", "10.000–14.999 USD", "15.000–19.999 USD", "20.000–24.999 USD", "25.000–34.999 USD", "35.000–49.999 USD", "50.000–74.999 USD", "Từ 75.000 USD trở lên"],
        "questions":{
            "height":"Chiều cao (cm)","weight":"Cân nặng (kg)","bmi_auto":"BMI được tự động tính",
            "cholcheck":"Bạn có kiểm tra cholesterol trong 5 năm qua không?","fruits":"Bạn có ăn trái cây ít nhất một lần mỗi ngày không?","veggies":"Bạn có ăn rau ít nhất một lần mỗi ngày không?",
            "alcohol":"Bạn thường uống hơn 14 đơn vị đồ uống có cồn mỗi tuần (nam) hoặc 7 đơn vị (nữ) không?","healthcare":"Bạn có bảo hiểm y tế hoặc hình thức chăm sóc sức khỏe nào không?",
            "cost":"Trong 12 tháng qua, có khi nào bạn cần đi khám nhưng không thể vì chi phí không?","mental":"Trong 30 ngày qua, có bao nhiêu ngày sức khỏe tinh thần của bạn không tốt (căng thẳng, trầm cảm hoặc vấn đề cảm xúc)?",
            "physical":"Trong 30 ngày qua, có bao nhiêu ngày sức khỏe thể chất của bạn không tốt (ốm đau hoặc chấn thương)?","walk":"Bạn có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không?",
            "education":"Trình độ học vấn cao nhất của bạn là gì?","income":"Thu nhập hằng năm của hộ gia đình bạn là bao nhiêu?","fruit_help":"Số khẩu phần trung bình mỗi ngày (0–5)","veg_help":"Số khẩu phần trung bình mỗi ngày (0–5)"
        },
        "feature_labels":{"HighBP":"Huyết áp cao","HighChol":"Cholesterol cao","CholCheck":"Kiểm tra cholesterol","BMI":"Chỉ số BMI","Smoker":"Tiền sử hút thuốc","Stroke":"Tiền sử đột quỵ","HeartDiseaseorAttack":"Bệnh tim hoặc nhồi máu cơ tim","PhysActivity":"Hoạt động thể chất","Fruits":"Ăn trái cây","Veggies":"Ăn rau","HvyAlcoholConsump":"Uống nhiều đồ uống có cồn","AnyHealthcare":"Có bảo hiểm/chăm sóc y tế","NoDocbcCost":"Không thể đi khám do chi phí","GenHlth":"Sức khỏe tổng quát","MentHlth":"Sức khỏe tinh thần","PhysHlth":"Sức khỏe thể chất","DiffWalk":"Khó khăn khi đi bộ","Age":"Nhóm tuổi","Sex":"Giới tính theo khảo sát","Education":"Trình độ học vấn","Income":"Thu nhập hộ gia đình"},
        "dimensions":{"Age group":"Nhóm tuổi","Sex group":"Giới tính","Income group":"Nhóm thu nhập","Education group":"Trình độ học vấn"},
        "group_label":"Nhóm", "metric_label":"Chỉ số", "score_label":"Giá trị", "decision_rule":"Quy tắc quyết định: ngưỡng = chi phí dương tính giả / (chi phí dương tính giả + chi phí âm tính giả). Đây là các giả định tương đối, không phải chi phí tiền tệ thực tế hay lời khuyên y tế.",
        "subgroup_note":"Các chỉ số theo nhóm chỉ mang tính mô tả. Khác biệt giữa các nhóm không tự chứng minh mô hình có thiên lệch hay công bằng. Cần xem xét cỡ mẫu, độ bất định, tỷ lệ hiện mắc và hậu quả của sai sót.",
        "data_source_text":"Bộ dữ liệu Diabetes Health Indicators, khảo sát CDC BRFSS năm 2015 (Kaggle). Dự án sử dụng bộ dữ liệu gốc có phân bố không cân bằng, không dùng phiên bản cân bằng 50/50.",
        "rows_label":"Số bản ghi khảo sát", "original_features_label":"Chỉ báo sức khỏe ban đầu", "model_inputs_label":"Biến đầu vào của mô hình",
        "model_card_title":"Thẻ thông tin mô hình", "model_label":"Mô hình", "target_label":"Biến dự đoán", "evaluation_label":"Đánh giá", "fairness_label":"Đánh giá công bằng",
        "model_target_text":"Tình trạng tiểu đường do người tham gia tự khai báo (nhãn khảo sát nhị phân)", "fairness_detail":"Mô hình được xem xét theo nhóm tuổi, giới tính, thu nhập và học vấn. Các chỉ số theo nhóm chỉ mang tính mô tả, không tự chứng minh tính công bằng.",
        "source_detail":"Thông tin về nguồn dữ liệu và đánh giá được trình bày trong README.md và models/model_meta.json.",
        "machine_translation_note":"Một số nội dung giao diện được dịch tự động. Cách diễn đạt y tế có thể chưa chính xác. Không dùng ứng dụng giáo dục này để tự chẩn đoán hoặc quyết định điều trị.",
        "global_language_setup":"Để mở danh sách ngôn ngữ rộng hơn, hãy thêm khóa Google Cloud Translation Basic API vào Streamlit Secrets. API có thể phát sinh phí sử dụng.",
        "score_note":"Đây là đầu ra của mô hình được hiển thị trên thang điểm 0–100. Điểm này chưa được kiểm định y khoa, không phải xác suất cá nhân mắc bệnh và cũng không phải chẩn đoán.",
        "result_score_explain":"Máy tính tạo ra con số này dựa trên câu trả lời của bạn. Đây chỉ là thông tin tham khảo mang tính giáo dục, không thể xác định bạn có mắc tiểu đường hay không.",
        "plot_group_axis":"Nhóm", "plot_score_axis":"Tỷ lệ"
    },
    "fr": {
        "brand_caption":"INFORMATIONS SUR LA SANTÉ ET LE DIABÈTE", "home_eyebrow":"UN ACCOMPAGNEMENT BIENVEILLANT", "sim_eyebrow":"ANALYSE DE DÉCISION", "fair_eyebrow":"ÉVALUATION DU MODÈLE", "about_eyebrow":"À PROPOS DU PROJET",
        "basic_information":"Informations générales",
        "education_levels":["Jamais scolarisé / maternelle", "Niveaux 1 à 8", "Niveaux 9 à 11", "Niveau 12 / équivalent GED", "Études supérieures : 1 à 3 ans", "Études supérieures : 4 ans ou plus"],
        "income_levels":["Moins de 10 000 USD", "10 000–14 999 USD", "15 000–19 999 USD", "20 000–24 999 USD", "25 000–34 999 USD", "35 000–49 999 USD", "50 000–74 999 USD", "75 000 USD ou plus"],
        "questions":{"height":"Taille (cm)","weight":"Poids (kg)","bmi_auto":"IMC calculé automatiquement","cholcheck":"Avez-vous fait vérifier votre cholestérol au cours des 5 dernières années ?","fruits":"Mangez-vous des fruits au moins une fois par jour ?","veggies":"Mangez-vous des légumes au moins une fois par jour ?","alcohol":"Buvez-vous habituellement plus de 14 boissons alcoolisées par semaine (hommes) ou 7 (femmes) ?","healthcare":"Avez-vous une assurance maladie ou une couverture de soins ?","cost":"Au cours des 12 derniers mois, avez-vous eu besoin de consulter un médecin sans pouvoir le faire à cause du coût ?","mental":"Au cours des 30 derniers jours, pendant combien de jours votre santé mentale (stress, dépression ou difficultés émotionnelles) n’a-t-elle pas été bonne ?","physical":"Au cours des 30 derniers jours, pendant combien de jours votre santé physique (maladie ou blessure) n’a-t-elle pas été bonne ?","walk":"Avez-vous de grandes difficultés à marcher ou à monter des escaliers ?","education":"Quel est votre niveau d’études le plus élevé ?","income":"Quel est le revenu annuel de votre ménage ?","fruit_help":"Portions moyennes par jour (0–5)","veg_help":"Portions moyennes par jour (0–5)"},
        "feature_labels":{"HighBP":"Hypertension","HighChol":"Cholestérol élevé","CholCheck":"Contrôle du cholestérol","BMI":"Indice de masse corporelle (IMC)","Smoker":"Antécédents de tabagisme","Stroke":"Antécédents d’AVC","HeartDiseaseorAttack":"Maladie cardiaque ou crise cardiaque","PhysActivity":"Activité physique","Fruits":"Consommation de fruits","Veggies":"Consommation de légumes","HvyAlcoholConsump":"Consommation élevée d’alcool","AnyHealthcare":"Couverture de soins de santé","NoDocbcCost":"Consultation impossible en raison du coût","GenHlth":"Santé générale","MentHlth":"Santé mentale","PhysHlth":"Santé physique","DiffWalk":"Difficulté à marcher","Age":"Tranche d’âge","Sex":"Sexe (codage de l’enquête)","Education":"Niveau d’études","Income":"Revenu du ménage"},
        "dimensions":{"Age group":"Tranche d’âge","Sex group":"Sexe","Income group":"Revenu du ménage","Education group":"Niveau d’études"},
        "group_label":"Groupe","metric_label":"Indicateur","score_label":"Valeur","decision_rule":"Règle de décision : seuil = coût des faux positifs / (coût des faux positifs + coût des faux négatifs). Il s’agit d’hypothèses relatives, et non de coûts financiers mesurés ou de conseils médicaux.",
        "subgroup_note":"Les indicateurs par groupe sont descriptifs. Les différences ne prouvent pas à elles seules un biais ou l’équité. Il faut tenir compte de la taille des échantillons, de l’incertitude, de la prévalence et des conséquences des erreurs.",
        "data_source_text":"Diabetes Health Indicators, enquête CDC BRFSS 2015 (Kaggle). Le projet utilise le jeu de données original déséquilibré, et non une version équilibrée à 50/50.",
        "rows_label":"Nombre de réponses à l’enquête", "original_features_label":"Indicateurs de santé d’origine", "model_inputs_label":"Variables utilisées par le modèle",
        "model_card_title":"Fiche du modèle", "model_label":"Modèle", "target_label":"Cible", "evaluation_label":"Évaluation", "fairness_label":"Évaluation de l’équité",
        "model_target_text":"Statut diabétique déclaré par la personne (étiquette binaire d’enquête)", "fairness_detail":"Les résultats sont examinés par âge, sexe, revenu et niveau d’études. Les indicateurs par groupe sont descriptifs et ne prouvent pas à eux seuls l’équité.",
        "source_detail":"Les détails sur la source et l’évaluation figurent dans README.md et models/model_meta.json.",
        "machine_translation_note":"Certains textes de l’interface sont traduits automatiquement. Les termes médicaux peuvent comporter des erreurs. N’utilisez pas cette application éducative pour établir un diagnostic ou prendre une décision clinique.",
        "global_language_setup":"Pour afficher davantage de langues, ajoutez une clé Google Cloud Translation Basic API dans Streamlit Secrets. L’utilisation de l’API peut entraîner des frais.",
        "score_note":"Il s’agit de la sortie du modèle affichée sur une échelle de 0 à 100. Ce score n’est pas un score médical validé, une probabilité personnelle ni un diagnostic.",
        "result_score_explain":"Ce nombre est calculé par ordinateur à partir de vos réponses. Il est fourni à titre éducatif et ne permet pas de savoir si vous avez le diabète.",
        "plot_group_axis":"Groupe", "plot_score_axis":"Taux"
    },
    "zh": {
        "brand_caption":"血糖健康信息", "home_eyebrow":"友善的健康小检查", "sim_eyebrow":"决策分析", "fair_eyebrow":"模型评估", "about_eyebrow":"项目介绍",
        "basic_information":"基本信息",
        "education_levels":["未上学／幼儿园", "1–8 年级", "9–11 年级", "12 年级／同等学历", "大专或大学 1–3 年", "大学 4 年或以上"],
        "income_levels":["低于 10,000 美元", "10,000–14,999 美元", "15,000–19,999 美元", "20,000–24,999 美元", "25,000–34,999 美元", "35,000–49,999 美元", "50,000–74,999 美元", "75,000 美元或以上"],
        "questions":{"height":"身高（厘米）","weight":"体重（公斤）","bmi_auto":"BMI 自动计算","cholcheck":"过去 5 年内，您是否检查过胆固醇？","fruits":"您是否每天至少吃一次水果？","veggies":"您是否每天至少吃一次蔬菜？","alcohol":"您通常每周饮酒超过 14 杯（男性）或 7 杯（女性）吗？","healthcare":"您是否拥有医疗保险或其他医疗保障？","cost":"过去 12 个月中，您是否曾因费用原因而无法就医？","mental":"过去 30 天中，有多少天您的心理健康状况不佳（包括压力、抑郁或情绪问题）？","physical":"过去 30 天中，有多少天您的身体健康状况不佳（包括疾病或受伤）？","walk":"您是否在步行或爬楼梯时有严重困难？","education":"您的最高受教育程度是什么？","income":"您家庭的年收入是多少？","fruit_help":"每天平均份数（0–5）","veg_help":"每天平均份数（0–5）"},
        "feature_labels":{"HighBP":"高血压","HighChol":"高胆固醇","CholCheck":"胆固醇检查","BMI":"身体质量指数（BMI）","Smoker":"吸烟史","Stroke":"中风史","HeartDiseaseorAttack":"心脏病或心脏病发作","PhysActivity":"身体活动","Fruits":"水果摄入","Veggies":"蔬菜摄入","HvyAlcoholConsump":"大量饮酒","AnyHealthcare":"医疗保障","NoDocbcCost":"因费用无法就医","GenHlth":"总体健康状况","MentHlth":"心理健康","PhysHlth":"身体健康","DiffWalk":"行走困难","Age":"年龄组","Sex":"性别（调查编码）","Education":"受教育程度","Income":"家庭收入"},
        "dimensions":{"Age group":"年龄组","Sex group":"性别","Income group":"收入组","Education group":"受教育程度"},
        "group_label":"组别","metric_label":"指标","score_label":"数值","decision_rule":"决策规则：阈值 = 假阳性成本 /（假阳性成本 + 假阴性成本）。这些是相对假设，并非实际货币成本或医疗建议。",
        "subgroup_note":"分组指标仅用于描述。组间差异本身不能证明存在偏差或公平。还需考虑样本量、不确定性、患病率以及错误可能带来的后果。",
        "data_source_text":"Diabetes Health Indicators，CDC BRFSS 2015 调查数据（Kaggle）。本项目使用原始的不平衡数据集，而不是 50/50 平衡版本。",
        "rows_label":"调查记录数", "original_features_label":"原始健康指标数", "model_inputs_label":"模型输入变量数",
        "model_card_title":"模型说明卡", "model_label":"模型", "target_label":"预测目标", "evaluation_label":"评估方式", "fairness_label":"公平性审查",
        "model_target_text":"受访者自我报告的糖尿病状态（二元调查标签）", "fairness_detail":"按年龄、性别、收入和教育程度审查分组表现。分组指标仅作描述，不能单独证明公平性。",
        "source_detail":"数据来源和评估细节见 README.md 和 models/model_meta.json。",
        "machine_translation_note":"部分界面文字由机器翻译生成，医学用语可能存在错误。本教育应用不能用于诊断或临床决策。",
        "global_language_setup":"如需启用更多语言，请在 Streamlit Secrets 中添加 Google Cloud Translation Basic API 密钥。API 使用可能产生费用。",
        "score_note":"这是模型输出并显示在 0–100 的刻度上。它不是经过医学验证的风险评分、个人患病概率或诊断结果。",
        "result_score_explain":"这个数字由计算机根据您的回答计算得出，仅供教育参考，无法判断您是否患有糖尿病。",
        "plot_group_axis":"组别", "plot_score_axis":"比例"
    },
    "ja": {
        "brand_caption":"血糖と健康に関する情報", "home_eyebrow":"やさしい健康チェック", "sim_eyebrow":"意思決定分析", "fair_eyebrow":"モデル評価", "about_eyebrow":"プロジェクトについて",
        "basic_information":"基本情報",
        "education_levels":["未就学／幼稚園", "1～8年生", "9～11年生", "12年生／GED相当", "大学等1～3年", "大学等4年以上"],
        "income_levels":["10,000米ドル未満", "10,000～14,999米ドル", "15,000～19,999米ドル", "20,000～24,999米ドル", "25,000～34,999米ドル", "35,000～49,999米ドル", "50,000～74,999米ドル", "75,000米ドル以上"],
        "questions":{"height":"身長（cm）","weight":"体重（kg）","bmi_auto":"BMIは自動計算されます","cholcheck":"過去5年以内にコレステロール検査を受けましたか？","fruits":"果物を毎日少なくとも1回食べますか？","veggies":"野菜を毎日少なくとも1回食べますか？","alcohol":"通常、週に14杯（男性）または7杯（女性）を超える量の酒類を飲みますか？","healthcare":"健康保険または医療保障に加入していますか？","cost":"過去12か月間に、費用のために必要な受診ができなかったことがありますか？","mental":"過去30日間で、ストレス・うつ・感情面の問題など、心の健康状態が良くなかった日は何日ありましたか？","physical":"過去30日間で、病気やけがなど、身体の健康状態が良くなかった日は何日ありましたか？","walk":"歩行や階段の上り下りに大きな困難がありますか？","education":"最終学歴を教えてください。","income":"世帯の年間収入を教えてください。","fruit_help":"1日あたりの平均摂取量（0～5）","veg_help":"1日あたりの平均摂取量（0～5）"},
        "feature_labels":{"HighBP":"高血圧","HighChol":"高コレステロール","CholCheck":"コレステロール検査","BMI":"体格指数（BMI）","Smoker":"喫煙歴","Stroke":"脳卒中の既往","HeartDiseaseorAttack":"心疾患・心臓発作","PhysActivity":"身体活動","Fruits":"果物の摂取","Veggies":"野菜の摂取","HvyAlcoholConsump":"多量の飲酒","AnyHealthcare":"医療保障","NoDocbcCost":"費用による受診困難","GenHlth":"全般的な健康状態","MentHlth":"心の健康","PhysHlth":"身体の健康","DiffWalk":"歩行の困難","Age":"年齢層","Sex":"性別（調査上の分類）","Education":"学歴","Income":"世帯収入"},
        "dimensions":{"Age group":"年齢層","Sex group":"性別","Income group":"所得層","Education group":"学歴"},
        "group_label":"グループ","metric_label":"指標","score_label":"値","decision_rule":"判断ルール：閾値 = 偽陽性のコスト ÷（偽陽性のコスト + 偽陰性のコスト）。これは相対的な仮定であり、実際の金銭的コストや医療上の助言ではありません。",
        "subgroup_note":"グループ別指標は記述的なものです。差があるだけで偏りや公平性が証明されるわけではありません。サンプル数、不確実性、有病率、誤りの影響も考慮してください。",
        "data_source_text":"Diabetes Health Indicators、CDC BRFSS 2015調査データ（Kaggle）。本プロジェクトでは、50/50に調整したデータではなく、元の不均衡データを使用しています。",
        "rows_label":"調査回答数", "original_features_label":"元データの健康指標数", "model_inputs_label":"モデルへの入力変数数",
        "model_card_title":"モデルカード", "model_label":"モデル", "target_label":"予測対象", "evaluation_label":"評価方法", "fairness_label":"公平性の確認",
        "model_target_text":"本人が申告した糖尿病の状態（調査上の二値ラベル）", "fairness_detail":"年齢、性別、所得、学歴のグループ別に確認しています。グループ指標は記述的なもので、それだけで公平性を証明するものではありません。",
        "source_detail":"データソースと評価の詳細は README.md および models/model_meta.json に記載されています。",
        "machine_translation_note":"一部の画面表示は機械翻訳です。医療用語に誤りが含まれる可能性があります。この教育用アプリを診断や臨床判断に使用しないでください。",
        "global_language_setup":"さらに多くの言語を利用するには、Streamlit Secrets に Google Cloud Translation Basic API キーを追加してください。API の利用には料金が発生する場合があります。",
        "score_note":"これはモデルの出力を0～100の目盛りで表示したものです。医学的に検証されたリスクスコア、個人の発症確率、診断ではありません。",
        "result_score_explain":"この数値は回答内容をもとにコンピューターが計算したものです。教育目的の参考情報であり、糖尿病の有無を判断するものではありません。",
        "plot_group_axis":"グループ", "plot_score_axis":"割合"
    },
    "ko": {
        "brand_caption":"혈당 및 건강 정보", "home_eyebrow":"친근한 건강 확인", "sim_eyebrow":"의사결정 분석", "fair_eyebrow":"모델 검토", "about_eyebrow":"프로젝트 소개",
        "basic_information":"기본 정보",
        "education_levels":["학교 교육 없음 / 유치원", "1–8학년", "9–11학년", "12학년 / GED", "대학 1–3년", "대학 4년 이상"],
        "income_levels":["10,000달러 미만", "10,000–14,999달러", "15,000–19,999달러", "20,000–24,999달러", "25,000–34,999달러", "35,000–49,999달러", "50,000–74,999달러", "75,000달러 이상"],
        "questions":{"height":"키 (cm)","weight":"몸무게 (kg)","bmi_auto":"BMI 자동 계산","cholcheck":"지난 5년 동안 콜레스테롤 검사를 받은 적이 있습니까?","fruits":"하루에 한 번 이상 과일을 먹습니까?","veggies":"하루에 한 번 이상 채소를 먹습니까?","alcohol":"보통 일주일에 14잔(남성) 또는 7잔(여성)보다 많이 술을 마십니까?","healthcare":"건강보험 또는 의료 보장을 받고 있습니까?","cost":"지난 12개월 동안 비용 때문에 필요한 진료를 받지 못한 적이 있습니까?","mental":"지난 30일 동안 스트레스, 우울감 또는 정서적 문제 등 정신 건강이 좋지 않았던 날은 며칠입니까?","physical":"지난 30일 동안 질병이나 부상 등 신체 건강이 좋지 않았던 날은 며칠입니까?","walk":"걷거나 계단을 오르는 데 심각한 어려움이 있습니까?","education":"최종 학력은 무엇입니까?","income":"가구의 연간 소득은 얼마입니까?","fruit_help":"하루 평균 섭취량 (0–5회)","veg_help":"하루 평균 섭취량 (0–5회)"},
        "feature_labels":{"HighBP":"고혈압","HighChol":"고콜레스테롤","CholCheck":"콜레스테롤 검사","BMI":"체질량지수(BMI)","Smoker":"흡연 이력","Stroke":"뇌졸중 병력","HeartDiseaseorAttack":"심장 질환 또는 심장마비","PhysActivity":"신체 활동","Fruits":"과일 섭취","Veggies":"채소 섭취","HvyAlcoholConsump":"과도한 음주","AnyHealthcare":"의료 보장","NoDocbcCost":"비용으로 인한 진료 불가","GenHlth":"전반적인 건강","MentHlth":"정신 건강","PhysHlth":"신체 건강","DiffWalk":"보행 어려움","Age":"연령대","Sex":"성별(설문 코딩)","Education":"교육 수준","Income":"가구 소득"},
        "dimensions":{"Age group":"연령대","Sex group":"성별","Income group":"소득 그룹","Education group":"교육 수준"},
        "group_label":"그룹","metric_label":"지표","score_label":"값","decision_rule":"의사결정 규칙: 임계값 = 위양성 비용 / (위양성 비용 + 위음성 비용). 이는 상대적 가정이며 실제 금전 비용이나 의학적 조언이 아닙니다.",
        "subgroup_note":"그룹별 지표는 설명을 위한 자료입니다. 차이만으로 편향이나 공정성이 입증되지는 않습니다. 표본 수, 불확실성, 유병률 및 오류의 결과를 함께 고려해야 합니다.",
        "data_source_text":"Diabetes Health Indicators, CDC BRFSS 2015 설문 데이터(Kaggle). 이 프로젝트는 50/50 균형 데이터가 아닌 원본의 불균형 데이터셋을 사용합니다.",
        "rows_label":"설문 기록 수", "original_features_label":"원본 건강 지표 수", "model_inputs_label":"모델 입력 변수 수",
        "model_card_title":"모델 카드", "model_label":"모델", "target_label":"예측 대상", "evaluation_label":"평가 방식", "fairness_label":"공정성 검토",
        "model_target_text":"자가 보고한 당뇨병 상태(설문 이진 레이블)", "fairness_detail":"연령, 성별, 소득 및 교육 그룹별 결과를 검토합니다. 그룹 지표는 설명용이며 그 자체로 공정성을 입증하지 않습니다.",
        "source_detail":"데이터 출처와 평가 세부 사항은 README.md 및 models/model_meta.json에 설명되어 있습니다.",
        "machine_translation_note":"일부 화면 문구는 기계 번역되었습니다. 의료 용어에 오류가 있을 수 있습니다. 이 교육용 앱을 진단이나 임상적 결정에 사용하지 마세요.",
        "global_language_setup":"더 많은 언어를 사용하려면 Streamlit Secrets에 Google Cloud Translation Basic API 키를 추가하세요. API 사용료가 발생할 수 있습니다.",
        "score_note":"이 값은 모델의 출력을 0–100 척도로 표시한 것입니다. 의학적으로 검증된 위험 점수, 개인별 발병 확률 또는 진단이 아닙니다.",
        "result_score_explain":"이 숫자는 입력한 답변을 바탕으로 컴퓨터가 계산한 값입니다. 교육용 참고 정보일 뿐 당뇨병 여부를 알려주지는 않습니다.",
        "plot_group_axis":"그룹", "plot_score_axis":"비율"
    },
    "es": {
        "brand_caption":"INFORMACIÓN SOBRE SALUD Y GLUCOSA", "home_eyebrow":"UNA REVISIÓN DE SALUD AMABLE", "sim_eyebrow":"ANÁLISIS DE DECISIONES", "fair_eyebrow":"REVISIÓN DEL MODELO", "about_eyebrow":"SOBRE ESTE PROYECTO",
        "basic_information":"Información básica",
        "education_levels":["Sin escolarización / jardín de infancia", "Grados 1–8", "Grados 9–11", "Grado 12 / GED", "Estudios universitarios: 1–3 años", "Estudios universitarios: 4 años o más"],
        "income_levels":["Menos de 10.000 USD", "10.000–14.999 USD", "15.000–19.999 USD", "20.000–24.999 USD", "25.000–34.999 USD", "35.000–49.999 USD", "50.000–74.999 USD", "75.000 USD o más"],
        "questions":{"height":"Estatura (cm)","weight":"Peso (kg)","bmi_auto":"El IMC se calcula automáticamente","cholcheck":"¿Se ha realizado una prueba de colesterol en los últimos 5 años?","fruits":"¿Come fruta al menos una vez al día?","veggies":"¿Come verduras al menos una vez al día?","alcohol":"¿Suele consumir más de 14 bebidas alcohólicas por semana (hombres) o 7 (mujeres)?","healthcare":"¿Tiene seguro médico o algún tipo de cobertura sanitaria?","cost":"En los últimos 12 meses, ¿necesitó ir al médico pero no pudo por el coste?","mental":"En los últimos 30 días, ¿durante cuántos días no estuvo bien su salud mental (estrés, depresión o problemas emocionales)?","physical":"En los últimos 30 días, ¿durante cuántos días no estuvo bien su salud física (enfermedad o lesión)?","walk":"¿Tiene mucha dificultad para caminar o subir escaleras?","education":"¿Cuál es su nivel educativo más alto?","income":"¿Cuál es el ingreso anual de su hogar?","fruit_help":"Porciones medias al día (0–5)","veg_help":"Porciones medias al día (0–5)"},
        "feature_labels":{"HighBP":"Presión arterial alta","HighChol":"Colesterol alto","CholCheck":"Control del colesterol","BMI":"Índice de masa corporal (IMC)","Smoker":"Antecedentes de tabaquismo","Stroke":"Antecedentes de accidente cerebrovascular","HeartDiseaseorAttack":"Enfermedad cardíaca o infarto","PhysActivity":"Actividad física","Fruits":"Consumo de fruta","Veggies":"Consumo de verduras","HvyAlcoholConsump":"Consumo elevado de alcohol","AnyHealthcare":"Cobertura sanitaria","NoDocbcCost":"No pudo acudir al médico por el coste","GenHlth":"Salud general","MentHlth":"Salud mental","PhysHlth":"Salud física","DiffWalk":"Dificultad para caminar","Age":"Grupo de edad","Sex":"Sexo (codificación de la encuesta)","Education":"Nivel educativo","Income":"Ingresos del hogar"},
        "dimensions":{"Age group":"Grupo de edad","Sex group":"Sexo","Income group":"Grupo de ingresos","Education group":"Nivel educativo"},
        "group_label":"Grupo","metric_label":"Indicador","score_label":"Valor","decision_rule":"Regla de decisión: umbral = coste de falsos positivos / (coste de falsos positivos + coste de falsos negativos). Son supuestos relativos, no costes monetarios medidos ni consejos médicos.",
        "subgroup_note":"Las métricas por grupo son descriptivas. Las diferencias por sí solas no demuestran sesgo ni equidad. Considere el tamaño de la muestra, la incertidumbre, la prevalencia y las consecuencias de los errores.",
        "data_source_text":"Diabetes Health Indicators, encuesta CDC BRFSS de 2015 (Kaggle). El proyecto utiliza el conjunto original desequilibrado, no una versión equilibrada al 50/50.",
        "rows_label":"Número de registros de la encuesta", "original_features_label":"Indicadores de salud originales", "model_inputs_label":"Variables de entrada del modelo",
        "model_card_title":"Ficha del modelo", "model_label":"Modelo", "target_label":"Objetivo", "evaluation_label":"Evaluación", "fairness_label":"Revisión de equidad",
        "model_target_text":"Estado de diabetes autodeclarado (etiqueta binaria de la encuesta)", "fairness_detail":"Se revisan los grupos por edad, sexo, ingresos y educación. Las métricas por grupo son descriptivas y no demuestran por sí solas la equidad.",
        "source_detail":"Los detalles de la fuente y la evaluación se documentan en README.md y models/model_meta.json.",
        "machine_translation_note":"Parte del texto de la interfaz se traduce automáticamente. Los términos médicos pueden contener errores. No utilice esta aplicación educativa para diagnosticar ni tomar decisiones clínicas.",
        "global_language_setup":"Para habilitar más idiomas, añada una clave de Google Cloud Translation Basic API en Streamlit Secrets. El uso de la API puede generar cargos.",
        "score_note":"Este es el resultado del modelo mostrado en una escala de 0 a 100. No es una puntuación de riesgo validada médicamente, una probabilidad personal ni un diagnóstico.",
        "result_score_explain":"El ordenador calcula este número a partir de sus respuestas. Es solo una referencia educativa y no permite saber si tiene diabetes.",
        "plot_group_axis":"Grupo", "plot_score_axis":"Tasa"
    }
}


MORE_CATALOG = {
    "en": {
        "hero":"Understand your health. Take your next step with confidence.",
        "estimate":"Model output score (0–100)",
        "sidebar_source_label":"CDC BRFSS 2015 · Educational project",
        "language_api_error":"Google Cloud Translation is unavailable. The app is showing its built-in languages instead.",
        "training_metrics_notice":"Run `python train.py` to create the model and evaluation metrics.",
        "metric_threshold_info":"Decision threshold used for these classification metrics:",
        "factor_more":"Answers linked with a higher model score", "factor_less":"Answers linked with a lower model score",
        "next_steps_text":"If you are concerned, have symptoms, or want to know your blood sugar level, contact a healthcare professional to discuss an appropriate test. Do not rely on this score alone.",
    },
    "vi": {
        "hero":"Cùng tìm hiểu sức khỏe và chủ động bước tiếp theo.",
        "estimate":"Điểm đầu ra của mô hình (0–100)",
        "sidebar_source_label":"CDC BRFSS 2015 · Dự án giáo dục",
        "language_api_error":"Không kết nối được Google Cloud Translation. Ứng dụng đang hiển thị các ngôn ngữ có sẵn.",
        "training_metrics_notice":"Chạy `python train.py` để tạo mô hình và các chỉ số đánh giá.",
        "metric_threshold_info":"Ngưỡng quyết định được dùng để tính các chỉ số phân loại:",
        "factor_more":"Câu trả lời gắn với điểm mô hình cao hơn", "factor_less":"Câu trả lời gắn với điểm mô hình thấp hơn",
        "next_steps_text":"Nếu bạn lo lắng, có triệu chứng hoặc muốn biết chỉ số đường huyết, hãy đến cơ sở y tế để được tư vấn về xét nghiệm phù hợp. Không nên chỉ dựa vào điểm số này.",
    },
    "fr": {
        "hero":"Mieux comprendre votre santé et choisir la prochaine étape.",
        "estimate":"Score de sortie du modèle (0–100)",
        "sidebar_source_label":"CDC BRFSS 2015 · Projet éducatif",
        "language_api_error":"Google Cloud Translation est indisponible. Les langues intégrées sont affichées.",
        "training_metrics_notice":"Exécutez `python train.py` pour créer le modèle et les indicateurs d’évaluation.",
        "metric_threshold_info":"Seuil de décision utilisé pour calculer ces indicateurs :",
        "factor_more":"Réponses associées à un score de modèle plus élevé", "factor_less":"Réponses associées à un score de modèle plus faible",
        "next_steps_text":"En cas d’inquiétude, de symptômes ou si vous souhaitez connaître votre glycémie, contactez un professionnel de santé pour discuter d’un test adapté. Ne vous fiez pas uniquement à ce score.",
    },
    "zh": {
        "hero":"了解自己的健康状况，自信地迈出下一步。", "estimate":"模型输出分数（0–100）",
        "sidebar_source_label":"CDC BRFSS 2015 · 教育项目", "language_api_error":"Google Cloud Translation 暂时不可用。现在显示应用内置语言。",
        "training_metrics_notice":"运行 `python train.py` 以生成模型和评估指标。", "metric_threshold_info":"计算这些分类指标时使用的决策阈值：",
        "factor_more":"与较高模型分数相关的回答", "factor_less":"与较低模型分数相关的回答",
        "next_steps_text":"如果您感到担忧、有相关症状或想了解血糖水平，请联系医疗专业人员，咨询适合的检查。请勿仅依赖此分数。"
    },
    "ja": {
        "hero":"健康状態を知り、自信を持って次の一歩へ。", "estimate":"モデル出力スコア（0～100）",
        "sidebar_source_label":"CDC BRFSS 2015 · 教育プロジェクト", "language_api_error":"Google Cloud Translationを利用できません。アプリ内蔵の言語を表示しています。",
        "training_metrics_notice":"`python train.py` を実行してモデルと評価指標を作成してください。", "metric_threshold_info":"これらの分類指標に使用した判断閾値：",
        "factor_more":"モデルスコアの上昇に関連する回答", "factor_less":"モデルスコアの低下に関連する回答",
        "next_steps_text":"不安がある場合、症状がある場合、または血糖値を知りたい場合は、医療専門家に相談して適切な検査について確認してください。このスコアだけを頼りにしないでください。"
    },
    "ko": {
        "hero":"건강 상태를 이해하고 다음 단계를 자신 있게 준비하세요.", "estimate":"모델 출력 점수(0–100)",
        "sidebar_source_label":"CDC BRFSS 2015 · 교육 프로젝트", "language_api_error":"Google Cloud Translation을 사용할 수 없습니다. 앱에 기본 제공된 언어를 표시합니다.",
        "training_metrics_notice":"모델과 평가 지표를 생성하려면 `python train.py`를 실행하세요.", "metric_threshold_info":"이 분류 지표에 사용된 의사결정 임계값:",
        "factor_more":"모델 점수가 높아지는 것과 관련된 답변", "factor_less":"모델 점수가 낮아지는 것과 관련된 답변",
        "next_steps_text":"걱정이 되거나 증상이 있거나 혈당 수치를 알고 싶다면 의료 전문가에게 적절한 검사를 상담하세요. 이 점수만으로 판단하지 마세요.",
    },
    "es": {
        "hero":"Comprende mejor tu salud y da el siguiente paso con confianza.", "estimate":"Puntuación de salida del modelo (0–100)",
        "sidebar_source_label":"CDC BRFSS 2015 · Proyecto educativo", "language_api_error":"Google Cloud Translation no está disponible. Se muestran los idiomas integrados.",
        "training_metrics_notice":"Ejecuta `python train.py` para crear el modelo y las métricas de evaluación.", "metric_threshold_info":"Umbral de decisión utilizado para estas métricas de clasificación:",
        "factor_more":"Respuestas relacionadas con una puntuación más alta del modelo", "factor_less":"Respuestas relacionadas con una puntuación más baja del modelo",
        "next_steps_text":"Si tiene dudas, presenta síntomas o desea conocer su nivel de glucosa, consulte a un profesional sanitario sobre una prueba adecuada. No se base únicamente en esta puntuación.",
    },
}

for _locale_code, _locale_values in EXTRA_CATALOG.items():
    T.setdefault(_locale_code, {})
    T[_locale_code].update(_locale_values)
for _locale_code, _locale_values in MORE_CATALOG.items():
    T.setdefault(_locale_code, {})
    T[_locale_code].update(_locale_values)

# Keep the English wording accurate now that the result is shown as a 0–100 model output.
T["en"]["score_note"] = EXTRA_CATALOG["en"]["score_note"]



st.set_page_config(page_title="DiaCheck", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

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

@st.cache_resource
def load_assets():
    model = joblib.load(ROOT/"models"/"pipeline.joblib")
    meta = json.loads((ROOT/"models"/"model_meta.json").read_text(encoding="utf-8"))
    fair = pd.read_csv(ROOT/"data"/"fairness_metrics.csv")
    return model, meta, fair

model, META, FAIR = load_assets()
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
        return copy.deepcopy(T["en"]), False, False
    if api_key:
        try:
            translated = _cached_translation(catalog_json, language_code, api_key)
            return merge_catalog(translated, T.get(language_code, {})), language_code not in EXTRA_CATALOG, False
        except LocalizationError:
            # Prefer a built-in reviewed pack when available; otherwise use English as a safe fallback.
            fallback = merge_catalog(copy.deepcopy(T["en"]), T.get(language_code, {}))
            return fallback, language_code not in EXTRA_CATALOG, True
    return merge_catalog(copy.deepcopy(T["en"]), T.get(language_code, {})), False, False


translation_api_key = _get_translation_key()
language_catalog_json = json.dumps(T["en"], ensure_ascii=False, sort_keys=True)
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
    old_language = st.session_state.get("lang", "English")
    old_code = {name: code for code, name in MANUAL_LANGUAGE_NAMES.items()}.get(old_language, "en")
    st.session_state.language_code = old_code
if st.session_state.language_code not in language_options:
    st.session_state.language_code = "en"

L, locale_is_machine_translated, locale_translation_failed = _build_locale(st.session_state.language_code, translation_api_key, language_catalog_json)

with st.sidebar:
    st.markdown("## 🩺 DiaCheck")
    st.caption(L["brand_caption"])
    selected_language = st.selectbox(
        "🌐 " + L["select_lang"],
        options=language_options,
        format_func=lambda code: language_names.get(code, code),
        key="language_code",
    )
    if selected_language != st.session_state.language_code:
        st.session_state.language_code = selected_language
    if locale_translation_failed or language_api_error:
        st.warning(L["language_api_error"])
    elif locale_is_machine_translated:
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

def fmt_pct(x):
    return f"{100*x:.1f}%"

def contribution_table(case, locale):
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["model"]
    arr = np.array([case[f] for f in META["features"]], dtype=float)
    z = (arr - scaler.mean_) / scaler.scale_
    contributions = z * lr.coef_[0]
    labels = locale.get("feature_labels", T["en"]["feature_labels"])
    out = pd.DataFrame({"key":META["features"], "feature":[labels.get(f, f) for f in META["features"]],
                        "contribution":contributions})
    return out.sort_values("contribution", ascending=False)


def render_hero(eyebrow, title, description):
    hero_text, hero_characters = st.columns([4.5, 1.25], vertical_alignment="center")
    with hero_text:
        st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)
    with hero_characters:
        if CHARACTER_IMAGE.exists():
            st.image(str(CHARACTER_IMAGE), width=220)


if page_key == "home":
    render_hero(L["home_eyebrow"], L["hero"], L["intro"])
    st.caption("🔒 " + L["privacy"])
    st.subheader(L["screen"])
    q = L["questions"]
    yes_no = [False, True]
    yes_no_label = lambda value: L["yes"] if value else L["no"]
    with st.form("screening_form"):
        st.markdown("#### " + L["basic_information"])
        c1,c2 = st.columns(2)
        with c1:
            age_labels = L["age_options"]
            age_idx = st.selectbox(L["age"], list(range(1,14)), format_func=lambda x: age_labels[x-1], index=6, key="age_group")
            sex_label = st.radio(L["sex"], ["female", "male"], format_func=lambda value: L[value], horizontal=True, key="sex")
            height_cm = st.number_input(q["height"], min_value=100.0, max_value=230.0, value=170.0, step=1.0, key="height_cm")
            weight_kg = st.number_input(q["weight"], min_value=25.0, max_value=300.0, value=70.0, step=0.5, key="weight_kg")
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.metric(q["bmi_auto"], f"{bmi:.1f} kg/m²")
            bp = st.radio(L["bp"], yes_no, format_func=yes_no_label, horizontal=True, key="high_bp")
            chol = st.radio(L["chol"], yes_no, format_func=yes_no_label, horizontal=True, key="high_chol")
            cholcheck = st.radio(q["cholcheck"], yes_no, format_func=yes_no_label, horizontal=True, key="chol_check")
            smoker = st.radio(L["smoke"], yes_no, format_func=yes_no_label, horizontal=True, key="smoking_history")
            stroke = st.radio(L["stroke"], yes_no, format_func=yes_no_label, horizontal=True, key="stroke_history")
            heart = st.radio(L["heart"], yes_no, format_func=yes_no_label, horizontal=True, key="heart_history")
        with c2:
            activity = st.radio(L["activity"], yes_no, format_func=yes_no_label, horizontal=True, key="physical_activity")
            fruits = st.radio(q["fruits"], yes_no, format_func=yes_no_label, horizontal=True, key="fruit_intake")
            veggies = st.radio(q["veggies"], yes_no, format_func=yes_no_label, horizontal=True, key="vegetable_intake")
            heavy_alcohol = st.radio(q["alcohol"], yes_no, format_func=yes_no_label, horizontal=True, key="alcohol_intake")
            healthcare = st.radio(q["healthcare"], yes_no, format_func=yes_no_label, horizontal=True, key="healthcare_coverage")
            no_doc_cost = st.radio(q["cost"], yes_no, format_func=yes_no_label, horizontal=True, key="care_access_cost")
            health_labels = [L["excellent"],L["very_good"],L["good"],L["fair"],L["poor"]]
            health_idx = st.select_slider(L["general_health"], options=[1,2,3,4,5],
                                          format_func=lambda x: health_labels[x-1], value=3, key="general_health_rating")
            mental_days = st.slider(q["mental"], 0, 30, 0, key="mental_health_days")
            physical_days = st.slider(q["physical"], 0, 30, 0, key="physical_health_days")
            diff_walk = st.radio(q["walk"], yes_no, format_func=yes_no_label, horizontal=True, key="walking_difficulty")
            education_labels = L["education_levels"]
            education = st.selectbox(q["education"], options=list(range(1,7)), format_func=lambda x: education_labels[x-1], index=5, key="education_level")
            income_labels = L["income_levels"]
            income = st.selectbox(q["income"], options=list(range(1,9)), format_func=lambda x: income_labels[x-1], index=7, key="household_income")
        submitted = st.form_submit_button(L["submit"], use_container_width=True, key="submit_screening")
    if submitted:
        if not 12 <= bmi <= 98:
            st.error(L.get("error", "Please check your height and weight."))
            st.stop()
        case = {
            "HighBP": int(bp),
            "HighChol": int(chol),
            "CholCheck": int(cholcheck),
            "BMI": float(round(bmi)),
            "Smoker": int(smoker),
            "Stroke": int(stroke),
            "HeartDiseaseorAttack": int(heart),
            "PhysActivity": int(activity),
            "Fruits": int(fruits),
            "Veggies": int(veggies),
            "HvyAlcoholConsump": int(heavy_alcohol),
            "AnyHealthcare": int(healthcare),
            "NoDocbcCost": int(no_doc_cost),
            "GenHlth": int(health_idx),
            "MentHlth": int(mental_days),
            "PhysHlth": int(physical_days),
            "DiffWalk": int(diff_walk),
            "Sex": 0 if sex_label == "female" else 1,
            "Age": int(age_idx),
            "Education": int(education),
            "Income": int(income),
        }
        st.session_state.last_case = case
        st.session_state.last_probability = float(model.predict_proba(pd.DataFrame([case], columns=META["features"]))[0,1])
    if st.session_state.last_probability is not None:
        probability = st.session_state.last_probability
        threshold = st.session_state.cost_fp/(st.session_state.cost_fp+st.session_state.cost_fn)
        st.markdown("---")
        st.subheader(L["result"])
        a,b = st.columns([1,1.5])
        with a:
            st.metric(L["estimate"], f"{probability * 100:.1f}/100")
            st.progress(min(max(probability,0.0),1.0))
            st.caption(L.get("score_note", T["en"]["score_note"]))
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
        pos = cdf[cdf.contribution>0].head(5)
        neg = cdf[cdf.contribution<0].sort_values("contribution").head(5)
        x1,x2=st.columns(2)
        plain = L.get("feature_plain", {})
        feature_labels = L.get("feature_labels", T["en"]["feature_labels"])
        with x1:
            st.markdown("**" + L.get("factor_more", L["up"]) + "**")
            if len(pos):
                for _,r in pos.iterrows(): st.write("• " + plain.get(r.key, feature_labels.get(r.key, r.feature)))
            else: st.caption("—")
        with x2:
            st.markdown("**" + L.get("factor_less", L["down"]) + "**")
            if len(neg):
                for _,r in neg.iterrows(): st.write("• " + plain.get(r.key, feature_labels.get(r.key, r.feature)))
            else: st.caption("—")
        st.info(L.get("next_steps_text", "If you are concerned, have symptoms, or want to know your blood sugar level, contact a healthcare professional to discuss an appropriate test. Do not rely on this score alone."))
    else:
        st.info(L["start"])

elif page_key == "sim":
    render_hero(L["sim_eyebrow"], L["cost_title"], L["cost_desc"])
    col1,col2=st.columns(2)
    with col1:
        st.session_state.cost_fn=st.number_input(L["fn"],min_value=0.1,max_value=100.0,value=float(st.session_state.cost_fn),step=0.5,key="cost_fn_input")
    with col2:
        st.session_state.cost_fp=st.number_input(L["fp"],min_value=0.1,max_value=100.0,value=float(st.session_state.cost_fp),step=0.5,key="cost_fp_input")
    threshold=st.session_state.cost_fp/(st.session_state.cost_fp+st.session_state.cost_fn)
    st.markdown(f'<div class="panel"><div class="metric-label">{L["threshold"]}</div><div class="metric-value" style="font-size:2.4rem">{threshold:.3f} · {fmt_pct(threshold)}</div><p class="small-muted">{L["example"]}</p></div>',unsafe_allow_html=True)
    if st.session_state.last_probability is not None:
        p=st.session_state.last_probability
        st.subheader(L["result"])
        st.metric(L["estimate"], f"{p * 100:.1f}/100")
        if p>=threshold: st.warning(L["above"]+" "+L["action_high"])
        else: st.success(L["below"]+" "+L["action_low"])
    else:
        st.info(L["no_result"])
    st.caption(L["decision_rule"])

elif page_key == "fair":
    render_hero(L["fair_eyebrow"], L["fair_title"], L["fair_desc"])
    dimension_options = ["Age group", "Sex group", "Income group", "Education group"]
    dimension = st.selectbox(L["dimension"], dimension_options, format_func=lambda key: L["dimensions"].get(key, key), key="fairness_dimension")
    sub=FAIR[FAIR.dimension==dimension].copy()

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
    rename={"group":L["group_label"], "n":L["n"],"sensitivity":L["sens"],"specificity":L["spec"],
            "precision":L["prec"],"positive_rate":L["pos"],"predicted_positive_rate":L["predpos"]}
    show=sub[["group","n","sensitivity","specificity","precision","positive_rate","predicted_positive_rate"]].rename(columns=rename)
    for c in [L["sens"],L["spec"],L["prec"],L["pos"],L["predpos"]]:
        show[c]=show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(show,use_container_width=True,hide_index=True)
    plot=sub.melt(id_vars=["group"],value_vars=["sensitivity","specificity"],var_name="Metric",value_name="Score")
    plot["Metric"] = plot["Metric"].map({"sensitivity":L["sens"], "specificity":L["spec"]})
    fig=px.bar(plot,x="group",y="Score",color="Metric",barmode="group",range_y=[0,1],
               labels={"group":L["plot_group_axis"],"Score":L["plot_score_axis"]})
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",legend_title_text="")
    st.plotly_chart(fig,use_container_width=True)
    st.caption(L["subgroup_note"])

else:
    render_hero(L["about_eyebrow"], L["about_title"], L["intended_text"])
    st.subheader(L["intended"]); st.write(L["intended_text"])
    st.subheader(L["data"])
    st.write(L["data_source_text"])
    data_cols = st.columns(3)
    data_values = [(L["rows_label"], f"{253680:,}"), (L["original_features_label"], "21"), (L["model_inputs_label"], str(len(META["features"]))) ]
    for col, (label, value) in zip(data_cols, data_values):
        with col:
            st.metric(label, value)
    st.subheader(L["metrics"])
    m=META.get("metrics") or {}
    if not m or m.get("roc_auc") is None:
        st.info(L["training_metrics_notice"])
        m = None
    if m:
        metric_cols=st.columns(4)
        cards=[
        (L["metric_auc"],m["roc_auc"]),
        (L["metric_ap"],m["average_precision"]),
        (L["metric_brier"],m["brier_score"]),
        (L["metric_recall"],m["recall_sensitivity"]),
        (L["metric_acc"],m["accuracy"]),
        (L["metric_precision"],m["precision"]),
        (L["metric_f1"],m["f1"]),
        (L["metric_spec"],m["specificity"])
    ]
        for i,(label,value) in enumerate(cards):
            with metric_cols[i%4]:
                st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value:.3f}</div></div>',unsafe_allow_html=True)
        st.write(L["split_text"])
        st.caption(f"{L['metric_threshold_info']} {m['threshold']:.3f} · {L['fn']}: {META['cost_assumptions']['false_negative']:.1f}; {L['fp']}: {META['cost_assumptions']['false_positive']:.1f}.")
    st.subheader(L["limitations"])
    for item in L["limitations_list"]: st.markdown(f"- {item}")
    st.subheader(L["disclaimer"]); st.warning(L["disclaimer_text"])
    st.subheader(L["model_card_title"])
    st.markdown(f"**{L['model_label']}:** {META['model']}  \n**{L['target_label']}:** {L['model_target_text']}  \n**{L['evaluation_label']}:** {L['split_text']}  \n**{L['fairness_label']}:** {L['fairness_detail']}")
    st.caption(L["source_detail"])

st.markdown("---")
st.caption("DiaCheck · " + L["footer"])
