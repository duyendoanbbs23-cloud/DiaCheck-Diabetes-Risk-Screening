import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
LANGS = ["English", "Tiếng Việt", "Français", "中文", "日本語", "한국어", "Español"]
LANG_CODES = {"English":"en","Tiếng Việt":"vi","Français":"fr","中文":"zh","日本語":"ja","한국어":"ko","Español":"es"}

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
"bp":"Bạn đã từng được thông báo bị huyết áp cao chưa?","chol":"Bạn đã từng được thông báo bị cholesterol cao chưa?","bmi":"Chỉ số BMI của bạn là bao nhiêu?","smoke":"Bạn đã từng hút ít nhất 100 điếu thuốc trong đời chưa?","stroke":"Bạn đã từng bị đột quỵ chưa?","heart":"Bạn đã từng mắc bệnh mạch vành hoặc nhồi máu cơ tim chưa?","activity":"Trong 30 ngày qua, ngoài công việc thường ngày, bạn có tham gia hoạt động thể chất nào không?","health":"Nhìn chung, bạn tự đánh giá sức khỏe của mình như thế nào?","yes":"Có","no":"Không","submit":"Ước tính kết quả sàng lọc","result":"Kết quả sàng lọc","estimate":"Điểm ước tính của mô hình","above":"Ước tính này cao hơn ngưỡng quyết định hiện tại.","below":"Ước tính này thấp hơn ngưỡng quyết định hiện tại.","action_high":"Bạn nên cân nhắc sắp xếp xét nghiệm đường huyết và trao đổi nguy cơ với nhân viên y tế có chuyên môn.","action_low":"Kết quả này không có nghĩa là bạn không cần xét nghiệm. Nếu có triệu chứng, lo ngại hoặc yếu tố nguy cơ, hãy liên hệ nhân viên y tế.","not_diag":"Đây không phải chẩn đoán. Chỉ đánh giá lâm sàng và xét nghiệm phù hợp mới xác định được tình trạng đái tháo đường.",
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
"error":"Vui lòng nhập BMI từ 12 đến 98.","start":"Vui lòng hoàn thành biểu mẫu sàng lọc để xem ước tính.","no_result":"Hãy hoàn thành biểu mẫu tại Trang chủ để xem giải thích cho trường hợp cụ thể.","metric_auc":"ROC-AUC","metric_ap":"Average precision","metric_brier":"Brier score","metric_recall":"Độ nhạy","metric_acc":"Độ chính xác","metric_precision":"Precision","metric_f1":"F1 score","metric_spec":"Độ đặc hiệu","split":"Thiết kế đánh giá","split_text":"Chia tập train/test phân tầng 80/20; random state 42. Các chỉ số là ước tính trên tập kiểm tra, chưa phải xác thực bên ngoài."
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

st.set_page_config(page_title="DiaCheck | Diabetes Risk Screening", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

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
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "last_case" not in st.session_state:
    st.session_state.last_case = None
if "last_probability" not in st.session_state:
    st.session_state.last_probability = None
if "cost_fn" not in st.session_state:
    st.session_state.cost_fn = float(META["cost_assumptions"]["false_negative"])
if "cost_fp" not in st.session_state:
    st.session_state.cost_fp = float(META["cost_assumptions"]["false_positive"])

with st.sidebar:
    st.markdown("## 🩺 DiaCheck")
    st.caption("DIABETES RISK SCREENING")
    lang = st.selectbox("🌐 " + T[LANG_CODES[st.session_state.lang]]["select_lang"], LANGS,
                        index=LANGS.index(st.session_state.lang), key="language_select")
    st.session_state.lang = lang
    L = T[LANG_CODES[lang]]
    page = st.radio(L["nav"], [L["home"], L["sim"], L["fair"], L["about"]], label_visibility="collapsed")
    st.divider()
    st.info("🎓 " + L["footer"])
    st.caption("T3 · CDC BRFSS 2015 · Educational screening")
L = T[LANG_CODES[st.session_state.lang]]

def fmt_pct(x):
    return f"{100*x:.1f}%"

def contribution_table(case):
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["model"]
    arr = np.array([case[f] for f in META["features"]], dtype=float)
    z = (arr - scaler.mean_) / scaler.scale_
    contributions = z * lr.coef_[0]
    labels = {
        "HighBP":"High blood pressure", "HighChol":"High cholesterol", "CholCheck":"Cholesterol check",
        "BMI":"BMI", "Smoker":"Smoking history", "Stroke":"History of stroke",
        "HeartDiseaseorAttack":"Heart disease / heart attack", "PhysActivity":"Physical activity",
        "Fruits":"Fruit consumption", "Veggies":"Vegetable consumption",
        "HvyAlcoholConsump":"Heavy alcohol consumption", "AnyHealthcare":"Health care coverage",
        "NoDocbcCost":"Could not see doctor due to cost", "GenHlth":"General health",
        "MentHlth":"Mental health days", "PhysHlth":"Physical health days",
        "DiffWalk":"Difficulty walking", "Age":"Age group", "Sex":"Sex",
        "Education":"Education", "Income":"Household income"
    }
    out = pd.DataFrame({"key":META["features"], "feature":[labels[f] for f in META["features"]],
                        "contribution":contributions})
    return out.sort_values("contribution", ascending=False)

if page == L["home"]:
    st.markdown(f'<div class="hero"><div class="eyebrow">T3 · PERSONAL HEALTH SCREENING</div><h1>{L["hero"]}</h1><p>{L["intro"]}</p></div>', unsafe_allow_html=True)
    st.caption("🔒 " + L["privacy"])
    st.subheader(L["screen"])
    Q = {
        "en": {"height":"Height (cm)","weight":"Weight (kg)","bmi_auto":"BMI calculated automatically",
               "cholcheck":"Have you had your cholesterol checked within the past 5 years?",
               "fruits":"Do you eat fruit at least once per day?","veggies":"Do you eat vegetables at least once per day?",
               "alcohol":"Do you usually drink more than 14 drinks per week (men) or 7 per week (women)?",
               "healthcare":"Do you have any kind of health care coverage?",
               "cost":"Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
               "mental":"For how many of the past 30 days was your mental health (including stress, depression, and emotional problems) not good?",
               "physical":"For how many of the past 30 days was your physical health (including illness and injury) not good?",
               "walk":"Do you have serious difficulty walking or climbing stairs?","education":"Highest education level",
               "income":"Annual household income","fruit_help":"","veg_help":""},
        "vi": {"height":"Chiều cao (cm)","weight":"Cân nặng (kg)","bmi_auto":"BMI được tự động tính",
               "cholcheck":"Bạn có kiểm tra cholesterol trong vòng 5 năm qua không?",
               "fruits":"Bạn có ăn trái cây ít nhất một lần mỗi ngày không?","veggies":"Bạn có ăn rau ít nhất một lần mỗi ngày không?",
               "alcohol":"Bạn thường uống hơn 14 đơn vị cồn mỗi tuần (nam) hoặc hơn 7 đơn vị mỗi tuần (nữ) không?",
               "healthcare":"Bạn có bất kỳ hình thức bảo hiểm/chăm sóc sức khỏe nào không?",
               "cost":"Trong 12 tháng qua, có khi nào bạn cần đi khám nhưng không thể vì chi phí không?",
               "mental":"Trong 30 ngày qua, có bao nhiêu ngày sức khỏe tinh thần của bạn không tốt (căng thẳng, trầm cảm hoặc vấn đề cảm xúc)?",
               "physical":"Trong 30 ngày qua, có bao nhiêu ngày sức khỏe thể chất của bạn không tốt (ốm đau hoặc chấn thương)?",
               "walk":"Bạn có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không?","education":"Trình độ học vấn cao nhất",
               "income":"Thu nhập hộ gia đình hằng năm","fruit_help":"Số khẩu phần trung bình mỗi ngày (0–5)","veg_help":"Số khẩu phần trung bình mỗi ngày (0–5)"}
    }
    q = Q.get(LANG_CODES[st.session_state.lang], Q["en"])
    yes_no = [L["no"], L["yes"]]
    with st.form("screening_form"):
        st.markdown("#### " + ("Basic information" if st.session_state.lang != "Tiếng Việt" else "Thông tin cơ bản"))
        c1,c2 = st.columns(2)
        with c1:
            age_labels = L["age_options"]
            age_idx = st.selectbox(L["age"], list(range(1,14)), format_func=lambda x: age_labels[x-1], index=6)
            sex_label = st.radio(L["sex"], [L["female"],L["male"]], horizontal=True)
            height_cm = st.number_input(q["height"], min_value=100.0, max_value=230.0, value=170.0, step=1.0)
            weight_kg = st.number_input(q["weight"], min_value=25.0, max_value=300.0, value=70.0, step=0.5)
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.metric(q["bmi_auto"], f"{bmi:.1f} kg/m²")
            bp = st.radio(L["bp"], yes_no, horizontal=True)
            chol = st.radio(L["chol"], yes_no, horizontal=True)
            cholcheck = st.radio(q["cholcheck"], yes_no, horizontal=True)
            smoker = st.radio(L["smoke"], yes_no, horizontal=True)
            stroke = st.radio(L["stroke"], yes_no, horizontal=True)
            heart = st.radio(L["heart"], yes_no, horizontal=True)
        with c2:
            activity = st.radio(L["activity"], [L["yes"],L["no"]], horizontal=True)
            fruits = st.radio(q["fruits"], yes_no, horizontal=True)
            veggies = st.radio(q["veggies"], yes_no, horizontal=True)
            heavy_alcohol = st.radio(q["alcohol"], yes_no, horizontal=True)
            healthcare = st.radio(q["healthcare"], yes_no, horizontal=True)
            no_doc_cost = st.radio(q["cost"], yes_no, horizontal=True)
            health_labels = [L["excellent"],L["very_good"],L["good"],L["fair"],L["poor"]]
            health_idx = st.select_slider(L["general_health"], options=[1,2,3,4,5],
                                          format_func=lambda x: health_labels[x-1], value=3)
            mental_days = st.slider(q["mental"], 0, 30, 0)
            physical_days = st.slider(q["physical"], 0, 30, 0)
            diff_walk = st.radio(q["walk"], yes_no, horizontal=True)
            education_labels = ["Never attended / kindergarten", "Grades 1–8", "Grades 9–11", "Grade 12 / GED", "College 1–3 years", "College 4+ years"]
            education = st.selectbox(q["education"], options=list(range(1,7)), format_func=lambda x: education_labels[x-1], index=5)
            income_labels = ["Below $10,000", "$10,000–14,999", "$15,000–19,999", "$20,000–24,999", "$25,000–34,999", "$35,000–49,999", "$50,000–74,999", "$75,000 or more"]
            income = st.selectbox(q["income"], options=list(range(1,9)), format_func=lambda x: income_labels[x-1], index=7)
        submitted = st.form_submit_button(L["submit"], use_container_width=True)
    if submitted:
        if not 12 <= bmi <= 98:
            st.error(L.get("error", "Please check your height and weight."))
            st.stop()
        case = {
            "HighBP": int(bp == L["yes"]),
            "HighChol": int(chol == L["yes"]),
            "CholCheck": int(cholcheck == L["yes"]),
            "BMI": float(round(bmi)),
            "Smoker": int(smoker == L["yes"]),
            "Stroke": int(stroke == L["yes"]),
            "HeartDiseaseorAttack": int(heart == L["yes"]),
            "PhysActivity": int(activity == L["yes"]),
            "Fruits": int(fruits == L["yes"]),
            "Veggies": int(veggies == L["yes"]),
            "HvyAlcoholConsump": int(heavy_alcohol == L["yes"]),
            "AnyHealthcare": int(healthcare == L["yes"]),
            "NoDocbcCost": int(no_doc_cost == L["yes"]),
            "GenHlth": int(health_idx),
            "MentHlth": int(mental_days),
            "PhysHlth": int(physical_days),
            "DiffWalk": int(diff_walk == L["yes"]),
            "Sex": 0 if sex_label == L["female"] else 1,
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
            st.metric(L["estimate"], fmt_pct(probability))
            st.progress(min(max(probability,0.0),1.0))
            st.caption(L.get("score_note", "This is an illustrative model score, not a clinically calibrated personal probability or diagnosis."))
        with b:
            if probability >= threshold:
                st.warning("⚠️ " + L["above"] + "\n\n" + L["action_high"])
            else:
                st.success("✓ " + L["below"] + "\n\n" + L["action_low"])
            st.info(L["not_diag"])
        st.subheader(L["explain"])
        st.caption(L["contrib"])
        cdf = contribution_table(st.session_state.last_case)
        pos = cdf[cdf.contribution>0].head(5)
        neg = cdf[cdf.contribution<0].sort_values("contribution").head(5)
        x1,x2=st.columns(2)
        plain = L.get("feature_plain", {})
        with x1:
            st.markdown("**" + L.get("factor_more", L["up"]) + "**")
            if len(pos):
                for _,r in pos.iterrows(): st.write("• " + plain.get(r.key, r.feature))
            else: st.caption("—")
        with x2:
            st.markdown("**" + L.get("factor_less", L["down"]) + "**")
            if len(neg):
                for _,r in neg.iterrows(): st.write("• " + plain.get(r.key, r.feature))
            else: st.caption("—")
        st.info(L.get("next_steps_text", "If you are concerned, have symptoms, or want to know your blood sugar level, contact a healthcare professional to discuss an appropriate test. Do not rely on this score alone."))
    else:
        st.info(L["start"])

elif page == L["sim"]:
    st.markdown(f'<div class="hero"><div class="eyebrow">DECISION ANALYSIS</div><h1>{L["cost_title"]}</h1><p>{L["cost_desc"]}</p></div>', unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.session_state.cost_fn=st.number_input(L["fn"],min_value=0.1,max_value=100.0,value=float(st.session_state.cost_fn),step=0.5)
    with col2:
        st.session_state.cost_fp=st.number_input(L["fp"],min_value=0.1,max_value=100.0,value=float(st.session_state.cost_fp),step=0.5)
    threshold=st.session_state.cost_fp/(st.session_state.cost_fp+st.session_state.cost_fn)
    st.markdown(f'<div class="panel"><div class="metric-label">{L["threshold"]}</div><div class="metric-value" style="font-size:2.4rem">{threshold:.3f} · {fmt_pct(threshold)}</div><p class="small-muted">{L["example"]}</p></div>',unsafe_allow_html=True)
    if st.session_state.last_probability is not None:
        p=st.session_state.last_probability
        st.subheader(L["result"])
        st.metric(L["estimate"],fmt_pct(p))
        if p>=threshold: st.warning(L["above"]+" "+L["action_high"])
        else: st.success(L["below"]+" "+L["action_low"])
    else:
        st.info(L["no_result"])
    st.caption("Decision rule: threshold = Cost(FP) / [Cost(FP) + Cost(FN)]. Costs are relative units chosen for the simulator. They are assumptions, not measured monetary costs or clinical recommendations.")

elif page == L["fair"]:
    st.markdown(f'<div class="hero"><div class="eyebrow">MODEL AUDIT</div><h1>{L["fair_title"]}</h1><p>{L["fair_desc"]}</p></div>',unsafe_allow_html=True)
    dimension_map={"Age group":"Age group","Sex group":"Sex group","Income group":"Income group","Education group":"Education group"}
    dimension=st.selectbox(L["dimension"],list(dimension_map.keys()))
    sub=FAIR[FAIR.dimension==dimension].copy()
    rename={"group":"Group", "n":L["n"],"sensitivity":L["sens"],"specificity":L["spec"],
            "precision":L["prec"],"positive_rate":L["pos"],"predicted_positive_rate":L["predpos"]}
    show=sub[["group","n","sensitivity","specificity","precision","positive_rate","predicted_positive_rate"]].rename(columns=rename)
    for c in [L["sens"],L["spec"],L["prec"],L["pos"],L["predpos"]]:
        show[c]=show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(show,use_container_width=True,hide_index=True)
    plot=sub.melt(id_vars=["group"],value_vars=["sensitivity","specificity"],var_name="Metric",value_name="Score")
    fig=px.bar(plot,x="group",y="Score",color="Metric",barmode="group",range_y=[0,1],
               labels={"group":"Group","Score":"Score"})
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",legend_title_text="")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Subgroup metrics are descriptive. Differences do not by themselves establish bias or fairness; consider subgroup sample sizes, uncertainty, prevalence, and the consequences of errors.")

else:
    st.markdown(f'<div class="hero"><div class="eyebrow">TRANSPARENCY</div><h1>{L["about_title"]}</h1><p>{L["intended_text"]}</p></div>',unsafe_allow_html=True)
    st.subheader(L["intended"]); st.write(L["intended_text"])
    st.subheader(L["data"])
    st.write("Diabetes Health Indicators, CDC BRFSS 2015 (Kaggle) — binary target dataset. The project uses the original, imbalanced file (not the 50/50 balanced variant).")
    st.write(f"Rows: {253680:,} · Original features: 21 predictors · Model inputs: {len(META['features'])} predictors (full questionnaire).")
    st.subheader(L["metrics"])
    m=META.get("metrics") or {}
    if not m or m.get("roc_auc") is None:
        st.info("Run `python train.py` to generate the model and held-out evaluation metrics.")
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
        st.caption(f"Decision threshold used for these classification metrics: {m['threshold']:.3f} (FN cost {META['cost_assumptions']['false_negative']:.1f}; FP cost {META['cost_assumptions']['false_positive']:.1f}).")
    st.subheader(L["limitations"])
    for item in L["limitations_list"]: st.markdown(f"- {item}")
    st.subheader(L["disclaimer"]); st.warning(L["disclaimer_text"])
    st.subheader("Model card")
    st.markdown(f"**Model:** {META['model']}  \n**Target:** self-reported diabetes status (binary survey label)  \n**Evaluation:** {L['split_text']}  \n**Fairness audit:** age, sex, income and education groups. These characteristics are included in this expanded questionnaire model; subgroup metrics are descriptive.")
    st.caption("Source and evaluation details are documented in README.md and models/model_meta.json.")

st.markdown("---")
st.caption("DiaCheck · T3 Healthcare Decision-Support App · " + L["footer"])
