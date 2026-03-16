import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, 
                             precision_score, f1_score, roc_curve, confusion_matrix)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

def run_step3():
    # 1. 데이터 로드 및 경로 설정
    print("🚀 [Step 3-1] 최종 통합 데이터셋 로딩...")
    DATA_PATH = "./preprocessed_data/df_final_features.csv"
    REPORT_DIR = './surgical_crisis_dept_reports'
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 에러: {DATA_PATH} 파일이 없습니다. Step 2를 먼저 완료하세요.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 진료과 리스트 추출
    target_depts = df['department'].unique()
    print(f"🏥 분석 대상 진료과: {target_depts}")

    # 모든 진료과 성적을 저장할 리스트
    all_dept_results = []

    # ========================================================
    # 2. 진료과별 루프 시작
    # ========================================================
    for dept in target_depts:
        print(f"\n{'='*50}\n🔥 [{dept}] 진료과 특이적 모델 학습 및 평가 시작\n{'='*50}")
        
        dept_df = df[df['department'] == dept].copy()
        if len(dept_df) < 100: # 데이터가 너무 적으면 스킵
            print(f"⚠️ {dept}과는 데이터 수가 너무 적어 제외합니다.")
            continue

        # 피처 선정 (Step 1에서 생성된 preop_ 컬럼들 포함)
        potential_base = ['age', 'asa', 'emop', 'bmi', 'cci']
        base_info = [c for c in potential_base if c in dept_df.columns]
        preop_lab_feats = [c for c in dept_df.columns if 'preop_' in c] 
        ts_feat = ['TS_Prob']
        feats = list(set(base_info + preop_lab_feats + ts_feat))

        X = dept_df[feats].fillna(0)
        y = dept_df['target_event']
        
        # 특수문자 방지
        X.columns = [str(c).replace('[', '').replace(']', '').replace('<', '') for c in X.columns]
        
        # 데이터 분할 (진료과 내에서 Train/Test 분리)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y if y.nunique() > 1 else None, random_state=42
        )

        # 모델 풀 구성 (불균형 보정 포함)
        pos_count = (y == 1).sum()
        neg_count = (y == 0).sum()
        ratio = neg_count / (pos_count + 1e-6)

        model_pool = {
            "CatBoost": CatBoostClassifier(iterations=200, auto_class_weights='Balanced', verbose=0, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, scale_pos_weight=ratio, eval_metric='logloss', random_state=42),
            "LGBM": LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            "LogisticReg": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        }

        dept_results = []
        plt.figure(figsize=(10, 8))
        custom_threshold = y_train.mean() # 진료과별 양성 비율에 따른 임계값

        for name, model in model_pool.items():
            try:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                preds = (probs > custom_threshold).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
                auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
                
                res = {
                    "Dept": dept,
                    "Model": name,
                    "AUC": auc,
                    "Accuracy": accuracy_score(y_test, preds),
                    "Sensitivity": recall_score(y_test, preds),
                    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                    "F1_Score": f1_score(y_test, preds)
                }
                dept_results.append(res)
                all_dept_results.append(res)
                
                fpr, tpr, _ = roc_curve(y_test, probs)
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
                
            except Exception as e:
                print(f"⚠️ {dept}-{name} 학습 중 오류: {e}")

        # 진료과별 시각화 저장
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve for {dept}')
        plt.legend()
        plt.savefig(f"{REPORT_DIR}/roc_{dept}.png")
        plt.close()

        # 진료과별 중간 성적 출력
        dept_perf = pd.DataFrame(dept_results).sort_values(by='AUC', ascending=False)
        print(f"\n🏆 [{dept}] 성적표:")
        print(dept_perf.to_string(index=False))

    # 3. 전체 요약 리포트 저장
    final_perf_df = pd.DataFrame(all_dept_results)
    final_perf_df.to_csv(f"{REPORT_DIR}/all_dept_performance_report.csv", index=False)
    
    print(f"\n✅ 모든 진료과 분석 완료! 리포트 확인: {REPORT_DIR}")

if __name__ == "__main__":
    run_step3()