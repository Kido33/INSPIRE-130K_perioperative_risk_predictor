import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, 
                             brier_score_loss, f1_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# 경고 제어
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_step4():
    print("🚀 [Step 4] 통합 검증, SHAP 분석 및 모델 아카이빙 시작...")
    
    # 1. 설정 및 데이터 로드
    DATA_PATH = "./preprocessed_data/df_final_features.csv"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "df_final_features.csv" if os.path.exists("df_final_features.csv") else None
    
    if DATA_PATH is None:
        print("❌ 데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 저장 경로 설정 (최종 아카이브)
    BASE_DIR = './mfds_final_archive'
    MODEL_DIR = f'{BASE_DIR}/models'
    PLOT_DIR = f'{BASE_DIR}/plots'
    REPORT_DIR = f'{BASE_DIR}/reports'
    for d in [MODEL_DIR, PLOT_DIR, REPORT_DIR]: os.makedirs(d, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    all_final_results = []
    target_depts = ['CTS', 'UR', 'NS', 'GS', 'OS']

    # 통합 시각화를 위한 설정
    fig_roc, axes_roc = plt.subplots(2, 3, figsize=(18, 12))
    axes_roc = axes_roc.flatten()

    for i, dept in enumerate(target_depts):
        dept_df = df[df['department'].str.upper() == dept].copy()
        if len(dept_df) < 50:
            print(f"⚠️ [{dept}] 데이터 부족으로 스킵 (N={len(dept_df)})")
            continue

        print(f"\n⚙️ [{dept}] 프로세스 시작 (N={len(dept_df)})...")

        # --- [특수 전처리] GS 진료과 CRP 로그 변환 (코드 통합) ---
        if dept == 'GS' and 'preop_crp' in dept_df.columns:
            print(f"🪄 [{dept}] preop_crp 로그 변환(log1p) 적용...")
            dept_df['preop_crp'] = np.log1p(dept_df['preop_crp'].fillna(0))

        # 2. 피처 준비 (심근효소 등 강건성 저해 변수 제외)
        exclude_feats = ['preop_troponin_i', 'preop_ckmb', 'preop_ck']
        feats = [c for c in dept_df.columns if ('preop_' in c or c in ['age', 'asa', 'TS_Prob', 'emop', 'bmi']) 
                 and c not in exclude_feats]
        
        X = dept_df[feats].fillna(0)
        y = dept_df['target_event']
        
        # 데이터 분할 (Train 60%, Calib 20%, Test 20%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # 3. 모델 학습
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 4. 확률 보정 (Calibration) - 과별 최적 알고리즘 적용
        raw_calib_probs = model.predict_proba(X_calib)[:, 1]
        raw_test_probs = model.predict_proba(X_test)[:, 1]

        if dept == 'UR': # 비뇨의학과는 비모수적 보정(Isotonic)이 효과적
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(raw_calib_probs, y_calib)
            cal_calib_probs = calibrator.transform(raw_calib_probs)
            cal_test_probs = calibrator.transform(raw_test_probs)
        else: # 그 외 과는 Platt Scaling(Logistic) 적용
            calibrator = LogisticRegression(C=np.inf, solver='lbfgs')
            calibrator.fit(raw_calib_probs.reshape(-1, 1), y_calib)
            cal_calib_probs = calibrator.predict_proba(raw_calib_probs.reshape(-1, 1))[:, 1]
            cal_test_probs = calibrator.predict_proba(raw_test_probs.reshape(-1, 1))[:, 1]

        # 5. 유덴 지수(Youden's Index) 기반 최적 임계값 설정
        fpr, tpr, thresholds = roc_curve(y_calib, cal_calib_probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        
        # 6. 최종 성능 평가
        preds = (cal_test_probs >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        auc_score = roc_auc_score(y_test, cal_test_probs)
        
        res = {
            "Dept": dept,
            "Strategy": "Youden-Index",
            "Best_Threshold": round(best_threshold, 4),
            "AUC": round(auc_score, 4),
            "Sensitivity": round(tp/(tp+fn), 4),
            "Specificity": round(tn/(tn+fp), 4),
            "PPV": round(tp/(tp+fp), 4) if (tp+fp)>0 else 0,
            "NPV": round(tn/(tn+fn), 4) if (tn+fn)>0 else 0,
            "Brier_Score": round(brier_score_loss(y_test, cal_test_probs), 4),
            "F1_Score": round(f1_score(y_test, preds), 4)
        }
        all_final_results.append(res)

        # 7. 상세 SHAP 분석 (판단 근거 시각화)
        print(f"🧬 [{dept}] SHAP 계산 중...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        
        # RF 차원에 따른 SHAP 값 추출 (Class 1 기준)
        current_shap = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(current_shap, X_test, max_display=15, show=False)
        plt.title(f"Feature Importance (SHAP): {dept}")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/shap_{dept}.png")
        plt.close()

        # 8. 개별 ROC 커브 시각화 (통합용)
        test_fpr, test_tpr, _ = roc_curve(y_test, cal_test_probs)
        axes_roc[i].plot(test_fpr, test_tpr, label=f'AUC = {auc_score:.3f}', color='red', lw=2)
        axes_roc[i].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes_roc[i].scatter(1-(tn/(tn+fp)), tp/(tp+fn), color='black', s=60, label='Youden Pt', zorder=5)
        axes_roc[i].set_title(f"ROC Curve: {dept}")
        axes_roc[i].legend(loc='lower right')

        # 9. 모델 패키징 및 저장
        model_pack = {
            'model': model, 
            'calibrator': calibrator, 
            'threshold': best_threshold, 
            'features': feats, 
            'stats': res
        }
        joblib.dump(model_pack, f"{MODEL_DIR}/model_final_{dept}.pkl")
        print(f"💾 [{dept}] 모델 및 보정기 저장 완료.")

    # 10. 최종 통합 결과 정리
    if len(target_depts) < 6: fig_roc.delaxes(axes_roc[5])
    fig_roc.tight_layout()
    fig_roc.savefig(f"{REPORT_DIR}/integrated_roc_analysis.png", dpi=300)
    plt.close()

    final_df = pd.DataFrame(all_final_results)
    final_df.to_csv(f"{REPORT_DIR}/final_mfds_performance_report.csv", index=False)
    
    # 최종 성능 요약 차트
    plt.figure(figsize=(14, 7))
    melt_df = final_df.melt(id_vars='Dept', value_vars=['Sensitivity', 'Specificity', 'AUC'])
    sns.barplot(data=melt_df, x='Dept', y='value', hue='variable', palette='muted')
    plt.title("Clinical Performance Summary (Youden Optimized)")
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{REPORT_DIR}/performance_summary_bar.png")
    
    print("\n" + "="*80)
    print("✅ 모든 진료과 분석, SHAP 시각화 및 모델 아카이빙 완료")
    print(final_df.to_string(index=False))
    print(f"\n📂 결과물 경로: {BASE_DIR}")
    print("="*80)
    plt.show()

if __name__ == "__main__":
    run_step4()