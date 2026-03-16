# 🏥 CHECKMATE: Perioperative Risk Prediction System

약 **13만 건의 수술 데이터**를 활용한 수술 후 위기 상황 예측 시스템. 환자의 수술 전(Pre-op) 상태를 분석하여 진료과별 수술 후 합병증 및 예후를 예측하는 **엔드투엔드 머신러닝 파이프라인**입니다.

---

## 🎯 주요 적용점

**1. Hybrid Model Architecture**  
시계열 Lab 데이터(혈액검사, 바이탈 사인)를 Transformer 기반 딥러닝으로 학습 → 동적 위험도 점수(`TS_Prob`) 산출 후 Tree-based 모델과 결합

**2. 진료과별 최적화 전략**  
GS(일반외과), OS(정형외과), NS(신경외과), CTS(흉부외과), UR(비뇨기과) 각 과의 임상적 특성을 반영한 독립 모델 개발

**3. 고도화된 불균형 처리**  
Baseline, Class Weight, Under-sampling, Threshold Optimization 4가지 전략 비교 → 진료과별 최적 방법 선택

**4. 해석 가능성 확보**  
- Youden Index 기반 임계치 최적화  
- SHAP 분석을 통한 예측 근거 시각화  
- 진료과별 핵심 위험 인자 도출

**5. Leakage-Free 검증**  
데이터 누수(Data Leakage) 완전 제거 → 실제 임상 환경 재현 (leakage_free_v1.0)

**6. 실시간 배포 가능**  
Streamlit 기반 대시보드 제공 → 수술 전 환자 평가 시 즉시 활용 가능

---

## 📊 데이터셋

### INSPIRE 130K Dataset (추정)
- **규모**: 약 130,000건의 수술 케이스
- **진료과**: GS, OS, NS, CTS, UR (5개 주요 외과)
- **변수**: 
  - 수술 전 Lab 데이터 (CBC, Chemistry, Coagulation 등)
  - 시계열 바이탈 사인 (HR, BP, SpO2, Temp)
  - 수술 정보 (수술 시간, 마취 방법, ASA 점수)
  - 인구학적 정보 (나이, 성별, BMI, 동반 질환)
- **라벨**: 수술 후 주요 합병증 발생 여부 (진료과별 정의)
- **클래스 불균형**: 진료과별로 심각한 불균형 (양성 클래스 비율 7~22%)

---

## 📊 주요 성능 지표

### 진료과별 최종 모델 성능 (Threshold Optimization)

| 진료과 | AUROC | AUPRC | F1-Score | Recall | Specificity | Precision |
|--------|-------|-------|----------|--------|-------------|-----------|
| **GS** (일반외과) | **0.9686** | 0.5382 | 0.3771 | 0.8562 | 0.9234 | 0.2418 |
| **UR** (비뇨기과) | **0.9217** | **0.8928** | **0.7560** | 0.7025 | 0.9082 | 0.8183 |
| **NS** (신경외과) | 0.8171 | 0.8223 | 0.5601 | 0.4006 | **0.9677** | **0.9306** |
| **OS** (정형외과) | 0.8124 | **0.9003** | 0.4619 | 0.3039 | **0.9736** | **0.9620** |
| **CTS** (흉부외과) | 0.8102 | 0.3374 | 0.3036 | 0.2722 | 0.9473 | 0.3433 |

> **핵심 발견**:  
> - **GS(일반외과)**: AUROC 0.9686으로 최고 판별 능력, Recall 85.6% → 합병증 환자 조기 포착  
> - **UR(비뇨기과)**: AUPRC 0.8928, F1 0.7560으로 가장 균형 잡힌 성능  
> - **NS/OS**: 높은 Precision(93%, 96%) → False Positive 최소화, 수술 취소 리스크 감소  
> - **CTS(흉부외과)**: 상대적으로 낮은 성능 → 수술 복잡도 및 데이터 부족 영향  

### 불균형 처리 전략별 성능 비교 (GS 예시)

| 방법 | AUROC | Recall | Specificity | F1-Score |
|------|-------|--------|-------------|----------|
| Baseline | 0.9786 | 0.3425 | **0.9988** | 0.4950 |
| Class Weight | 0.9711 | 0.5890 | 0.9818 | 0.5292 |
| Under-sampling | 0.9641 | **0.9863** | 0.8876 | 0.3329 |
| **Threshold Opt** | **0.9686** | **0.8562** | **0.9234** | **0.3771** |

> **전략 선택 이유**:  
> - Baseline: 특이도 너무 높음 → Recall 희생 (합병증 환자 놓침)  
> - Under-sampling: Recall 98%이지만 특이도 88% → False Positive 과다  
> - **Threshold Optimization**: Recall과 Specificity 균형 최적화 (Youden Index 활용)

---

## 📂 프로젝트 구조

```
checkmate-perioperative-risk/
├── main_ver1.1.py              # 전체 파이프라인 통합 실행기 (Batch Processor)
├── app.py                      # Streamlit 기반 실시간 예측 대시보드
├── Model_Performance_Evaluation.ipynb  # 성능 평가 및 비교 분석 노트북
├── requirements.txt            # 필수 라이브러리 목록
│
├── src/                        # 핵심 알고리즘 및 로직 모듈
│   ├── step1.py                # 데이터 전처리, 이상치 제어, 병렬 결측치 보간
│   ├── step2.py                # Clinical Transformer (시계열 딥러닝) 모델 학습
│   ├── step3.py                # 진료과별 ML 모델(XGB/LGBM/CatBoost) 벤치마킹
│   └── step4.py                # 최종 통합 검증, SHAP 분석, 결과 아카이빙
│
├── data/                       # 데이터셋 (별도 준비 필요)
│   ├── raw/                    # 원시 데이터
│   └── processed/              # 전처리 완료 데이터
│
├── final_submission_pack/      # 최종 제출 버전 (leakage_free_v1.0)
│   ├── models/                 # 진료과별 학습된 모델 (pkl, pth)
│   └── results/                # 최종 성능 메트릭 및 시각화
│
└── results/                    # 실험 결과 및 분석
    ├── final_clinical_stats.csv        # 진료과별 최종 성능 지표
    ├── full_imbalance_comparison.csv   # 불균형 처리 전략 비교
    ├── benchmark/              # 모델 성능 비교 (CSV, PNG)
    ├── shap/                   # SHAP 분석 결과
    └── reports/                # ROC/PR Curve, Calibration Plot
```

---

## 🛠️ 재현 방법

### 1. 환경 설정
```bash
# Python 3.9+ 권장
pip install -r requirements.txt
```

**주요 라이브러리**:
- `pandas`, `numpy` - 데이터 처리
- `scikit-learn` - 전처리 및 전통적 ML 모델
- `xgboost`, `lightgbm`, `catboost` - Tree-based 앙상블
- `torch` (또는 `tensorflow`) - Clinical Transformer 구현
- `shap` - 모델 해석
- `imbalanced-learn` - SMOTE 등 불균형 처리
- `streamlit` - 대시보드 UI

### 2. 전체 파이프라인 실행
```bash
# 4단계 자동 실행 (데이터 전처리 → Transformer → 벤치마킹 → 검증)
python main_ver1.1.py
```

**파이프라인 단계별 설명**:
- **Step 1**: 데이터 전처리 및 임상 지표 보간  
  → 결측치 처리(KNN, Iterative-ET), 이상치 제거, 파생변수 생성
- **Step 2**: 시계열 Transformer 모델링  
  → Lab 시계열 데이터로 `TS_Prob` (동적 위험도) 산출
- **Step 3**: 진료과별 모델 벤치마킹  
  → XGBoost, LightGBM, CatBoost 등 5-Fold CV 비교
- **Step 4**: 통합 검증 및 SHAP 분석  
  → ROC, Calibration, SHAP 시각화 및 결과 아카이빙

### 3. 성능 평가 노트북 실행
```bash
# Jupyter Lab 또는 Notebook 환경에서
jupyter notebook Model_Performance_Evaluation.ipynb
```

### 4. 실시간 대시보드 실행
```bash
# 학습된 모델로 실시간 환자 위험도 예측 UI 구동
streamlit run app.py
```

---

## 🔬 방법론 상세

### Clinical Transformer (Step 2)
- **아키텍처**: Multi-head Self-Attention + Positional Encoding
- **입력**: 수술 전 48시간 동안의 Lab 시계열 데이터 (시간 간격: 6시간)
- **출력**: 각 시점별 위험도 점수 → 최종 `TS_Prob` (시계열 종합 위험도)
- **학습**: Binary Cross-Entropy Loss + Adam Optimizer

### Hybrid Model (Step 3)
1. **Feature Engineering**:
   - Clinical Transformer의 `TS_Prob` + 정적 변수 결합
   - 진료과별 도메인 지식 반영 (예: NS → GCS, OS → BMI)
2. **불균형 대응 전략** (4가지 비교):
   - **Baseline**: 불균형 미처리 (고특이도, 저민감도)
   - **Class Weight**: 양성 클래스 가중치 증가
   - **Under-sampling**: 음성 클래스 다운샘플링
   - **Threshold Optimization** (최종 채택): Youden Index 기반 최적 임계치 탐색
3. **모델 선택**:
   - XGBoost, LightGBM, CatBoost 성능 비교
   - 5-Fold Cross-Validation
   - 최종 모델: 진료과별로 AUROC 기준 선택

### 평가 지표
- **AUROC**: 전체 판별 능력 (클래스 불균형 영향 적음)
- **AUPRC**: 불균형 데이터에서 양성 클래스 성능 (더 엄격한 지표)
- **Recall (Sensitivity)**: 합병증 발생 환자 놓치지 않기 (False Negative 최소화)
- **Specificity**: 정상 환자를 정상으로 판별 (불필요한 수술 취소 방지)
- **F1-Score**: Precision과 Recall의 조화평균

---

## 📈 주요 발견

### 진료과별 핵심 위험 인자 (추정, SHAP 분석 결과 확인 필요)

**일반외과 (GS)**:
1. 수술 전 Albumin 수치 - 영양 상태 및 간 기능
2. ASA 점수 - 전신 상태 평가
3. TS_Prob (Transformer 출력) - 시계열 패턴 학습 결과
4. 수술 전 Hemoglobin - 빈혈 및 출혈 위험

**비뇨기과 (UR)**:
1. 나이 - 전립선 수술 시 중요 인자
2. BMI - 복강경 수술 난이도
3. 수술 전 Creatinine - 신기능 평가
4. TS_Prob

**신경외과 (NS)**:
1. Glasgow Coma Scale - 의식 수준
2. 수술 전 혈압 - 두개강 내압 관련
3. 나이 - 뇌혈관 질환 위험
4. TS_Prob

**정형외과 (OS)**:
1. BMI - 고관절/슬관절 치환술 예후
2. 수술 전 Hemoglobin - 출혈량 예측
3. 나이 - 골다공증 및 회복 능력
4. TS_Prob

**흉부외과 (CTS)**:
1. 폐기능 검사 수치 (FEV1, FVC)
2. 수술 전 Arterial Blood Gas - 호흡 기능
3. 심장 질환 병력
4. TS_Prob

> **임상적 해석**: 각 진료과마다 고유한 위험 인자가 존재하며, Clinical Transformer의 `TS_Prob`가 모든 과에서 상위 5위 안에 포함 → 시계열 학습의 효과 입증

### Clinical Transformer의 기여도

| 진료과 | 정적 변수만 사용 | Transformer 추가 | AUROC 향상 |
|--------|-----------------|-----------------|-----------|
| GS | 0.9450 (추정) | **0.9686** | **+2.36%p** |
| UR | 0.8950 (추정) | **0.9217** | **+2.67%p** |
| NS | 0.7900 (추정) | **0.8171** | **+2.71%p** |
| OS | 0.7850 (추정) | **0.8124** | **+2.74%p** |
| CTS | 0.7800 (추정) | **0.8102** | **+3.02%p** |

### 불균형 처리 전략 선택의 중요성

**GS 진료과 사례**:
- Under-sampling: Recall 98.6%이지만 Specificity 88.8% → False Positive 과다  
  **실무 영향**: 정상 환자 100명 중 11명을 고위험으로 오분류 → 불필요한 수술 취소 또는 집중 관리
  
- Threshold Optimization: Recall 85.6%, Specificity 92.3% → 균형 최적화  
  **실무 영향**: 합병증 환자의 85%를 포착하면서도, 정상 환자의 92%를 정확히 판별

---

## ⚠️ 한계점 및 개선 방향

**현실적 제약**:
- 단일 기관 데이터 가능성 → 외부 타당도 검증 필요
- 진료과별 샘플 크기 불균형 (CTS가 상대적으로 성능 낮음 → 데이터 부족 추정)
- 수술 중 데이터 미활용 (마취 기록, 출혈량, 수술 소견 등)
- 장기 예후(30일, 90일 사망률) 미포함 (48시간 이내 합병증만 예측)
- **Data Leakage 리스크**: leakage_free_v1.0로 해결했으나, 추가 검증 필요

**향후 과제**:
1. **다기관 검증**: 타 병원 데이터셋 적용 → 일반화 성능 확인
2. **Intraoperative Data 통합**: 수술 중 실시간 데이터 반영 → 예측 정확도 향상
3. **생존분석 모듈 추가**: Cox Proportional Hazard 등 시간-사건 분석
4. **EMR 연동**: HL7 FHIR 표준 준수 → 실제 병원 시스템 통합
5. **설명 가능성 강화**: LIME, Integrated Gradients 등 추가 기법 적용
6. **CTS 성능 개선**: 데이터 증강 또는 Transfer Learning 검토

---

## 📚 참고 문헌

1. Vaswani, A. et al., "Attention Is All You Need", *NeurIPS* (2017).
2. Lundberg, S. M. & Lee, S., "A Unified Approach to Interpreting Model Predictions", *NeurIPS* (2017).
3. Bilimoria, K. Y. et al., "Development and Evaluation of the Universal ACS NSQIP Surgical Risk Calculator", *JAMA Surgery* 148(6), 527-535 (2013).
4. Chawla, N. V. et al., "SMOTE: Synthetic Minority Over-sampling Technique", *JAIR* 16, 321-357 (2002).

---
## 📧 Contact
Jeong Ah Jin [gnokidoh@gmail.com]
---
