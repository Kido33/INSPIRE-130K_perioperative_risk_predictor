import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import (SimpleImputer, KNNImputer, IterativeImputer)
from sklearn.ensemble import (ExtraTreesRegressor)
from scipy.stats.mstats import winsorize
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 경로 및 전략 설정
# ==========================================
BASE_PATH = "/home/kido/final/INSPIRE 130K/"
SAVE_PATH = './preprocessed_data'
os.makedirs(SAVE_PATH, exist_ok=True)

# 진료과별 핵심 감시 지표 및 위기(Crisis) 판단 임계치 정의
CLINICAL_STRATEGY = {
    'UR': {
        'target': 'creatinine', 
        'method': 'LOCF',
        'threshold_type': 'increase', 
        'threshold_val': 0.3          
    },
    'CTS': {
        'target': 'troponin_i', 
        'method': 'Iterative_ET', # RF보다 빠른 ExtraTrees로 성능은 유지하며 속도 향상
        'threshold_type': 'ratio',    
        'threshold_val': 3.0          
    },
    'GS': {
        'target': 'crp', 
        'method': 'KNN', # KNN 유지하되 내부에서 청크 처리
        'threshold_type': 'increase', 
        'threshold_val': 10.0         
    },
    'OS': {
        'target': 'hb', 
        'method': 'Linear',
        'threshold_type': 'decrease', 
        'threshold_val': 2.0          
    },
    'NS': {
        'target': 'sodium', 
        'method': 'Median',
        'threshold_type': 'abs_diff', 
        'threshold_val': 5.0          
    }
}

# ==========================================
# 2. 전처리 핵심 함수들 (청크 처리 로직 포함)
# ==========================================

def handle_outliers(series):
    if series.isnull().all(): return series
    return winsorize(series, limits=[0.01, 0.01]).data

def apply_imputation(df_subset, target_col, method):
    """
    대규모 데이터를 Chunk 단위로 나누어 KNN/Iterative 연산을 수행함으로서 
    메모리 Swap을 방지하고 성능을 유지함.
    """
    if target_col not in df_subset.columns:
        return np.array([np.nan] * len(df_subset))
    
    # 1. 단순 보간법은 즉시 처리
    if method == 'Median':
        return SimpleImputer(strategy='median').fit_transform(df_subset[[target_col]]).flatten()
    elif method == 'LOCF':
        return df_subset[target_col].ffill().bfill().values
    elif method == 'Linear':
        return df_subset[target_col].interpolate(method='linear').ffill().bfill().values

    # 2. 고급 보간법 (KNN, Iterative) - 청크 분할 처리
    # 23GiB RAM 환경에서 최적의 청크 사이즈: 2000~3000
    chunk_size = 2500
    num_cols = df_subset.select_dtypes(include=[np.number]).columns
    data_all = df_subset[num_cols].copy()
    target_idx = list(data_all.columns).index(target_col)
    
    imputed_values = []
    
    # 데이터를 청크별로 나누어 학습 및 보간 (메모리 내 연산 보장)
    for i in range(0, len(data_all), chunk_size):
        chunk = data_all.iloc[i : i + chunk_size].copy()
        
        # 청크 내 모든 값이 결측치인 경우 방어
        if chunk[target_col].isnull().all():
            chunk[target_col] = chunk[target_col].fillna(0) # 혹은 전체 중앙값

        if method == 'KNN':
            # 상관관계 기반 KNN (성능 중시)
            imputer = KNNImputer(n_neighbors=5)
            chunk_imputed = imputer.fit_transform(chunk)
            imputed_values.extend(chunk_imputed[:, target_idx])
            
        elif method == 'Iterative_ET':
            # RF보다 연산 효율이 좋은 ExtraTrees 기반 다중 보간 (성능 확보)
            et = ExtraTreesRegressor(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
            imputer = IterativeImputer(estimator=et, max_iter=5, random_state=42)
            chunk_imputed = imputer.fit_transform(chunk)
            imputed_values.extend(chunk_imputed[:, target_idx])

    return np.array(imputed_values[:len(df_subset)])

def define_target_by_dept(group, dept):
    strategy = CLINICAL_STRATEGY.get(dept)
    if not strategy: 
        group['target_event'] = 0
        return group
    
    target_base_name = strategy['target']
    baseline_col = f"preop_{target_base_name}"
    current_col = f"postop_{target_base_name}_max" if strategy['threshold_type'] in ['increase', 'ratio', 'abs_diff'] else f"postop_{target_base_name}_min"
    
    if baseline_col not in group.columns or current_col not in group.columns:
        group['target_event'] = 0
        return group

    # 수술 전 데이터 보간 (청크 분할 알고리즘 적용)
    group[baseline_col] = apply_imputation(group, baseline_col, strategy['method'])
    
    baseline = group[baseline_col]
    current = group[current_col]
    val = strategy['threshold_val']
    
    # 타겟 레이블링
    if strategy['threshold_type'] == 'increase':
        group['target_event'] = ((current - baseline) >= val).astype(int)
    elif strategy['threshold_type'] == 'decrease':
        group['target_event'] = ((baseline - current) >= val).astype(int)
    elif strategy['threshold_type'] == 'ratio':
        group['target_event'] = ((current / (baseline + 1e-6)) >= val).astype(int)
    elif strategy['threshold_type'] == 'abs_diff':
        group['target_event'] = ((current - baseline).abs() >= val).astype(int)
    
    return group

def calculate_clinical_deltas(group, dept):
    strategy = CLINICAL_STRATEGY.get(dept)
    if not strategy: return group
    target_name = strategy['target']
    
    baseline_col = f"preop_{target_name}"
    current_col = f"postop_{target_name}_max" if strategy['threshold_type'] != 'decrease' else f"postop_{target_name}_min"
    
    if baseline_col in group.columns and current_col in group.columns:
        if dept == 'UR': group['delta_feature'] = group[current_col] - group[baseline_col]
        elif dept == 'CTS': group['delta_feature'] = group[current_col] / (group[baseline_col] + 1e-6)
        elif dept == 'GS': group['delta_feature'] = group[current_col] - group[baseline_col]
        elif dept == 'OS': group['delta_feature'] = group[baseline_col] - group[current_col]
        elif dept == 'NS': group['delta_feature'] = (group[current_col] - group[baseline_col]).abs()
    
    if 'delta_feature' not in group.columns: group['delta_feature'] = 0
    return group

def process_dept_worker(dept, group):
    # 1. 이상치 처리
    num_cols = group.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        group[col] = handle_outliers(group[col])
    
    # 2. 보간 적용 및 타겟 레이블링
    if dept in CLINICAL_STRATEGY:
        group = define_target_by_dept(group, dept)
        group = calculate_clinical_deltas(group, dept)
    
    return group

# ==========================================
# 3. 메인 실행 프로세스
# ==========================================

def run_step1():
    print("🚀 [Step 1] 데이터 로딩 및 고도화된 전처리 가동...")
    df_ops = pd.read_parquet(os.path.join(BASE_PATH, "operations.parquet"))
    labs_df = pd.read_parquet(os.path.join(BASE_PATH, "labs.parquet"))
    
    # [Lab 데이터 분리 및 요약]
    print("📊 Lab 데이터 시점별 요약 중...")
    preop_labs = labs_df[labs_df['chart_time'] <= 0].sort_values('chart_time').groupby(['subject_id', 'item_name']).tail(1)
    preop_summary = preop_labs.pivot_table(index='subject_id', columns='item_name', values='value', aggfunc='last')
    preop_summary.columns = [f"preop_{c}".lower().replace(' ', '_') for c in preop_summary.columns]

    postop_summary = labs_df[labs_df['chart_time'] > 0].pivot_table(
        index='subject_id', columns='item_name', values='value', aggfunc=['max', 'min']
    )
    postop_summary.columns = [f"postop_{c[1]}_{c[0]}".lower().replace(' ', '_') for c in postop_summary.columns]
    
    df_master = df_ops.merge(preop_summary, on='subject_id', how='left')
    df_master = df_master.merge(postop_summary, on='subject_id', how='left')

    # [진료과별 병렬 전처리]
    print(f"⚡ 진료과별 고성능 보간(KNN/ET) 및 타겟 산출 시작 (RAM 여유: {SAVE_PATH})...")
    target_depts = ['GS', 'OS', 'NS', 'CTS', 'UR']
    df_filtered = df_master[df_master['department'].isin(target_depts)].copy()
    
    # 병렬 처리 시 loky 백엔드 사용 (메모리 안정성)
    processed_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_dept_worker)(d, g.copy()) for d, g in tqdm(df_filtered.groupby('department'))
    )
    
    df_final = pd.concat(processed_results).sort_values('op_id').reset_index(drop=True)
    
    if 'delta_feature' in df_final.columns:
        df_final['delta_feature'] = df_final['delta_feature'].fillna(0)
    
    # 최종 저장
    df_final.to_parquet(f"{SAVE_PATH}/df_master_preprocessed.parquet", index=False)
    
    print(f"✅ Step 1 완료: {len(df_final)}명 데이터 정제 완료.")
    print(f"📊 타겟 이벤트 발생 현황:\n{df_final['target_event'].value_counts()}")

if __name__ == "__main__":
    run_step1()