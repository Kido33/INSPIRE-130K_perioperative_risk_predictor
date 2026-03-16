import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 환경 설정 및 경로 확인
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "/home/kido/final/INSPIRE 130K/"
PREP_DATA_PATH = './preprocessed_data/df_master_preprocessed.parquet'
MODEL_SAVE_PATH = './saved_models'
FINAL_SAVE_PATH = './preprocessed_data'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ==========================================
# 2. Clinical Transformer 모델 정의
# ==========================================
class ClinicalTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=8, seq_len=12):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        # Global Average Pooling (시퀀스 차원의 평균 사용)
        return self.fc(x.mean(dim=1))

# ==========================================
# 3. 데이터 로딩 및 시계열 처리 엔진
# ==========================================
def run_step2():
    print("🚀 [Step 2] 데이터 로드 및 텐서 생성 시작...")
    
    # Step 1 결과 로드
    df_master = pd.read_parquet(PREP_DATA_PATH)
    # 반드시 op_id 순서대로 정렬되어 있는지 재확인
    df_master = df_master.sort_values('op_id').reset_index(drop=True)
    ordered_op_ids = df_master['op_id'].tolist()
    
    # 바이탈 데이터 로드 (파일명 확인 필요)
    vitals = pd.read_parquet(os.path.join(BASE_PATH, "vitals-005.parquet"))
    vitals['hour'] = (vitals['chart_time'] // 60).astype(int)
    
    # 수술 전 12시간 데이터 필터링 (-11시 ~ 0시)
    vitals_filtered = vitals[(vitals['hour'] >= -11) & (vitals['hour'] <= 0)]
    
    required_vitals = ['hr', 'sbp', 'dbp', 'spo2', 'rr']
    vitals_filtered = vitals_filtered[vitals_filtered['item_name'].isin(required_vitals)]

    # 피벗 테이블 생성
    pivot = vitals_filtered.pivot_table(index=['op_id', 'hour'], columns='item_name', values='value', aggfunc='mean')

    # 없는 컬럼 생성
    for col in required_vitals:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # 모든 환자 x 12시간 격자 재구성 (데이터 누락 환자 포함)
    grid_idx = pd.MultiIndex.from_product([ordered_op_ids, np.arange(-11, 1)], names=['op_id', 'hour'])
    full_vitals = pivot.reindex(grid_idx)

    # 결측치 보간 (임상적 정상 수치 기준)
    default_values = {'hr': 80, 'sbp': 120, 'dbp': 80, 'spo2': 98, 'rr': 18}
    for col in required_vitals:
        full_vitals[col] = full_vitals[col].fillna(default_values[col])

    # [중요] 데이터 스케일링 (학습 속도 및 AUC 향상)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(full_vitals[required_vitals])
    
    # 텐서 변환 (Batch, Seq_len, Features)
    X_data = scaled_values.reshape(len(ordered_op_ids), 12, 5)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(df_master['target_event'].values, dtype=torch.float32).view(-1, 1)

    # 데이터 분할 (전체 데이터 순서 기반)
    idx_train, idx_val = train_test_split(
        np.arange(len(df_master)), 
        test_size=0.2, 
        stratify=df_master['target_event'], 
        random_state=42
    )
    
    # 모델 및 학습 설정
    model = ClinicalTransformer(seq_len=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # lr 살짝 조정
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 Transformer 학습 시작 (Device: {DEVICE}, Train samples: {len(idx_train)})...")
    train_loader = DataLoader(
        TensorDataset(X_tensor[idx_train], y_tensor[idx_train]), 
        batch_size=128, 
        shuffle=True
    )
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            outputs = model(bx.to(DEVICE))
            loss = criterion(outputs, by.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")
    
    # 위험도 점수(TS_Prob) 추출 및 저장
    print("🚀 [Step 2-2] 최종 시계열 위험도 점수(TS_Prob) 산출 중...")
    model.eval()
    with torch.no_grad():
        # 메모리 부족 방지를 위해 1024 배치씩 처리
        all_probs = []
        for i in range(0, len(X_tensor), 1024):
            batch_x = X_tensor[i:i+1024].to(DEVICE)
            batch_prob = torch.sigmoid(model(batch_x)).cpu().numpy().flatten()
            all_probs.extend(batch_prob)
        df_master['TS_Prob'] = all_probs
    
    # 최종 결과 저장
    df_master.to_csv(f"{FINAL_SAVE_PATH}/df_final_features.csv", index=False)
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/model_ts.pth")
    
    print(f"✅ Step 2 완료! 확보 환자 수: {len(df_master)}")
    print(f"    - 저장 경로: {FINAL_SAVE_PATH}/df_final_features.csv")

if __name__ == "__main__":
    run_step2()