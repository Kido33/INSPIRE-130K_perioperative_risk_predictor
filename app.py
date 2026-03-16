import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="Perioperative Risk Predictor v2", layout="wide")

@st.cache_resource
def load_models():
    depts = ['CTS', 'GS', 'NS', 'OS', 'UR']
    models = {}
    # main_ver1.1.py가 생성한 경로로 수정
    base_model_path = './mfds_final_archive/models' 
    
    for d in depts:
        try:
            model_file = f'{base_model_path}/model_final_{d}.pkl'
            models[d] = joblib.load(model_file)
        except Exception as e:
            st.warning(f"⚠️ {d} 모델 파일을 찾을 수 없습니다. 경로: {model_file}")
    return models

models_dict = load_models()

# 2. 사이드바: 입력 모드 및 진료과 선택
st.sidebar.title("🏥 Patient Data Input")
input_mode = st.sidebar.radio("입력 방식 선택", ["파일 업로드 (Drag & Drop)", "수동 직접 입력"])

# 모델이 하나도 로드되지 않았을 경우를 대비한 처리
if not models_dict:
    st.error("❌ 로드된 모델이 없습니다. 모델 파일(.pkl) 경로를 확인해주세요.")
    st.stop()  # 이후 코드 실행 중단

selected_dept = st.sidebar.selectbox("진료과 선택", list(models_dict.keys()))

# [수정 포인트] selected_dept가 models_dict에 있는지 확인 후 접근
if selected_dept in models_dict:
    model_pack = models_dict[selected_dept]
    feats = model_pack['features']
    threshold = model_pack['threshold']
else:
    st.error(f"선택된 {selected_dept} 모델을 불러올 수 없습니다.")
    st.stop()

# --- 데이터 준비 로직 ---
input_df = pd.DataFrame()

# 데이터 로드 로직
if input_mode == "파일 업로드 (Drag & Drop)":
    uploaded_file = st.file_uploader("CSV 파일을 드래그하세요.", type=['csv'])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        missing_feats = [f for f in feats if f not in raw_df.columns]
        if missing_feats:
            st.error(f"⚠️ 필수 항목 누락: {missing_feats}")
        else:
            p_idx = st.selectbox("분석할 환자 선택", raw_df.index)
            input_df = raw_df.loc[[p_idx], feats]
else:
    st.subheader("⌨️ 수동 입력")
    col_a, col_b = st.columns(2)
    manual_data = {}
    for i, f in enumerate(feats):
        with col_a if i < len(feats)/2 else col_b:
            manual_data[f] = st.number_input(f, value=0.0)
    input_df = pd.DataFrame([manual_data])

# --- 3. 핵심 분석 및 시뮬레이션 (데이터가 있을 때만 실행) ---
if not input_df.empty:
    # 전처리
    if selected_dept == 'GS' and 'preop_crp' in input_df.columns:
        input_df['preop_crp'] = np.log1p(input_df['preop_crp'].fillna(0))

    # 예측
    model = model_pack['model']
    calibrator = model_pack['calibrator']
    raw_prob = model.predict_proba(input_df)[:, 1]
    
    if hasattr(calibrator, 'predict_proba'):
        prob = calibrator.predict_proba(raw_prob.reshape(-1, 1))[0, 1]
    else:
        prob = calibrator.transform(raw_prob)[0]

    st.header(f"📊 {selected_dept} 통합 분석 리포트")
    
    # [A] 상단: 위험도 표시 섹션
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        st.subheader("📌 위험도 결과")
        risk_color = "red" if prob >= threshold else "green"
        risk_status = "⚠️ 고위험" if prob >= threshold else "✅ 저위험"
        
        st.markdown(f"""
            <div style='padding:20px; border-radius:10px; background-color:{risk_color}15; border:2px solid {risk_color};'>
                <h2 style='color:{risk_color}; margin:0;'>{risk_status}</h2>
                <p style='font-size:24px; margin:0;'>예측 확률: <b>{prob*100:.2f}%</b></p>
                <small>기준점: {threshold*100:.1f}%</small>
            </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': risk_color},
                   'threshold': {'line': {'color': "black", 'width': 4}, 'value': threshold*100}},
            title={'text': "Risk Index (%)"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.subheader("🔍 위험 기여 요인 (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df, check_additivity=False)
        
        # RF 모델 특성에 따른 SHAP 추출
        current_shap = shap_values[1] if isinstance(shap_values, list) else (shap_values[:,:,1] if len(shap_values.shape)==3 else shap_values)
        
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        explanation = shap.Explanation(values=current_shap[0], base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                                       data=input_df.iloc[0].values, feature_names=feats)
        shap.plots.bar(explanation, show=False)
        st.pyplot(fig_shap)

    # [B] 하단: 시계열 코호트 시뮬레이션
    st.divider()
    st.subheader("📈 유사 코호트 내 위험도 추세 (Digital Twin)")
    
    dept_stats = {'CTS': (0.02, 0.015), 'GS': (0.03, 0.025), 'NS': (0.1, 0.05), 'OS': (0.12, 0.06), 'UR': (0.25, 0.1)}
    base, std = dept_stats.get(selected_dept, (0.05, 0.03))
    time_pts = ["Pre-op", "Day 1", "Day 3", "Day 5", "Discharge"]
    
    c_mean = np.array([base, base*1.5, base*1.2, base*0.8, base*0.4])
    c_max, c_min = c_mean + std*2, np.maximum(0, c_mean - std*1.5)
    p_trend = np.array([prob, prob*1.3, prob*1.6, prob*0.9, prob*0.4]) if prob >= threshold else np.array([prob, prob*0.9, prob*0.7, prob*0.4, prob*0.2])

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=time_pts+time_pts[::-1], y=np.concatenate([c_max, c_min[::-1]]), fill='toself', fillcolor='rgba(100,100,100,0.1)', line_color='rgba(0,0,0,0)', name="코호트 범위"))
    fig_trend.add_trace(go.Scatter(x=time_pts, y=c_mean, line=dict(color='gray', dash='dash'), name="코호트 평균"))
    fig_trend.add_trace(go.Scatter(x=time_pts, y=p_trend, line=dict(color=risk_color, width=4), marker=dict(size=10), name="환자 예측 경로"))
    
    fig_trend.update_layout(yaxis_title="Risk Probability", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

else:
    st.info("💡 데이터를 입력하거나 파일을 업로드하면 분석 리포트가 활성화됩니다.")