import os
import sys
import time
import warnings
import subprocess

# 1. 경로 및 환경 설정
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

# src 디렉토리를 파이썬 모듈 검색 경로 맨 앞에 추가
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 2. 모듈 임포트 (경로 설정 후 진행)
try:
    from step1 import run_step1
    # step2는 스크립트 실행 방식이므로 생략
    from step3 import run_step3
    from step4 import run_step4
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

def run_process(step_name, step_target):
    """각 단계를 실행하고 시간을 측정하는 헬퍼 함수"""
    print(f"\n" + "="*60)
    print(f"🚀 {step_name} 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    try:
        if callable(step_target):
            # step1, 3, 4와 같이 함수로 정의된 경우
            step_target()
        else:
            # step2.py와 같이 스크립트 형태인 경우
            script_path = os.path.join(src_dir, step_target)
            # PYTHONPATH를 src로 설정하여 하위 프로세스에서도 모듈을 찾을 수 있게 함
            env = os.environ.copy()
            env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, script_path], check=True, env=env)
            
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n✅ {step_name} 완료! (소요 시간: {elapsed/60:.2f}분)")
    except Exception as e:
        print(f"\n❌ {step_name} 중 오류 발생: {e}")
        sys.exit(1)

def main():
    print("🏥 [수술 위기 예측 시스템: INSPIRE 130K] 파이프라인 가동")
    print(f"ROOT 경로: {current_dir}")
    print(f"SRC  경로: {src_dir}")
    
    # [Step 1] 데이터 전처리
    run_process("Step 1: 데이터 전처리 및 임상 지표 보간", run_step1)

    # [Step 2] 시계열 Transformer (파일 실행)
    run_process("Step 2: 시계열 Transformer 모델링", "step2.py")

    # [Step 3] 진료과별 모델 학습 및 비교
    run_process("Step 3: 진료과별 모델 벤치마킹", run_step3)

    # [Step 4] 최종 통합 검증 및 아카이빙
    run_process("Step 4: 통합 검증 및 SHAP 분석", run_step4)

    print("\n" + "="*60)
    print(f"🎉 모든 파이프라인 공정이 성공적으로 종료되었습니다!")
    print(f"최종 완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # [Step 5] 실시간 대시보드(Streamlit) 실행 안내
    print("\n" + "="*60)
    print("🖥️  Step 5: 서비스 대시보드 준비 완료")
    print("아래 명령어를 입력하여 웹 인터페이스를 실행하세요:")
    print(f"👉 streamlit run app.py")
    print("="*60)
    
if __name__ == "__main__":
    main()