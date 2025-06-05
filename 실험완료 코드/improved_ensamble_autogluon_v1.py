# 필수 라이브러리 임포트
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder


# 경고 메시지 무시 (필요에 따라 추가)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")




def data_processing(dataframe):
    """
    데이터 클렌징 및 새로운 피처 생성 수행
    - 불필요 컬럼 제거
    - 연령대 변환
    - 비율 기반 파생변수 생성
    """
    
    # 중복/불필요 컬럼 제거
    remove_cols = [
        '불임 원인 - 자궁경부 문제', 
        '미세주입 후 저장된 배아 수', 
        '불임 원인 - 정자 면역학적 요인'
    ]
    dataframe = dataframe.drop(
        columns=[col for col in remove_cols if col in dataframe.columns]
    )

    # 1. 수치형 파생변수 생성
    # 이식 배아 수의 제곱값 (비선형 관계 모델링)
    dataframe["이식 배아 수 제곱값"] = dataframe["이식된 배아 수"] ** 2
    
    # 2. 범주형 변수 매핑
    # 연령대를 수치형 중앙값으로 변환 (알 수 없음은 -1 처리)
    age_conversion = {
        "만18-34세": 26,
        "만35-37세": 36,
        "만38-39세": 38.5,
        "만40-42세": 41,
        "만43-44세": 43.5,
        "만45-50세": 47.5,
        "알 수 없음": -1  
    }
    dataframe["환자 연령 수치"] = dataframe["시술 당시 나이"].map(age_conversion)
    
    # 3. 비율 기반 파생변수
    # 배아 보존 효율 계산 (0 분할 방지를 위해 분모 +1)
    dataframe["배아 보존 효율"] = dataframe["저장된 배아 수"] / (dataframe["총 생성 배아 수"] + 1e-5)  
    
    # 연령 대비 이식 배아 수 비율
    dataframe["연령-배아 수 비율"] = dataframe["환자 연령 수치"] / (dataframe["이식된 배아 수"] + 1)
    
    return dataframe



if __name__ == "__main__":
    # 데이터셋 로드
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    
    # 전처리 적용
    train_processed = data_processing(train_data)
    test_processed = data_processing(test_data)
    
    # 식별자 컬럼 분리
    submission_ids = test_processed["ID"]
    train_processed = train_processed.drop(columns=["ID"])
    test_processed = test_processed.drop(columns=["ID"])
    
    # 모델 하이퍼파라미터 설정 (앙상블 구성)
    model_configs = {
        "GBM": {},      # Gradient Boosting
        "CAT": {},       # CatBoost
        "XGB": {},      # XGBoost
        "RF": {}        # Random Forest
    }
    
    # 모델 저장 경로 설정
    model_directory = "advanced_model_v2"
    
    # 예측기 초기화 및 학습
    pregnancy_predictor = TabularPredictor(
        label="임신 성공 여부", 
        eval_metric="roc_auc",
        path=model_directory
    )
    
    # 앙상블 학습 설정
    pregnancy_predictor.fit(
        train_data=train_processed,
        hyperparameters=model_configs,
        num_bag_folds=10,          # 10-fold bagging
        num_stack_levels=1,         # 1-level stacking
        presets="best_quality",     # 최고 품질 프리셋
        time_limit=7200             # 2시간 학습
    )
    
    # 테스트 데이터 예측 확률 추출
    prob_predictions = pregnancy_predictor.predict_proba(test_processed)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        "ID": submission_ids,
        "probability": prob_predictions[1]  # 양성 클래스 확률
    })
    submission_df.to_csv("Improved_ensamble_autoglon.csv", index=False)
    print("제출 파일 생성 완료: Improved_ensamble_autoglon.csv")
    
    # 모델 성능 분석 리포트
    performance_report = pregnancy_predictor.leaderboard(silent=False)
    
    # 피처 중요도 분석
    feature_importance = pregnancy_predictor.feature_importance(train_processed)
    print("\n피처 중요도 상위 20개:")
    print(feature_importance.head(20))
    
    # 이식 배아 수별 성공률 분석
    embryo_success = train_processed.groupby("이식된 배아 수")["임신 성공 여부"]\
        .value_counts().unstack().fillna(0).astype(int)
    print("\n이식 배아 수별 성공/실패 건수:")
    print(embryo_success)