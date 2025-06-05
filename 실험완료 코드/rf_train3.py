import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
warnings.filterwarnings('ignore')

# 데이터 로드
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# Train Data X, y 분리
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 결측치 비율이 50%가 넘는 칼럼 제거
def highly_null(df, threshold=0.5):
    df_copy = df.copy()
    missing_ratio = df_copy.isnull().mean()
    null_culumns = df_copy.columns[missing_ratio > threshold]
    df_copy.drop(columns=null_culumns, inplace=True)
    return df_copy

X = highly_null(X)
test = highly_null(test)

# 칼럼 확인
train_columns = X.columns
test_columns = test.columns
columns_in_both = set(train_columns).intersection(set(test_columns))
print(f"same columns: {len(columns_in_both)}")
print(f"train columns: {len(train_columns)}")
print(f"test columns: {len(test_columns)}")

# 같은 데이터로만 구성된 칼럼 제거
def remove_all_same(df):
    df_copy = df.copy()
    same_value_columns = df_copy.columns[df_copy.nunique()==1]
    df_copy.drop(columns = same_value_columns, inplace = True)
    return df_copy

X = remove_all_same(X)
test = remove_all_same(test)

# 칼럼 재확인
train_columns = X.columns
test_columns = test.columns
columns_in_both = set(train_columns).intersection(set(test_columns))
print(f"same columns: {len(columns_in_both)}")
print(f"train columns: {len(train_columns)}")
print(f"test columns: {len(test_columns)}")

# null 개수 확인
print(f"train null : {X.isnull().sum().sum()}")
print(f"test null : {test.isnull().sum().sum()}")  

# 범주형 컬럼 지정
categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
]

# 카테고리(범주)형 컬럼들을 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# 문자열을 수치형으로 변환 - 알 수 없는 값은 -1로 인코딩
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 학습 데이터 변환 - 카테고리 값들을 학습하고 변환 (정수형으로 변환)
X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns]).astype(int)

# 테스트 데이터 변환 - 학습된 인코딩 규칙으로 테스트 데이터 변환 (정수형으로 변환)
X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns]).astype(int)

# 원래 수치형이었던 칼럼들
numeric_columns = [
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 혼합 경과일",
    "배아 이식 경과일",
]

# 결측치 처리
X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)

# 평가지표 함수
def get_clf_eval(y_test, pred=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}'.format(accuracy, precision, recall, f1))

def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def create_models():
    # 범주형 특성 인덱스 추출
    cat_features_indices = [X_train_encoded.columns.get_loc(col) for col in categorical_columns]
    
    # LightGBM (변경 없음)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        random_state=42,
        n_jobs=-1
    )
    
    # ExtraTrees (변경 없음)
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost (변경 없음)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 수정된 CatBoost 부분
    cb_model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function='Logloss',
        verbose=0,
        random_state=42,
        cat_features=cat_features_indices,  # 인덱스 리스트 전달
        thread_count=-1
    )
    
    return lgb_model, et_model, xgb_model, cb_model

# 교차 검증 및 앙상블 예측
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
models_cv_scores = {
    'lgb': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'et': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'xgb': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'cb': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}

# 테스트 데이터 예측값 저장 배열 확장 (4개 모델)
test_predictions = np.zeros((len(X_test_encoded), 4))

print("교차 검증 및 모델 학습 중...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y), 1):
    X_train_fold = X_train_encoded.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X_train_encoded.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # 모델 생성
    lgb_model, et_model, xgb_model, cb_model = create_models()
    
    # 각 모델 학습 및 예측
    models = [lgb_model, et_model, xgb_model, cb_model]
    model_names = ['lgb', 'et', 'xgb', 'cb']
    val_predictions = []
    
    for model, name in zip(models, model_names):
        # CatBoost 전용 학습 (범주형 특성 지정)
        if name == 'cb':
            model.fit(
                X_train_fold, y_train_fold,
                
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50
            )
        else:
            model.fit(X_train_fold, y_train_fold)
        
        # 검증 세트 예측
        val_pred = model.predict(X_val_fold)
        val_predictions.append(model.predict_proba(X_val_fold)[:, 1])
        
        # 개별 모델 성능 평가
        metrics = get_metrics(y_val_fold, val_pred)
        for metric, value in metrics.items():
            models_cv_scores[name][metric].append(value)
        
        # 테스트 데이터 예측 (확률값)
        test_fold_pred = model.predict_proba(X_test_encoded)[:, 1]
        test_predictions[:, model_names.index(name)] += test_fold_pred / n_splits
    
    # 앙상블 예측 (검증 세트)
    val_ensemble_pred_proba = np.mean(val_predictions, axis=0)
    val_ensemble_pred = (val_ensemble_pred_proba > 0.5).astype(int)
    
    # 앙상블 성능 평가
    ensemble_metrics = get_metrics(y_val_fold, val_ensemble_pred)
    for metric, value in ensemble_metrics.items():
        models_cv_scores['ensemble'][metric].append(value)
    
    print(f"\nFold {fold} 결과:")
    print(f"LGB - F1: {models_cv_scores['lgb']['f1'][-1]:.4f}")
    print(f"ET - F1: {models_cv_scores['et']['f1'][-1]:.4f}")
    print(f"XGB - F1: {models_cv_scores['xgb']['f1'][-1]:.4f}")
    print(f"CB - F1: {models_cv_scores['cb']['f1'][-1]:.4f}")
    print(f"Ensemble - F1: {models_cv_scores['ensemble']['f1'][-1]:.4f}")

# 최종 성능 출력
print("\n=== 최종 교차 검증 평균 성능 ===")
for model_name in ['lgb', 'et', 'xgb', 'cb', 'ensemble']:
    print(f"\n{model_name.upper()} 모델:")
    for metric, scores in models_cv_scores[model_name].items():
        print(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# 최종 테스트 데이터 예측 (앙상블)
final_predictions = np.mean(test_predictions, axis=1)

# 제출 파일 생성
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['probability'] = final_predictions
sample_submission.to_csv('./ensemble_submit3.csv', index=False)
print("\n제출 파일이 생성되었습니다: ensemble_submit.csv")

# 특성 중요도 비교 분석
def plot_feature_importance(models, model_names):
    plt.figure(figsize=(15, 10))
    for idx, model in enumerate(models):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif isinstance(model, CatBoostClassifier):
            importance = model.get_feature_importance()
        else:
            continue
            
        feat_imp = pd.DataFrame({
            'feature': X_train_encoded.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.subplot(2, 2, idx+1)
        sns.barplot(x='importance', y='feature', data=feat_imp.head(10))
        plt.title(f'{model_names[idx]} Feature Importance')
    plt.tight_layout()  # 그래프 간격 조정 추가
    plt.show()  # 그래프 출력 추가
# 모델 재학습 (전체 데이터)
final_models = create_models()
for model, name in zip(final_models, ['lgb', 'et', 'xgb', 'cb']):
    if name == 'cb':
        model.fit(X_train_encoded, y)
    else:
        model.fit(X_train_encoded, y)

# 중요도 시각화
plot_feature_importance(final_models, ['LightGBM', 'ExtraTrees', 'XGBoost', 'CatBoost'])


