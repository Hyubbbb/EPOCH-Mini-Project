아래는 모델링 클래스, 전처리 클래스, 모델 평가 클래스에 대한 설명과 예시 사용법을 담은 마크다운 형식의 README 파일입니다. 각 클래스의 주요 기능을 설명하고, 사용법 예시를 제공합니다.

---

# 머신러닝 프로젝트 클래스 설명 및 사용법

## 클래스 목록
1. **DataPreprocessor**: 데이터 전처리 클래스
2. **ModelTrainer**: 머신러닝 모델 학습 및 튜닝 클래스
3. **ModelEvaluator**: 모델 평가 클래스

---

## 1. DataPreprocessor

데이터 전처리 작업을 효율적으로 수행하기 위한 클래스입니다. 결측치 처리, 스케일링, 인코딩, 이상치 제거, 특성 선택 등의 작업을 지원합니다.

### 주요 메서드:
- `handle_missing_values(strategy='mean', fill_value=None)`: 결측치 처리 (평균, 중간값, 최빈값, 상수 값으로 대체).
- `normalize_data()`: 데이터 정규화 (MinMaxScaler).
- `standardize_data()`: 데이터 표준화 (StandardScaler).
- `encode_labels(column)`: 레이블 인코딩 (LabelEncoder).
- `one_hot_encode(columns)`: 원-핫 인코딩.
- `feature_selection(target_column, k=10)`: 특성 선택 (SelectKBest).
- `drop_columns(columns)`: 특정 열 삭제.
- `remove_outliers(z_threshold=3)`: 이상치 제거 (Z-score 기준).
- `split_data(target_column, test_size=0.2, random_state=42)`: 데이터를 학습/테스트 세트로 분리.

### 예시 사용법:
```python
from preprocessing import DataPreprocessor

# 데이터프레임 초기화
preprocessor = DataPreprocessor(data)

# 결측치 처리
preprocessor.handle_missing_values(strategy='mean')

# 데이터 정규화
preprocessor.normalize_data()

# 레이블 인코딩
preprocessor.encode_labels('Category')

# 전처리된 데이터 반환
processed_data = preprocessor.get_processed_data()

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = preprocessor.split_data(target_column='Target')
```

---

## 2. ModelTrainer

머신러닝 모델을 학습시키고, 교차 검증 및 하이퍼파라미터 튜닝을 지원하는 클래스입니다. 다양한 모델을 지원하며, 성능을 최적화할 수 있습니다.

### 주요 메서드:
- `train(X_train, y_train)`: 모델 학습.
- `predict(X_test)`: 예측 값 반환.
- `evaluate(X_test, y_test)`: 모델 평가 (accuracy, f1_score).
- `cross_validate(X, y, cv=5)`: 교차 검증.
- `hyperparameter_tuning(param_grid, X_train, y_train, cv=5)`: 그리드 서치를 사용한 하이퍼파라미터 튜닝.
- `save_model(file_path)`: 모델 저장 (pickle 파일로).
- `load_model(file_path)`: 모델 불러오기 (pickle 파일에서).

### 예시 사용법:
```python
from modeling import ModelTrainer

# 랜덤 포레스트 모델 생성
model_trainer = ModelTrainer(model_type='random_forest')

# 모델 학습
model_trainer.train(X_train, y_train)

# 모델 예측
predictions = model_trainer.predict(X_test)

# 모델 평가
results = model_trainer.evaluate(X_test, y_test)
print(results)

# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
best_model, best_params = model_trainer.hyperparameter_tuning(param_grid, X_train, y_train)
print(best_params)

# 모델 저장 및 불러오기
model_trainer.save_model('random_forest_model.pkl')
model_trainer.load_model('random_forest_model.pkl')
```

---

## 3. ModelEvaluator

모델의 성능을 평가하는 클래스입니다. 정확도, 정밀도, 재현율, F1 스코어 등의 다양한 지표를 제공하며, ROC 커브 및 혼동 행렬의 시각화도 지원합니다.

### 주요 메서드:
- `evaluate()`: 정확도와 F1 스코어를 반환.
- `get_precision()`: 정밀도(Precision) 반환.
- `get_recall()`: 재현율(Recall) 반환.
- `get_confusion_matrix()`: 혼동 행렬 반환.
- `get_classification_report()`: 정밀도, 재현율, F1 스코어 등의 상세 보고서 생성.
- `plot_confusion_matrix()`: 혼동 행렬 시각화.
- `get_roc_auc()`: ROC AUC 점수 반환 (이진 분류 전용).
- `plot_roc_curve()`: ROC 커브 시각화 (이진 분류 전용).
- `compare_models(other_model, X_test_other, y_test_other)`: 두 모델의 성능 비교.

### 예시 사용법:
```python
from evaluation import ModelEvaluator

# 모델 평가
evaluator = ModelEvaluator(model, X_test, y_test)

# 기본 평가 (정확도와 F1 스코어)
results = evaluator.evaluate()
print(results)

# 정밀도와 재현율 계산
precision = evaluator.get_precision()
recall = evaluator.get_recall()
print(f'Precision: {precision}, Recall: {recall}')

# 혼동 행렬 시각화
evaluator.plot_confusion_matrix()

# ROC 커브 시각화 (이진 분류일 경우)
roc_auc = evaluator.get_roc_auc()
print(f'ROC AUC: {roc_auc}')
evaluator.plot_roc_curve()

# 다른 모델과 성능 비교
evaluator.compare_models(other_model, X_test_other, y_test_other)
```

---

## 의존성

이 프로젝트를 실행하기 위해서는 다음 패키지들이 필요합니다:
```bash
pip install scikit-learn pandas matplotlib seaborn
```

---

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

---

이 리드미 파일을 통해, 데이터 전처리, 모델 학습, 모델 평가를 체계적으로 수행할 수 있는 클래스를 효율적으로 사용할 수 있습니다. 각 클래스를 적절히 활용해 데이터 분석 및 머신러닝 프로젝트를 성공적으로 진행하세요!
