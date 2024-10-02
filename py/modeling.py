import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

class ModelTrainer:
    def __init__(self, model_type='random_forest', params=None):
        """모델 초기화
        Args:
            model_type (str): 사용할 모델 선택 ('random_forest', 'logistic_regression', 'svc', 'gradient_boosting')
            params (dict): 모델에 전달할 파라미터
        """
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**(params or {}))
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**(params or {}))
        elif model_type == 'svc':
            self.model = SVC(**(params or {}))
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**(params or {}))

    def train(self, X_train, y_train):
        """모델 학습
        Args:
            X_train: 학습 데이터
            y_train: 타겟 값
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """예측 값 반환
        Args:
            X_test: 테스트 데이터
        Returns:
            예측 값
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """모델 평가 (accuracy, precision, recall, f1)
        Args:
            X_test: 테스트 데이터
            y_test: 실제 값
        Returns:
            dict: 각 평가 지표가 담긴 딕셔너리
        """
        predictions = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }

    def get_confusion_matrix(self, X_test, y_test):
        """혼동 행렬 생성
        Args:
            X_test: 테스트 데이터
            y_test: 실제 값
        Returns:
            혼동 행렬 (confusion matrix)
        """
        predictions = self.predict(X_test)
        return confusion_matrix(y_test, predictions)

    def get_classification_report(self, X_test, y_test):
        """정밀도, 재현율, F1 점수를 포함한 분류 보고서 생성
        Args:
            X_test: 테스트 데이터
            y_test: 실제 값
        Returns:
            분류 보고서
        """
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)

    def cross_validate(self, X, y, cv=5):
        """교차 검증
        Args:
            X: 전체 데이터
            y: 타겟 값
            cv (int): 교차 검증 폴드 수
        Returns:
            교차 검증 점수 리스트
        """
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores

    def hyperparameter_tuning(self, param_grid, X_train, y_train, cv=5):
        """그리드 서치를 통한 하이퍼파라미터 튜닝
        Args:
            param_grid (dict): 하이퍼파라미터 검색 공간
            X_train: 학습 데이터
            y_train: 타겟 값
            cv (int): 교차 검증 폴드 수
        Returns:
            튜닝된 모델과 그리드 서치 결과
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return self.model, grid_search.best_params_

    def save_model(self, file_path):
        """모델 저장 (pickle 파일로)
        Args:
            file_path (str): 저장할 파일 경로
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, file_path):
        """모델 불러오기 (pickle 파일에서)
        Args:
            file_path (str): 불러올 파일 경로
        """
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)

