import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2

class DataPreprocessor:
    def __init__(self, data):
        """데이터프레임을 초기화"""
        self.data = data

    def handle_missing_values(self, strategy='mean', fill_value=None):
        """결측치 처리 함수
        Args:
            strategy (str): 결측치 대체 방법 ('mean', 'median', 'most_frequent', 'constant')
            fill_value: strategy가 'constant'일 경우 대체할 값
        """
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def normalize_data(self):
        """데이터 정규화 (MinMaxScaler)"""
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def standardize_data(self):
        """데이터 표준화 (StandardScaler)"""
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def encode_labels(self, column):
        """카테고리형 데이터 레이블 인코딩 (Label Encoding)
        Args:
            column (str): 레이블 인코딩할 열 이름
        """
        encoder = LabelEncoder()
        self.data[column] = encoder.fit_transform(self.data[column])

    def one_hot_encode(self, columns):
        """카테고리형 데이터 원-핫 인코딩
        Args:
            columns (list): 원-핫 인코딩할 열들의 리스트
        """
        self.data = pd.get_dummies(self.data, columns=columns)

    def feature_selection(self, target_column, k=10):
        """특성 선택 (SelectKBest 사용, chi2 점수 기준)
        Args:
            target_column (str): 종속 변수(타겟) 열 이름
            k (int): 선택할 특성 개수
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        selector = SelectKBest(score_func=chi2, k=k)
        X_new = selector.fit_transform(X, y)
        self.data = pd.DataFrame(X_new, columns=[self.data.columns[i] for i in selector.get_support(indices=True)])
        self.data[target_column] = y

    def drop_columns(self, columns):
        """특정 열 삭제
        Args:
            columns (list): 삭제할 열들의 리스트
        """
        self.data.drop(columns=columns, inplace=True)

    def remove_outliers(self, z_threshold=3):
        """이상치 제거 (Z-score 기준)
        Args:
            z_threshold (float): Z-score의 임계값
        """
        from scipy.stats import zscore
        z_scores = pd.DataFrame(zscore(self.data.select_dtypes(include=['float64', 'int64'])))
        filtered_entries = (z_scores < z_threshold).all(axis=1)
        self.data = self.data[filtered_entries]

    def get_processed_data(self):
        """전처리된 데이터 반환"""
        return self.data

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """데이터를 학습 및 테스트 세트로 분리
        Args:
            target_column (str): 종속 변수 열 이름
            test_size (float): 테스트 세트의 비율
            random_state (int): 랜덤 시드 값
        Returns:
            X_train, X_test, y_train, y_test: 학습 및 테스트 세트
        """
        from sklearn.model_selection import train_test_split
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

"""
주요 함수 설명:
handle_missing_values(): 결측치 처리 방법을 선택할 수 있습니다. mean, median, most_frequent, constant로 결측치를 대체합니다.
normalize_data(): MinMaxScaler를 사용하여 데이터를 [0, 1] 범위로 정규화합니다.
standardize_data(): StandardScaler를 사용해 평균이 0이고 표준편차가 1인 데이터로 표준화합니다.
encode_labels(): 카테고리형 데이터를 LabelEncoder를 사용해 레이블 인코딩합니다.
one_hot_encode(): 카테고리형 데이터를 원-핫 인코딩으로 변환합니다.
feature_selection(): SelectKBest를 사용해 가장 상관관계가 높은 특성을 선택합니다.
drop_columns(): 필요 없는 열을 삭제합니다.
remove_outliers(): Z-score 기준으로 이상치를 제거합니다.
split_data(): 데이터를 학습 및 테스트 세트로 분리합니다.
"""