from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        """모델 평가 초기화
        Args:
            model: 학습된 모델
            X_test: 테스트 데이터 (피처)
            y_test: 테스트 데이터 (타겟)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluate(self):
        """기본 평가: 정확도와 F1 스코어"""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        return {'accuracy': accuracy, 'f1_score': f1}

    def get_precision(self):
        """정밀도(Precision) 계산"""
        predictions = self.model.predict(self.X_test)
        return precision_score(self.y_test, predictions, average='weighted')

    def get_recall(self):
        """재현율(Recall) 계산"""
        predictions = self.model.predict(self.X_test)
        return recall_score(self.y_test, predictions, average='weighted')

    def get_confusion_matrix(self):
        """혼동 행렬 반환"""
        predictions = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, predictions)

    def get_classification_report(self):
        """정밀도, 재현율, F1 스코어 등이 포함된 상세 분류 보고서 생성"""
        predictions = self.model.predict(self.X_test)
        return classification_report(self.y_test, predictions)

    def plot_confusion_matrix(self):
        """혼동 행렬 시각화"""
        import seaborn as sns
        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(self.y_test), yticklabels=set(self.y_test))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def get_roc_auc(self):
        """ROC AUC 점수 계산"""
        if len(set(self.y_test)) == 2:  # 이진 분류만 적용
            y_probs = self.model.predict_proba(self.X_test)[:, 1]  # 양성 클래스에 대한 확률
            return roc_auc_score(self.y_test, y_probs)
        else:
            return "ROC AUC는 이진 분류에만 적용됩니다."

    def plot_roc_curve(self):
        """ROC 커브 시각화"""
        if len(set(self.y_test)) == 2:  # 이진 분류만 적용
            y_probs = self.model.predict_proba(self.X_test)[:, 1]  # 양성 클래스에 대한 확률
            fpr, tpr, _ = roc_curve(self.y_test, y_probs)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.show()
        else:
            print("ROC 커브는 이진 분류에만 적용됩니다.")

    def compare_models(self, other_model, X_test_other, y_test_other):
        """두 모델의 성능을 비교하여 정확도 및 F1 스코어를 출력
        Args:
            other_model: 비교할 다른 모델
            X_test_other: 다른 모델의 테스트 데이터
            y_test_other: 다른 모델의 타겟 값
        """
        predictions_self = self.model.predict(self.X_test)
        accuracy_self = accuracy_score(self.y_test, predictions_self)
        f1_self = f1_score(self.y_test, predictions_self, average='weighted')

        predictions_other = other_model.predict(X_test_other)
        accuracy_other = accuracy_score(y_test_other, predictions_other)
        f1_other = f1_score(y_test_other, predictions_other, average='weighted')

        print(f"Current model - Accuracy: {accuracy_self}, F1 Score: {f1_self}")
        print(f"Other model - Accuracy: {accuracy_other}, F1 Score: {f1_other}")
        
        return {
            'current_model': {'accuracy': accuracy_self, 'f1_score': f1_self},
            'other_model': {'accuracy': accuracy_other, 'f1_score': f1_other}
        }
