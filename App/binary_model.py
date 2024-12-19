from base_model import BaseModel
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class BinaryModel(BaseModel):
    def __init__(self):
        self.model = None
        self.target = None
        self.predictors = []

    def load_data(self, file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

    def select_variables(self, data):
        valid_targets = [col for col in data.columns if data[col].isin([0, 1]).all()]
        valid_predictors = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        return valid_targets, valid_predictors

    def train(self, data, target, predictors):
        x = data[predictors]
        y = data[target]
        x = sm.add_constant(x)
        self.model = sm.Logit(y, x).fit()
        self.target = target
        self.predictors = predictors

    def get_summary(self):
        if self.model is None:
            raise ValueError("No model trained")
        return self.model.summary()

    def plot_results(self, data, target, predictors):
        predicted_probs = self.model.predict(sm.add_constant(data[predictors]))
        true_values = data[target]
        fpr, tpr, _ = roc_curve(true_values, predicted_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def predict(self, file_path):
        prediction_data = self.load_data(file_path)
        missing_predictors = [p for p in self.predictors if p not in prediction_data.columns]
        if missing_predictors:
            raise ValueError(f"Missing predictors: {', '.join(missing_predictors)}")
        x_predict = sm.add_constant(prediction_data[self.predictors])
        prediction_data['Predicted'] = self.model.predict(x_predict)
        return prediction_data
