import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
)
import statsmodels.api as sm

RESULTS_DIR = "C:/Users/emeka/ecommerce-cohort-analysis/results"

class RegressionModel:
    def __init__(self, df, metrics, factors, model_type='linear'):
        """
        model_type: 'linear' or 'logistic'
        """
        self.df = df
        self.metrics = metrics
        self.factors = factors
        self.model_type = model_type.lower()
        self.models = {}

        if self.model_type not in ['linear', 'logistic']:
            raise ValueError("model_type must be 'linear' or 'logistic'")

    def run(self):
        results = {}

        for metric in self.metrics:
            if len(self.factors) == 0:
                print(f"Running {self.model_type} regression for {metric} with intercept only...")
                X = pd.DataFrame({'Intercept': [1] * len(self.df)})
                y = self.df[metric]
            else:
                print(f"Running {self.model_type} regression for {metric} with: {', '.join(self.factors)}...")
                X = self.df[self.factors]
                y = self.df[metric]

            if self.model_type == 'linear':
                model = LinearRegression()
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X, y)
            self.models[metric] = model
            y_pred = model.predict(X)

            results[metric] = y_pred
            self._save_results(model, X, y, y_pred, metric)

        return results

    def predict(self, label="user_input"):
        if not self.models:
            raise ValueError("No models trained yet. Please run .run() first.")

        print(f"\n🧠 Enter the following values for {self.model_type} prediction:")

        user_input = {}
        for factor in self.factors:
            while True:
                try:
                    val = float(input(f"Enter value for '{factor}': "))
                    user_input[factor] = val
                    break
                except ValueError:
                    print(f"❌ Invalid value for '{factor}'. Please enter a number.")

        X_condition = pd.DataFrame([[user_input[f] for f in self.factors]], columns=self.factors)

        predictions = []
        for metric in self.metrics:
            model = self.models.get(metric)
            if not model:
                continue

            if self.model_type == 'linear':
                y_pred = model.predict(X_condition)[0]
                predictions.append({
                    "label": label,
                    "metric": metric,
                    "prediction": round(y_pred, 4)
                })
            else:
                prob = model.predict_proba(X_condition)[0][1]
                y_class = model.predict(X_condition)[0]
                predictions.append({
                    "label": label,
                    "metric": metric,
                    "predicted_class": y_class,
                    "predicted_probability": round(prob, 4)
                })

        df_preds = pd.DataFrame(predictions)
        pred_path = os.path.join(RESULTS_DIR, f"{label}_{self.model_type}_predictions.csv")
        df_preds.to_csv(pred_path, index=False)
        print(f"\n✅ Prediction saved to: {pred_path}")
        return predictions

    def _save_results(self, model, X, y, y_pred, metric):
        # Save predictions
        if self.model_type == 'linear':
            df_pred = pd.DataFrame({f"{metric}_pred": y_pred})
        else:
            y_proba = model.predict_proba(X)[:, 1]
            df_pred = pd.DataFrame({
                f"{metric}_class": y_pred,
                f"{metric}_probability": y_proba
            })

        df_pred.to_csv(os.path.join(RESULTS_DIR, f"{metric}_{self.model_type}_predictions.csv"), index=False)

        # Save coefficients
        pd.DataFrame(model.coef_.reshape(-1, 1), index=self.factors, columns=[f"{metric}_coef"]).to_csv(
            os.path.join(RESULTS_DIR, f"{metric}_{self.model_type}_coefficients.csv")
        )

        # Save intercept
        pd.DataFrame({"intercept": [model.intercept_[0] if hasattr(model.intercept_, '__iter__') else model.intercept_]}).to_csv(
            os.path.join(RESULTS_DIR, f"{metric}_{self.model_type}_intercept.csv"), index=False
        )

        # Save diagnostic plots and reports
        if self.model_type == 'linear':
            self._linear_diagnostics(model, X, y, y_pred, metric)
        else:
            self._logistic_diagnostics(model, X, y, metric)

    def _linear_diagnostics(self, model, X, y, y_pred, metric):
        residuals = y - y_pred

        # Residuals vs Fitted
        plt.figure(figsize=(6, 4))
        sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
        plt.title(f"{metric} - Residuals vs Fitted")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_linear_residuals_vs_fitted.png"))
        plt.close()

        # Q-Q plot
        sm.qqplot(residuals, line='45')
        plt.title(f"{metric} - Normal Q-Q")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_linear_qqplot.png"))
        plt.close()

        # Histogram of residuals
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=20)
        plt.title(f"{metric} - Residuals Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_linear_residuals_histogram.png"))
        plt.close()

        # Save residuals
        pd.DataFrame({"residuals": residuals}).to_csv(
            os.path.join(RESULTS_DIR, f"{metric}_linear_residuals.csv"), index=False
        )

    def _logistic_diagnostics(self, model, X, y, metric):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        cm_df.to_csv(os.path.join(RESULTS_DIR, f"{metric}_logistic_confusion_matrix.csv"))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{metric} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_logistic_roc_curve.png"))
        plt.close()

        # Save AUC
        with open(os.path.join(RESULTS_DIR, f"{metric}_logistic_auc.txt"), 'w') as f:
            f.write(f"AUC Score: {roc_auc:.4f}\n")

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(RESULTS_DIR, f"{metric}_logistic_classification_report.csv")
        )
