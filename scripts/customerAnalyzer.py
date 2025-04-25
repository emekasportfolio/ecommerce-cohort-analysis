import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from scipy import stats

class UnifiedCustomerAnalyzer:
    def __init__(self, filepath, save_dir=r"C:\Users\emeka\ecommerce-cohort-analysis\data\scripts"):
        self.filepath = Path(filepath)
        self.save_dir = Path(save_dir)
        self.df = None
        self.rfm = None
        self.cleaned_summary = None
        self.filtered_summary = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        df = pd.read_csv(self.filepath, encoding='ISO-8859-1')
        df.dropna(subset=["CustomerID", "InvoiceDate"], inplace=True)
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        self.df = df

    def calculate_rfm_features(self):
        snapshot_date = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = self.df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

        rfm["AvgQuantity"] = self.df.groupby("CustomerID")["Quantity"].mean().values
        rfm["AvgUnitPrice"] = self.df.groupby("CustomerID")["UnitPrice"].mean().values
        rfm["AvgReorderDays"] = rfm["Recency"] / rfm["Frequency"].replace(0, 1)

        self.rfm = rfm

    def train_models(self):
        X = self.rfm[["Recency", "Frequency", "Monetary", "AvgQuantity", "AvgUnitPrice", "AvgReorderDays"]]
        y = self.rfm["Frequency"] * self.rfm["AvgQuantity"] * self.rfm["AvgUnitPrice"]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        y_pred = rf_model.predict(X)
        print(f"[RF] MAE: {mean_absolute_error(y, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)

    def generate_customer_summary(self):
        snapshot_date = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)
        summary = summary_data_from_transaction_data(
            self.df, 'CustomerID', 'InvoiceDate', monetary_value_col='TotalPrice', observation_period_end=snapshot_date, freq='D')
        self.cleaned_summary = summary[(summary["frequency"] > 0) & (summary["recency"] <= summary["T"])]

    def predict_sales_budget(self, days=30):
        bgf = BetaGeoFitter(penalizer_coef=10.0)
        bgf.fit(self.cleaned_summary['frequency'], self.cleaned_summary['recency'], self.cleaned_summary['T'])

        filtered = self.cleaned_summary[self.cleaned_summary['monetary_value'] > 0].copy()
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(filtered['frequency'], filtered['monetary_value'])

        filtered['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            days, filtered['frequency'], filtered['recency'], filtered['T'])
        filtered['predicted_monetary'] = ggf.conditional_expected_average_profit(
            filtered['frequency'], filtered['monetary_value'])
        filtered['expected_revenue'] = filtered['predicted_purchases'] * filtered['predicted_monetary']

        self.filtered_summary = filtered

    def export_top_customers(self, days=30, top_percent=10):
        top_n = int(len(self.filtered_summary) * top_percent / 100)
        top_customers = self.filtered_summary.sort_values("expected_revenue", ascending=False).head(top_n)
        output_file = self.save_dir / f"top_{top_percent}_customers_next_{days}_days.csv"
        top_customers.to_csv(output_file)
        print(f"Exported: {output_file}")

    def confidence_interval_for_total(self, predictions: pd.Series, confidence_level=0.95):
        n = len(predictions)
        total = predictions.sum()
        std_dev = predictions.std(ddof=1)
        std_err_total = std_dev * np.sqrt(n)
        t_score = stats.t.ppf((1 + confidence_level) / 2., n - 1)
        margin = t_score * std_err_total
        return total - margin, total + margin

    def segment_customers(self, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.rfm['Cluster'] = kmeans.fit_predict(self.rfm[['Recency', 'Frequency', 'Monetary']])
        sns.pairplot(self.rfm.reset_index(), hue='Cluster', vars=['Recency', 'Frequency', 'Monetary'])
        plt.suptitle("Customer Segments Based on RFM", y=1.02)
        plt.show()

    def execute_pipeline(self, days=30, top_percent=10):
        self.load_and_prepare_data()
        self.calculate_rfm_features()
        self.train_models()
        self.generate_customer_summary()
        self.predict_sales_budget(days)
        self.export_top_customers(days, top_percent)

        total = self.filtered_summary["expected_revenue"].sum()
        ci_low, ci_high = self.confidence_interval_for_total(self.filtered_summary["expected_revenue"])

        print(f"\nExpected total revenue in {days} days: {total:,.2f}")
        print(f"95% confidence interval: ({ci_low:,.2f}, {ci_high:,.2f})")

    def run_with_user_input(self):
        try:
            days = int(input("Enter number of days for forecast (e.g., 30): "))
            top_percent = float(input("Enter top customer percentage to export (e.g., 10 for top 10%): "))
            self.execute_pipeline(days=days, top_percent=top_percent)
        except ValueError:
            print("Invalid input. Please enter numeric values for both days and top percent.")

analyzer = UnifiedCustomerAnalyzer(r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv")
analyzer.run_with_user_input()
analyzer.segment_customers()