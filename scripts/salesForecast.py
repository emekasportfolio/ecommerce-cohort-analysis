# Re-import necessary modules 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
import warnings
warnings.filterwarnings('ignore')

# Recreate the SalesForecast class with full functionality
class SalesForecast:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, encoding='ISO-8859-1')
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df.dropna(subset=['CustomerID'], inplace=True)
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        self.rfm = None
        self.model = None
        self.xgb_model = None
        self.summary = None
        self.eligible_customers = None

    def train_predictive_model(self):
        """Train RandomForest and XGBoost models on RFM and behavioral features."""
        self.rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (x.max() - x.min()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()

        self.rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
        self.rfm['AvgQuantity'] = self.df.groupby('CustomerID')['Quantity'].mean().values
        self.rfm['AvgUnitPrice'] = self.df.groupby('CustomerID')['UnitPrice'].mean().values
        self.rfm['AvgReorderDays'] = self.rfm['Recency'] / self.rfm['Frequency']
        self.rfm.replace([np.inf, -np.inf], 0, inplace=True)
        self.rfm.fillna(0, inplace=True)

        X = self.rfm[['Recency', 'Frequency', 'Monetary', 'AvgQuantity', 'AvgUnitPrice', 'AvgReorderDays']]
        y = self.rfm['Monetary']

        # Train RandomForest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        y_pred_rf = self.model.predict(X)
        mae_rf = mean_absolute_error(y, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
        print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

        # Train XGBoost
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.xgb_model.fit(X, y)
        y_pred_xgb = self.xgb_model.predict(X)
        mae_xgb = mean_absolute_error(y, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y, y_pred_xgb))
        print(f"XGBoost - MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

        # K-Means Clustering on RFM
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.rfm['Cluster'] = kmeans.fit_predict(X)
        print("\nCustomer Segments based on RFM + Behavior (Cluster IDs):")
        print(self.rfm[['CustomerID', 'Cluster']].head())

    def generate_customer_summary(self):
        """Generate a customer lifecycle summary for use with BG/NBD and Gamma-Gamma models."""
        summary = summary_data_from_transaction_data(self.df, 'CustomerID', 'InvoiceDate', monetary_value_col='TotalPrice', freq='D')
        summary = summary[summary['monetary_value'] > 0]
        self.summary = summary
        print("Summary columns generated:", list(self.summary.columns))
        print(self.summary.head())

    def calculate_sales_budget(self, days):
        """Calculate sales budget based on customers whose reorder days <= user input."""
        eligible = self.rfm[self.rfm['AvgReorderDays'] <= days].copy()
        eligible = eligible.sort_values('AvgReorderDays')
        eligible['ProjectedRevenue'] = eligible['AvgQuantity'] * eligible['AvgUnitPrice']
        total_expected_revenue = eligible['ProjectedRevenue'].sum()

        print(f"\nExpected total revenue: {total_expected_revenue:,.2f}")
        print("Sample of customer budget plan:")
        print(eligible[['CustomerID', 'AvgReorderDays', 'AvgQuantity', 'AvgUnitPrice', 'ProjectedRevenue']].head())

        self.eligible_customers = eligible

    def conversion_strategy(self, days):
        """Estimate impact of converting a percentage of slower customers into faster ones."""
        slow = self.rfm[self.rfm['AvgReorderDays'] > days].copy()
        slow['ProjectedRevenue'] = slow['AvgQuantity'] * slow['AvgUnitPrice']

        bgf = BetaGeoFitter()
        bgf.fit(self.summary['frequency'], self.summary['recency'], self.summary['T'])
        churn_rate = 1 - bgf.conditional_probability_alive(self.summary['frequency'], self.summary['recency'], self.summary['T']).mean()

        print(f"\nEstimated churn rate: {churn_rate:.2%}")
        retained_rate = 1 - churn_rate
        converted_customers = slow.sample(frac=retained_rate, random_state=42)
        converted_total = converted_customers['ProjectedRevenue'].sum()

        old_total = self.eligible_customers['ProjectedRevenue'].sum()
        new_total = old_total + converted_total
        change = new_total - old_total

        print(f"Expected new revenue if {retained_rate:.2%} of slower customers convert: {new_total:,.2f}")
        print(f"Change in revenue: {change:,.2f}")

    def export_summary_table(self, path):
        """Export the eligible customer table to CSV."""
        self.eligible_customers.to_csv(path, index=False)
        print(f"\nExported eligible customer summary to {path}")


# 1. Initialize the model with your data path
sf = SalesForecast(r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv")  
#/content/Dataset_ecommerce_cleaned.csv

# 2. Train models and segment customers
sf.train_predictive_model()

# 3. Generate customer lifecycle summary
sf.generate_customer_summary()

# 4. Set a user-defined reorder day threshold (e.g., 30 days)
sf.calculate_sales_budget(days=30)

# 5. Estimate potential revenue gain from converting slow customers
sf.conversion_strategy(days=30)

# 6. Optionally export the selected customers to CSV
sf.export_summary_table(# 1. Initialize the model with your data path
sf = SalesForecast(r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv") 
#/content/Dataset_ecommerce_cleaned.csv

# 2. Train models and segment customers
sf.train_predictive_model()

# 3. Generate customer lifecycle summary
sf.generate_customer_summary()

# 4. Set a user-defined reorder day threshold (e.g., 30 days)
sf.calculate_sales_budget(days=30)

# 5. Estimate potential revenue gain from converting slow customers
sf.conversion_strategy(days=30)

# 6. Optionally export the selected customers to CSV
sf.export_summary_table(r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv")