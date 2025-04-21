import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from itertools import product
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
from datetime import timedelta
import seaborn as sns
import plotly.express as px  # For interactive trend chart

# ---------------- CONFIGURATION ---------------- #
DATA_PATH = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"
CHART_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
OUTPUT_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\output"
REPORT_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\reports"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------- LOAD DATA ------------------ #
df = pd.read_csv(DATA_PATH)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.tz_localize(None)

# ----------------- USER INPUT ------------------ #
cohort_granularity = input("Select cohort granularity ('monthly' or 'weekly'): ").strip().lower()
assert cohort_granularity in ['monthly', 'weekly'], "Invalid granularity input!"

print("\nSelect cohort grouping options (comma-separated):")
print("a. Revenue based")
print("b. Product based")
print("c. Country based")
print("d. CustomerID based (optional for granularity)")

choices = input("Enter choices (e.g. a,c or a,b,d): ").lower().replace(' ', '').split(',')
choice_map = {
    'a': 'Revenue',
    'b': 'Description',
    'c': 'Country',
    'd': 'CustomerID'
}
group_cols = [choice_map[ch] for ch in choices if ch in choice_map]

# ---------------- OPTIONAL FILTERS ---------------- #
filter_vals = {}
for col in ['Description', 'Country']:
    if col in df.columns and col in group_cols:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        print(f"\nAvailable values for {col}:")
        for i, val in enumerate(unique_vals, 1):
            print(f"{i}. {val}")
        print(f"{len(unique_vals)+1}. All")
        selected_indices = input(f"Select index/indices for {col} (comma-separated): ")
        try:
            selected_indices = list(map(int, selected_indices.strip().split(',')))
            if (len(unique_vals)+1) in selected_indices:
                continue
            else:
                filter_vals[col] = [unique_vals[i - 1] for i in selected_indices if 0 < i <= len(unique_vals)]
        except:
            continue

for key, val in filter_vals.items():
    df = df[df[key].isin(val)]

# ---------------- GROUPED HEATMAPS + TRENDS ---------------- #
for group_col in group_cols:
    if group_col in ['Revenue', 'CustomerID']:
        continue

    for val in df[group_col].dropna().unique():
        subset = df[df[group_col] == val].copy()
        if subset.empty: continue

        subset['OrderPeriod'] = subset['InvoiceDate'].dt.to_period('M' if cohort_granularity == 'monthly' else 'W')
        subset['CohortPeriod'] = subset.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M' if cohort_granularity == 'monthly' else 'W')

        subset['CohortIndex'] = (
            (subset['OrderPeriod'].dt.to_timestamp() - subset['CohortPeriod'].dt.to_timestamp()).dt.days // 7 + 1
            if cohort_granularity == 'weekly'
            else (subset['OrderPeriod'].dt.year - subset['CohortPeriod'].dt.year) * 12 +
                 (subset['OrderPeriod'].dt.month - subset['CohortPeriod'].dt.month) + 1
        )

        retention_data = subset.groupby(['CohortPeriod', 'CohortIndex'])['CustomerID'].nunique().reset_index(name='n_customers')
        retention_table = retention_data.pivot(index='CohortPeriod', columns='CohortIndex', values='n_customers')
        if retention_table.empty: continue

        base_sizes = retention_table.iloc[:, 0]
        retention_rate = retention_table.divide(base_sizes, axis=0)
        retention_table['Total'] = retention_table.sum(axis=1)
        retention_rate['Total'] = retention_table['Total'] / base_sizes

        # Heatmap
        plt.figure(figsize=(12, 7))
        plt.imshow(retention_rate, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        plt.title(f"{cohort_granularity.capitalize()} Cohort Retention - {group_col}: {val}")
        plt.ylabel('Cohort Period')
        plt.xlabel(f'{cohort_granularity.capitalize()}s Since First Purchase')
        plt.colorbar(label='Retention Rate')
        for i in range(retention_rate.shape[0]):
            for j in range(retention_rate.shape[1]):
                val_cell = retention_rate.iloc[i, j]
                if not pd.isna(val_cell):
                    plt.text(j, i, f"{val_cell:.0%}", ha='center', va='center', color='black')
        plt.yticks(range(len(retention_rate.index)), [str(x) for x in retention_rate.index])
        plt.xticks(range(retention_rate.shape[1]), retention_rate.columns)
        plt.tight_layout()
        fname = f"{cohort_granularity}_retention_{group_col}_{str(val)}.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(CHART_DIR, fname))
        plt.close()

        # Trend line
        trend = retention_data.groupby('CohortIndex')['n_customers'].sum().reset_index()
        trend_fig = px.line(trend, x='CohortIndex', y='n_customers', title=f'Cohort Trend Over Time - {group_col}: {val}')
        trend_fig.update_xaxes(title_text='Cohort Index')
        trend_fig.update_yaxes(title_text='Number of Customers')
        trend_fig.write_html(os.path.join(CHART_DIR, f"cohort_trend_{group_col}_{val}.html"))

        subset.to_csv(os.path.join(REPORT_DIR, f"{cohort_granularity}_raw_data_{group_col}_{val}.csv".replace(" ", "_").replace("/", "_")), index=False)

# ---------------- CLUSTERING + CLASSIFICATION + SHAP ---------------- #
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
customer_df = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'InvoiceDate': ['min', 'max']
})
customer_df.columns = ['TotalSpend', 'OrderCount', 'FirstPurchase', 'LastPurchase']
customer_df['DaysBetween'] = (customer_df['LastPurchase'] - customer_df['FirstPurchase']).dt.days + 1

X = customer_df[['TotalSpend', 'OrderCount', 'DaysBetween']].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans & DBSCAN Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5)
customer_df['DBSCANCluster'] = dbscan.fit_predict(X_scaled)

# Classification & SHAP
X_train, X_test, y_train, y_test = train_test_split(X_scaled, customer_df['KMeansCluster'], test_size=0.3, random_state=42, stratify=customer_df['KMeansCluster'])
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

with open(os.path.join(REPORT_DIR, "classification_report.txt"), 'w') as f:
    f.write("Logistic Regression on KMeans Clusters\n")
    f.write(classification_report(y_test, y_pred))

# SHAP Feature Importance
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "shap_feature_importance.png"))
plt.close()

# ---------------- LINEAR REGRESSION FOR RETENTION ---------------- #
customer_df['DaysSinceLast'] = (df['InvoiceDate'].max() - customer_df['LastPurchase']).dt.days
y_days = customer_df['DaysSinceLast']
X_linear = X.copy()

lr_model = LinearRegression()
lr_model.fit(X_linear, y_days)
y_pred_days = lr_model.predict(X_linear)
mse = mean_squared_error(y_days, y_pred_days)

# Save predictions
customer_df['PredictedDaysUntilNextPurchase'] = y_pred_days
customer_df.to_csv(os.path.join(REPORT_DIR, "customer_next_purchase_predictions.csv"), index=False)

# ---------------- TOP AND LEAST 5 LOYALISTS REPORT ---------------- #
# Get retention rates for the top 5 and least 5 loyal customers based on 'Retention Rate'
top_5_loyalists = customer_df.nlargest(5, 'TotalSpend')
least_5_loyalists = customer_df.nsmallest(5, 'TotalSpend')

# Predict next order and generate current trends
top_5_loyalists['PredictedNextOrder'] = lr_model.predict(top_5_loyalists[['TotalSpend', 'OrderCount', 'DaysBetween']])
least_5_loyalists['PredictedNextOrder'] = lr_model.predict(least_5_loyalists[['TotalSpend', 'OrderCount', 'DaysBetween']])

# Add current trend (simplified as an example)
top_5_loyalists['CurrentTrend'] = 'Increasing'  # Example trend
least_5_loyalists['CurrentTrend'] = 'Decreasing'  # Example trend

# Add retention probability (using clustering as proxy for retention likelihood)
top_5_loyalists['RetentionProbability'] = np.random.rand(5)  # Placeholder random values
least_5_loyalists['RetentionProbability'] = np.random.rand(5)

# Prepare report data
report_data = pd.concat([top_5_loyalists[['TotalSpend', 'RetentionProbability']], least_5_loyalists[['TotalSpend', 'RetentionProbability']]])

# Save the report to CSV
report_data.to_csv(os.path.join(REPORT_DIR, "top_and_least_5_loyalists_report.csv"), index=False)

print("Report for top 5 and least 5 loyalists generated successfully.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from itertools import product
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
from datetime import timedelta
import seaborn as sns
import plotly.express as px  # For interactive trend chart

# ---------------- CONFIGURATION ---------------- #
DATA_PATH = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"
CHART_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
OUTPUT_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\output"
REPORT_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\reports"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------- LOAD DATA ------------------ #
df = pd.read_csv(DATA_PATH)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.tz_localize(None)

# ----------------- USER INPUT ------------------ #
cohort_granularity = input("Select cohort granularity ('monthly' or 'weekly'): ").strip().lower()
assert cohort_granularity in ['monthly', 'weekly'], "Invalid granularity input!"

print("\nSelect cohort grouping options (comma-separated):")
print("a. Revenue based")
print("b. Product based")
print("c. Country based")
print("d. CustomerID based (optional for granularity)")

choices = input("Enter choices (e.g. a,c or a,b,d): ").lower().replace(' ', '').split(',')
choice_map = {
    'a': 'Revenue',
    'b': 'Description',
    'c': 'Country',
    'd': 'CustomerID'
}
group_cols = [choice_map[ch] for ch in choices if ch in choice_map]

# ---------------- OPTIONAL FILTERS ---------------- #
filter_vals = {}
for col in ['Description', 'Country']:
    if col in df.columns and col in group_cols:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        print(f"\nAvailable values for {col}:")
        for i, val in enumerate(unique_vals, 1):
            print(f"{i}. {val}")
        print(f"{len(unique_vals)+1}. All")
        selected_indices = input(f"Select index/indices for {col} (comma-separated): ")
        try:
            selected_indices = list(map(int, selected_indices.strip().split(',')))
            if (len(unique_vals)+1) in selected_indices:
                continue
            else:
                filter_vals[col] = [unique_vals[i - 1] for i in selected_indices if 0 < i <= len(unique_vals)]
        except:
            continue

for key, val in filter_vals.items():
    df = df[df[key].isin(val)]

# ---------------- GROUPED HEATMAPS + TRENDS ---------------- #
for group_col in group_cols:
    if group_col in ['Revenue', 'CustomerID']:
        continue

    for val in df[group_col].dropna().unique():
        subset = df[df[group_col] == val].copy()
        if subset.empty: continue

        subset['OrderPeriod'] = subset['InvoiceDate'].dt.to_period('M' if cohort_granularity == 'monthly' else 'W')
        subset['CohortPeriod'] = subset.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M' if cohort_granularity == 'monthly' else 'W')

        subset['CohortIndex'] = (
            (subset['OrderPeriod'].dt.to_timestamp() - subset['CohortPeriod'].dt.to_timestamp()).dt.days // 7 + 1
            if cohort_granularity == 'weekly'
            else (subset['OrderPeriod'].dt.year - subset['CohortPeriod'].dt.year) * 12 +
                 (subset['OrderPeriod'].dt.month - subset['CohortPeriod'].dt.month) + 1
        )

        retention_data = subset.groupby(['CohortPeriod', 'CohortIndex'])['CustomerID'].nunique().reset_index(name='n_customers')
        retention_table = retention_data.pivot(index='CohortPeriod', columns='CohortIndex', values='n_customers')
        if retention_table.empty: continue

        base_sizes = retention_table.iloc[:, 0]
        retention_rate = retention_table.divide(base_sizes, axis=0)
        retention_table['Total'] = retention_table.sum(axis=1)
        retention_rate['Total'] = retention_table['Total'] / base_sizes

        # Heatmap
        plt.figure(figsize=(12, 7))
        plt.imshow(retention_rate, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        plt.title(f"{cohort_granularity.capitalize()} Cohort Retention - {group_col}: {val}")
        plt.ylabel('Cohort Period')
        plt.xlabel(f'{cohort_granularity.capitalize()}s Since First Purchase')
        plt.colorbar(label='Retention Rate')
        for i in range(retention_rate.shape[0]):
            for j in range(retention_rate.shape[1]):
                val_cell = retention_rate.iloc[i, j]
                if not pd.isna(val_cell):
                    plt.text(j, i, f"{val_cell:.0%}", ha='center', va='center', color='black')
        plt.yticks(range(len(retention_rate.index)), [str(x) for x in retention_rate.index])
        plt.xticks(range(retention_rate.shape[1]), retention_rate.columns)
        plt.tight_layout()
        fname = f"{cohort_granularity}_retention_{group_col}_{str(val)}.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(CHART_DIR, fname))
        plt.close()

        # Trend line
        trend = retention_data.groupby('CohortIndex')['n_customers'].sum().reset_index()
        trend_fig = px.line(trend, x='CohortIndex', y='n_customers', title=f'Cohort Trend Over Time - {group_col}: {val}')
        trend_fig.update_xaxes(title_text='Cohort Index')
        trend_fig.update_yaxes(title_text='Number of Customers')
        trend_fig.write_html(os.path.join(CHART_DIR, f"cohort_trend_{group_col}_{val}.html"))

        subset.to_csv(os.path.join(REPORT_DIR, f"{cohort_granularity}_raw_data_{group_col}_{val}.csv".replace(" ", "_").replace("/", "_")), index=False)

# ---------------- CLUSTERING + CLASSIFICATION + SHAP ---------------- #
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
customer_df = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'InvoiceDate': ['min', 'max']
})
customer_df.columns = ['TotalSpend', 'OrderCount', 'FirstPurchase', 'LastPurchase']
customer_df['DaysBetween'] = (customer_df['LastPurchase'] - customer_df['FirstPurchase']).dt.days + 1

X = customer_df[['TotalSpend', 'OrderCount', 'DaysBetween']].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans & DBSCAN Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5)
customer_df['DBSCANCluster'] = dbscan.fit_predict(X_scaled)

# Classification & SHAP
X_train, X_test, y_train, y_test = train_test_split(X_scaled, customer_df['KMeansCluster'], test_size=0.3, random_state=42, stratify=customer_df['KMeansCluster'])
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

with open(os.path.join(REPORT_DIR, "classification_report.txt"), 'w') as f:
    f.write("Logistic Regression on KMeans Clusters\n")
    f.write(classification_report(y_test, y_pred))

# SHAP Feature Importance
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "shap_feature_importance.png"))
plt.close()

# ---------------- LINEAR REGRESSION FOR RETENTION ---------------- #
customer_df['DaysSinceLast'] = (df['InvoiceDate'].max() - customer_df['LastPurchase']).dt.days
y_days = customer_df['DaysSinceLast']
X_linear = X.copy()

lr_model = LinearRegression()
lr_model.fit(X_linear, y_days)
y_pred_days = lr_model.predict(X_linear)
mse = mean_squared_error(y_days, y_pred_days)

# Save predictions
customer_df['PredictedDaysUntilNextPurchase'] = y_pred_days
customer_df.to_csv(os.path.join(REPORT_DIR, "customer_next_purchase_predictions.csv"), index=False)

# ---------------- TOP AND LEAST 5 LOYALISTS REPORT ---------------- #
# Get retention rates for the top 5 and least 5 loyal customers based on 'Retention Rate'
top_5_loyalists = customer_df.nlargest(5, 'TotalSpend')
least_5_loyalists = customer_df.nsmallest(5, 'TotalSpend')

# Predict next order and generate current trends
top_5_loyalists['PredictedNextOrder'] = lr_model.predict(top_5_loyalists[['TotalSpend', 'OrderCount', 'DaysBetween']])
least_5_loyalists['PredictedNextOrder'] = lr_model.predict(least_5_loyalists[['TotalSpend', 'OrderCount', 'DaysBetween']])

# Add current trend (simplified as an example)
top_5_loyalists['CurrentTrend'] = 'Increasing'  # Example trend
least_5_loyalists['CurrentTrend'] = 'Decreasing'  # Example trend

# Add retention probability (using clustering as proxy for retention likelihood)
top_5_loyalists['RetentionProbability'] = np.random.rand(5)  # Placeholder random values
least_5_loyalists['RetentionProbability'] = np.random.rand(5)

# Prepare report data
report_data = pd.concat([top_5_loyalists[['TotalSpend', 'RetentionProbability']], least_5_loyalists[['TotalSpend', 'RetentionProbability']]])

# Save the report to CSV
report_data.to_csv(os.path.join(REPORT_DIR, "top_and_least_5_loyalists_report.csv"), index=False)

print("Report for top 5 and least 5 loyalists generated successfully.")
