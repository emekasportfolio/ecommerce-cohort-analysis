import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product

# ---------------- CONFIGURATION ---------------- #
DATA_PATH = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"
CHART_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
OUTPUT_DIR = r"C:\Users\emeka\ecommerce-cohort-analysis\output"  # Directory to save raw data and stats
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# ----------------- USER INPUT ------------------ #
df = pd.read_csv(DATA_PATH)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.tz_localize(None)

cohort_granularity = input("Select cohort granularity ('monthly' or 'weekly'): ").strip().lower()
assert cohort_granularity in ['monthly', 'weekly']

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

# Optional filters
filter_vals = {}
for col in ['Description', 'Country']:
    if col in df.columns and col in group_cols:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        print(f"\nAvailable values for {col}:")
        for i, val in enumerate(unique_vals, 1):
            print(f"{i}. {val}")
        print(f"{len(unique_vals)+1}. All")
        selected_indices = input(f"Select index/indices for {col} (comma-separated, e.g. 1,2 or {len(unique_vals)+1} for All): ")
        try:
            selected_indices = list(map(int, selected_indices.strip().split(',')))
            if (len(unique_vals)+1) in selected_indices:
                if len(selected_indices) > 1:
                    print("You cannot select 'All' with other specific options.")
                    exit()
                continue  # No filter applied
            else:
                filter_vals[col] = [unique_vals[i - 1] for i in selected_indices if 0 < i <= len(unique_vals)]
        except:
            continue

# Apply filters
for key, val in filter_vals.items():
    df = df[df[key].isin(val)]

# ---------------- GROUPED HEATMAPS ---------------- #
for group_col in group_cols:
    if group_col in ['Revenue', 'CustomerID']:
        continue

    unique_values = df[group_col].dropna().unique()
    for val in unique_values:
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

        sub_data = subset.groupby(['CohortPeriod', 'CohortIndex'])['CustomerID'].nunique().reset_index(name='n_customers')
        sub_table = sub_data.pivot(index='CohortPeriod', columns='CohortIndex', values='n_customers')
        if sub_table.empty: continue

        sub_sizes = sub_table.iloc[:, 0]
        sub_retention = sub_table.divide(sub_sizes, axis=0)
        sub_table['Total'] = sub_table.sum(axis=1)
        sub_retention['Total'] = sub_table['Total'] / sub_sizes

        plt.figure(figsize=(12, 7))
        plt.imshow(sub_retention, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        plt.title(f"{cohort_granularity.capitalize()} Cohort Retention - {group_col}: {val}")
        plt.ylabel('Cohort Period')
        plt.xlabel(f'{cohort_granularity.capitalize()}s Since First Purchase')
        plt.colorbar(label='Retention Rate')

        for i in range(sub_retention.shape[0]):
            for j in range(sub_retention.shape[1]):
                val_cell = sub_retention.iloc[i, j]
                if not pd.isna(val_cell):
                    plt.text(j, i, f"{val_cell:.0%}", ha='center', va='center', color='black')

        plt.yticks(range(len(sub_retention.index)), [str(x) for x in sub_retention.index])
        plt.xticks(range(sub_retention.shape[1]), sub_retention.columns)
        plt.tight_layout()
        fname = f"{cohort_granularity}_retention_{group_col}_{str(val)}.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(CHART_DIR, fname))
        plt.close()

        # Save the raw data
        raw_data_file = f"{cohort_granularity}_raw_data_{group_col}_{str(val)}.csv".replace(" ", "_").replace("/", "_")
        subset.to_csv(os.path.join(OUTPUT_DIR, raw_data_file), index=False)

# ---------------- COMBINED HEATMAPS & DESCRIPTIVES ---------------- #
if 'Country' in group_cols and 'Description' in group_cols and 'Country' in filter_vals and 'Description' in filter_vals:
    countries = filter_vals['Country']
    products = filter_vals['Description']

    for country, product_name in product(countries, products):
        subset = df[(df['Country'] == country) & (df['Description'] == product_name)].copy()
        if subset.empty: continue

        # Cohort calculations
        subset['OrderPeriod'] = subset['InvoiceDate'].dt.to_period('M' if cohort_granularity == 'monthly' else 'W')
        subset['CohortPeriod'] = subset.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M' if cohort_granularity == 'monthly' else 'W')

        subset['CohortIndex'] = (
            (subset['OrderPeriod'].dt.to_timestamp() - subset['CohortPeriod'].dt.to_timestamp()).dt.days // 7 + 1
            if cohort_granularity == 'weekly'
            else (subset['OrderPeriod'].dt.year - subset['CohortPeriod'].dt.year) * 12 +
                 (subset['OrderPeriod'].dt.month - subset['CohortPeriod'].dt.month) + 1
        )

        combo_data = subset.groupby(['CohortPeriod', 'CohortIndex'])['CustomerID'].nunique().reset_index(name='n_customers')
        combo_table = combo_data.pivot(index='CohortPeriod', columns='CohortIndex', values='n_customers')
        if combo_table.empty: continue

        combo_sizes = combo_table.iloc[:, 0]
        combo_retention = combo_table.divide(combo_sizes, axis=0)
        combo_table['Total'] = combo_table.sum(axis=1)
        combo_retention['Total'] = combo_table['Total'] / combo_sizes

        plt.figure(figsize=(12, 7))
        plt.imshow(combo_retention, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
        plt.title(f"{cohort_granularity.capitalize()} Retention: {country} - {product_name}")
        plt.ylabel('Cohort Period')
        plt.xlabel(f'{cohort_granularity.capitalize()}s Since First Purchase')
        plt.colorbar(label='Retention Rate')

        for i in range(combo_retention.shape[0]):
            for j in range(combo_retention.shape[1]):
                val_cell = combo_retention.iloc[i, j]
                if not pd.isna(val_cell):
                    plt.text(j, i, f"{val_cell:.0%}", ha='center', va='center', color='black')

        plt.yticks(range(len(combo_retention.index)), [str(x) for x in combo_retention.index])
        plt.xticks(range(combo_retention.shape[1]), combo_retention.columns)
        plt.tight_layout()
        fname = f"{cohort_granularity}_retention_{country}_{product_name}.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(CHART_DIR, fname))
        plt.close()

        # Save the raw data
        raw_data_file = f"{cohort_granularity}_raw_data_{country}_{product_name}.csv".replace(" ", "_").replace("/", "_")
        subset.to_csv(os.path.join(OUTPUT_DIR, raw_data_file), index=False)

        # Descriptive statistics
        desc_stats = subset.describe().transpose()
        desc_stats_file = f"{cohort_granularity}_descriptive_stats_{country}_{product_name}.csv".replace(" ", "_").replace("/", "_")
        desc_stats.to_csv(os.path.join(OUTPUT_DIR, desc_stats_file))

print("Process completed successfully!")


from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import os
import warnings

# ── CONFIG ────────────────────────────────────────────────────────────────
CLEANED_PATH = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"
OUTPUT_DIR   = r"C:\Users\emeka\ecommerce-cohort-analysis\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv(CLEANED_PATH, parse_dates=['InvoiceDate'])

# ── CLASSIFY EXTRA VARIABLES ───────────────────────────────────────────────
def add_custom_classes(df):
    # Create Product class
    df['ProductClass'] = df['Description'].astype(str)

    # Create Customer class
    df['CustomerClass'] = df['CustomerID'].astype(str)

    # Bin TotalPrice into 'Low', 'Medium', 'High'
    price_bins = pd.qcut(df['TotalPrice'], q=3, labels=['Low', 'Medium', 'High'])
    df['PriceClass'] = price_bins

    # Country as a class
    df['CountryClass'] = df['Country'].astype(str)

    return df

df = add_custom_classes(df)

# ----------------- MACHINE LEARNING CLUSTERING ------------------ #
def clustering_rfm(df, clustering_method='kmeans'):
    now = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'InvoiceNo':   'nunique',
        'TotalPrice':  'sum'
    }).rename(columns={
        'InvoiceDate':'Recency',
        'InvoiceNo':'Frequency',
        'TotalPrice':'Monetary'
    }).reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

    if clustering_method == 'kmeans':
        model = KMeans(n_clusters=4, random_state=42)
    else:
        model = DBSCAN(eps=0.5, min_samples=5)

    rfm['Cluster'] = model.fit_predict(scaled)
    return rfm

# ------------------ PREDICTING CHURN ------------------ #
def churn_prediction(df, model_type='logistic_regression'):
    now = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'InvoiceNo':   'nunique',
        'TotalPrice':  'sum'
    }).rename(columns={
        'InvoiceDate':'Recency',
        'InvoiceNo':'Frequency',
        'TotalPrice':'Monetary'
    })

    rfm['Churn'] = (rfm['Recency'] > 180).astype(int)

    X = rfm[['Recency','Frequency','Monetary']]
    y = rfm['Churn']

    print("Feature matrix shape:", X.shape, "Target vector shape:", y.shape)

    if len(y.unique()) < 2:
        warnings.warn("Only one class present in target. Skipping training.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_type.title()} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return acc, model, y_pred

# ----------------- MAIN FUNCTION ------------------ #
def main():
    print("Welcome to Cohort+RFM ML Module\n")

    if input("Cluster customers by RFM? (y/n): ").strip().lower() == 'y':
        method = input("Method ('kmeans' or 'dbscan'): ").strip().lower()
        rfm_clusters = clustering_rfm(df, method)
        print(rfm_clusters.head())
        rfm_clusters.to_csv(
            os.path.join(OUTPUT_DIR, f"rfm_clusters_{method}.csv"), index=False
        )

    if input("\nPredict churn? (y/n): ").strip().lower() == 'y':
        print("\n-- Logistic Regression --")
        lr_acc, lr_model, lr_pred = churn_prediction(df, 'logistic_regression')

        print("\n-- Random Forest --")
        rf_acc, rf_model, rf_pred = churn_prediction(df, 'random_forest')

        if lr_model is None and rf_model is None:
            print("\nSkipping prediction: insufficient class variety in the data.")
            return

        best, best_name = (
            (lr_model, 'logistic') if (lr_acc or 0) >= (rf_acc or 0) else (rf_model, 'random_forest')
        )
        print(f"\nBest model: {best_name} (accuracy {max(lr_acc or 0, rf_acc or 0):.4f})")

        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).rename(columns={
            'InvoiceDate':'Recency',
            'InvoiceNo':'Frequency',
            'TotalPrice':'Monetary'
        })

        rfm['Churn_Prediction'] = best.predict(rfm[['Recency','Frequency','Monetary']])
        rfm.to_csv(os.path.join(OUTPUT_DIR, "churn_predictions.csv"))

    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
