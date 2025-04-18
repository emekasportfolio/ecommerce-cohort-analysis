import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- CONFIGURATION ---------------- #
DATA_PATH   = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"
CHART_DIR   = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
OUTPUT_DIR  = r"C:\Users\emeka\ecommerce-cohort-analysis\output"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- LOAD & INPUT ------------------ #
df = pd.read_csv(DATA_PATH, parse_dates=['InvoiceDate'])
df['InvoiceDate'] = df['InvoiceDate'].dt.tz_localize(None)

gran = input("Cohort granularity ('monthly' or 'weekly'): ").strip().lower()
assert gran in ['monthly','weekly']

print("\nGroup by (comma‑sep):")
print("a. Revenue  b. Product  c. Country  d. CustomerID")
map_ = {'a':'Revenue','b':'Description','c':'Country','d':'CustomerID'}
choices = input("Choices> ").lower().replace(' ','').split(',')
group_cols = [map_[c] for c in choices if c in map_]

# Build all context combinations
keys = [df[col].dropna().unique().tolist() for col in group_cols]
contexts = [dict(zip(group_cols, vals)) for vals in product(*keys)]

# Show contexts and let user pick
print("\nAvailable contexts:")
for idx, ctx in enumerate(contexts, 1):
    name = "|".join(f"{col[:3]}={val}" for col,val in ctx.items())
    print(f"{idx}. {name}")
sel = input("Select context number(s) to process (e.g. 1 or 1,3): ")
sel_idxs = [int(i)-1 for i in sel.split(',') if i.isdigit() and 1 <= int(i) <= len(contexts)]
contexts = [contexts[i] for i in sel_idxs]

# --------------- HELPER FUNCTIONS ---------------- #

def build_rfm(subdf):
    now = subdf['InvoiceDate'].max()
    return (subdf.groupby('CustomerID')
            .agg(Recency=('InvoiceDate', lambda x:(now-x.max()).days),
                 Frequency=('InvoiceNo','nunique'),
                 Monetary=('TotalPrice','sum')))

def churn_preds(subdf, model_type):
    rfm = build_rfm(subdf)
    rfm['Churn'] = (rfm['Recency']>180).astype(int)
    X, y = rfm[['Recency','Frequency','Monetary']], rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = (LogisticRegression(max_iter=1000, random_state=42)
             if model_type=='logistic'
             else RandomForestClassifier(random_state=42))
    model.fit(X_train, y_train)
    rfm[f'PredictedChurn_{model_type}'] = model.predict(X)
    print(f"\n[{model_type}] Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test)))
    return rfm

def next_order_timing(subdf):
    grp = subdf.groupby('CustomerID').InvoiceDate
    avg_repeat = grp.apply(lambda x: x.sort_values().diff().dropna().dt.days.mean())
    rfm = build_rfm(subdf)
    tmp = rfm.join(avg_repeat.rename('AvgRepeatDays'))
    tmp.dropna(inplace=True)

    subdf.loc[:, 'Period'] = subdf['InvoiceDate'].dt.to_period('M' if gran == 'monthly' else 'W')  # FIXED
    total_periods = subdf['Period'].nunique()
    active_periods = subdf.groupby('CustomerID')['Period'].nunique()
    tmp['RetentionRate'] = (active_periods / total_periods).loc[tmp.index]

    # FIXED: pass DataFrame with feature names
    X, y = tmp[['Recency', 'Frequency', 'Monetary']], tmp['AvgRepeatDays']
    reg = LinearRegression().fit(X, y)
    return tmp, reg

def visualize_customer_activity(subdf, uid):
    cust = subdf[subdf['CustomerID'] == uid].copy()
    if cust.empty:
        print(f" → No data for Customer {uid}")
        return
    cust.loc[:, 'Period'] = cust['InvoiceDate'].dt.to_period('M' if gran == 'monthly' else 'W')  # FIXED
    agg = cust.groupby('Period')['TotalPrice'].sum()
    agg.index = agg.index.to_timestamp()
    plt.figure(figsize=(10, 5))
    agg.plot(marker='o', title=f"Customer {uid} Activity Over Time")
    plt.ylabel("Total Spend")
    plt.grid()
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, f"activity_{uid}.png")
    plt.savefig(chart_path)
    plt.close()
    print(f" → Activity chart saved to {chart_path}")


def strategy_and_recs(subdf, context_name):
    sub = subdf.copy()
    sub['OrderPeriod'] = sub['InvoiceDate'].dt.to_period('M' if gran=='monthly' else 'W')
    sub['CohortPeriod'] = sub.groupby('CustomerID')['InvoiceDate'] \
                            .transform('min') \
                            .dt.to_period('M' if gran=='monthly' else 'W')
    sub['CohortIndex'] = (
        ((sub['OrderPeriod'].dt.to_timestamp() - sub['CohortPeriod'].dt.to_timestamp())
          .dt.days // 7 + 1)
        if gran=='weekly'
        else ((sub['OrderPeriod'].dt.year - sub['CohortPeriod'].dt.year)*12 +
             (sub['OrderPeriod'].dt.month - sub['CohortPeriod'].dt.month) + 1)
    )

    cohort = (sub.groupby(['CohortPeriod','CohortIndex'])['CustomerID']
              .nunique().reset_index(name='n_customers')
              .pivot(index='CohortPeriod', columns='CohortIndex', values='n_customers'))
    sizes = cohort.iloc[:,0]
    retention = cohort.divide(sizes, axis=0)
    best = retention[2].idxmax()
    print(f"\n[{context_name}] Most loyal cohort month: {best.strftime('%B %Y')}")

    loyal = sub.loc[sub['CohortPeriod']==best, 'CustomerID'].unique()
    top5 = (sub[sub['CustomerID'].isin(loyal)]
            .groupby('Description')['Quantity']
            .sum().nlargest(5))
    print("Top 5 products for that cohort:", top5.to_dict())

    churned = build_rfm(sub).loc[lambda x: x.Recency>180,'Recency']
    print("Churn timing summary:", churned.describe())

    print("Recommendations:")
    print(" • Personalized emails to those close to 180 days")
    print(" • Loyalty tiers for top cohorts")
    print(" • Retargeting ads for churn risk")
    print(" • Bundles of top5 products for each cohort\n")

def customer_retention_months(subdf, uid):
    cust_orders = subdf[subdf['CustomerID'] == uid]
    if cust_orders.empty:
        return []
    periods = cust_orders['InvoiceDate'].dt.to_period('M' if gran=='monthly' else 'W')
    months = sorted(periods.unique())
    return [p.strftime('%B') for p in months]

def active_products_by_month(subdf, uid):
    cust_orders = subdf[subdf['CustomerID'] == uid].copy()
    if cust_orders.empty:
        return {}
    cust_orders['Month'] = cust_orders['InvoiceDate'].dt.strftime('%B')
    grouped = cust_orders.groupby('Month')['Description'].unique()
    return {month: list(prods) for month, prods in grouped.items()}

# ---------------- PROCESS SELECTED CONTEXTS --------------- #
for ctx in contexts:
    d = df.copy()
    name = []
    for col, val in ctx.items():
        d = d[d[col]==val]
        name.append(f"{col[:3]}={val}")
    ctx_name = "|".join(name)

    print(f"\n=== CONTEXT: {ctx_name} ===")
    if d.empty:
        print(" No data, skipping.")
        continue

    # 1) Churn predictions
    for m in ['logistic','random_forest']:
        rfm = churn_preds(d, m)
        print(rfm[['Recency','Frequency','Monetary','Churn', f'PredictedChurn_{m}']].head())

    # 2) Strategy & Recommendations
    strategy_and_recs(d, ctx_name)

    # 3) Next‑order timing & user prediction
    cust_df, reg = next_order_timing(d)
    choice = input(
        f"Choose prediction option for {ctx_name}:\n"
        f"[1] Single customer\n"
        f"[2] Top 5 customers by loyalty\n"
        f"[3] Country-level overview\n"
        f"[4] Product or Revenue grouping\n"
        f"Enter 1, 2, 3, or 4: "
    ).strip()

    if choice == '1':
        uid = input(f"Enter a CustomerID to predict next-order in {ctx_name}: ").strip()
        if uid.isdigit() and int(uid) in cust_df.index:
            uid = int(uid)
            feat = cust_df.loc[uid, ['Recency','Frequency','Monetary']]
            pred = reg.predict([feat])[0]
            retention = cust_df.loc[uid, 'RetentionRate']
            print(f"\n→ Customer {uid} will likely reorder in ~{int(pred)} days")
            print(f"→ Retention Rate: {retention:.2f}")
            months = customer_retention_months(d, uid)
            print(f"→ Active Months: {months}")
            products_by_month = active_products_by_month(d, uid)
            print(f"→ Active Products: {products_by_month}")
            visualize_customer_activity(d, uid)
