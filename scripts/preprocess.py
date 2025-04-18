import pandas as pd
import os

# 1. Use absolute paths directly for clarity
input_path = r"C:\Users\emeka\ecommerce-cohort-analysis\data\raw\Dataset_ecommerce.csv"
output_path = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed\Dataset_ecommerce_cleaned.csv"

# 2. Load the dataset
df = pd.read_csv(input_path, encoding='ISO-8859-1')

# 3. Drop rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# 4. Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 5. Remove canceled invoices (InvoiceNo starts with 'C')
#df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
# 5. Remove cancelled invoices *and* their originals
#    a) Find all InvoiceNos that start with 'C' and strip the 'C' to get the original IDs
cancelled_ids = (
    df['InvoiceNo']
      .astype(str)
      .loc[lambda s: s.str.startswith('C')]
      .str[1:]
      .unique()
)

#    b) Build a set of IDs to drop: the 'C...' entries and their matching originals
drop_ids = set(cancelled_ids) | {f"C{inv}" for inv in cancelled_ids}

#    c) Filter out all rows whose InvoiceNo is in that drop set
df = df[~df['InvoiceNo'].astype(str).isin(drop_ids)]


# 6. Remove negative quantities (returns)
df = df[df['Quantity'] > 0]

# 7. Compute TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Optional: If you still want to drop paired purchase-return transactions,
#    identify InvoiceNos where TotalPrice sums to zero (shouldn't occur now as negatives removed).
#    We'll skip this step since removing negative quantities handles returns.

# 8. Save cleaned data
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Preprocessing complete. {len(df)} rows saved to: {output_path}")

