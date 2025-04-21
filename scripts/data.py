import pandas as pd
from functools import wraps

class Data:
    def __init__(self, file_path='C:\\Users\\emeka\\ecommerce-cohort-analysis\\data\\raw\\Dataset_ecommerce.csv'):
        self.file_path = file_path

    def cleansed_data(self, func):
        @wraps(func)
        def wrapper():
            df = func()
            print("📅 Converting 'InvoiceDate' to datetime format...")
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            print("🧹 Dropping rows with missing CustomerID...")
            df.dropna(subset=['CustomerID'], inplace=True)

            print("🚫 Identifying cancelled invoices...")
            cancelled_ids = (
                df['InvoiceNo']
                  .astype(str)
                  .loc[lambda s: s.str.startswith('C')]
                  .str[1:]
                  .unique()
            )

            print(f"🗑️ Removing {len(cancelled_ids)} cancelled invoices and their originals...")
            drop_ids = set(cancelled_ids) | {f"C{inv}" for inv in cancelled_ids}
            df = df[~df['InvoiceNo'].astype(str).isin(drop_ids)]

            print("🔎 Filtering out negative quantities (returns)...")
            df = df[df['Quantity'] > 0]

            print("🧮 Calculating TotalPrice = Quantity × UnitPrice...")
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

            print("✅ Data cleaning complete.\n")
            return df
        return wrapper

    def raw_data(self):
        return pd.read_csv(self.file_path)
