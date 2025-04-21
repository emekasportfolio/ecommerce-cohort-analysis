import pandas as pd
from operator import attrgetter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from operator import attrgetter
import os




class Computation:
    def __init__(self, df):
        self.df = df.copy()

    def run_with_eda(eda_method_name):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                # First, run the main function
                result = func(self, *args, **kwargs)
                
                # Then, get and run the EDA method if it exists
                eda_method = getattr(self, eda_method_name, None)
                if callable(eda_method):
                    print(f"\U0001F4CA Running EDA method: {eda_method_name}()")
                    eda_method()
                else:
                    print(f"⚠️ EDA method '{eda_method_name}' not found.")
                
                return result
            return wrapper
        return decorator
    #--------------------------EDA--------------------------#
 

    def eda_cohort_retention_rate(self):
        """
        Generate and save cohort analysis retention charts and return summary tables.
        """
        df = self.df.copy()
        save_dir = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
        os.makedirs(save_dir, exist_ok=True)

        df['event_date'] = pd.to_datetime(df['InvoiceDate'])
        df['user_id'] = df['CustomerID']
        df['country'] = df['Country']

        df['cohort_month'] = df.groupby('user_id')['event_date'].transform('min').dt.to_period('M')
        df['event_month'] = df['event_date'].dt.to_period('M')
        df['period_index'] = (df['event_month'] - df['cohort_month']).apply(attrgetter('n'))

        cohort_data = df.groupby(['cohort_month', 'period_index'])['user_id'].nunique().unstack(1).fillna(0)
        cohort_sizes = cohort_data.iloc[:, 0]
        retention = cohort_data.divide(cohort_sizes, axis=0)

        # 📊 Retention Heatmap
        plt.figure(figsize=(14, 6))
        sns.heatmap(retention, annot=True, fmt=".0%", cmap="YlGnBu")
        plt.title("📊 Cohort Retention Heatmap")
        plt.ylabel("Cohort Month")
        plt.xlabel("Months Since Signup")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "retention_heatmap.png"))
        plt.close()

        # 📉 Retention Line Plot: Top and Bottom Performing Cohorts
        top_cohorts = retention.iloc[:, 1:].mean(axis=1).nlargest(3).index
        bottom_cohorts = retention.iloc[:, 1:].mean(axis=1).nsmallest(3).index
        plt.figure(figsize=(12, 6))
        for cohort in top_cohorts:
            plt.plot(retention.columns, retention.loc[cohort], label=f"Top: {cohort}", linewidth=2)
        for cohort in bottom_cohorts:
            plt.plot(retention.columns, retention.loc[cohort], label=f"Bottom: {cohort}", linestyle='--', linewidth=2)
        plt.title("📉 Retention Trends: Top vs Bottom Cohorts")
        plt.xlabel("Months Since Signup")
        plt.ylabel("Retention Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "retention_lineplot_top_vs_bottom.png"))
        plt.close()

        # 📆 Cohort Acquisition Volume
        plt.figure(figsize=(12, 4))
        cohort_sizes.plot(kind='bar', color='skyblue')
        plt.title("📆 Cohort Acquisition Volumes")
        plt.ylabel("Number of Users")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cohort_acquisition_volume.png"))
        plt.close()

        # 📈 Time Trend of Overall Retention
        overall = df.groupby('period_index')['user_id'].nunique() / df['user_id'].nunique()
        plt.figure(figsize=(10, 4))
        overall.plot(marker='o')
        plt.title("📈 Overall Retention Trend")
        plt.xlabel("Months Since Signup")
        plt.ylabel("Retention Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "overall_retention_trend.png"))
        plt.close()

        # 🧮 Average Retention Duration per Cohort
        avg_duration = df.groupby('user_id')['period_index'].max()
        avg_per_cohort = df.drop_duplicates('user_id').groupby('cohort_month')['user_id'].apply(
            lambda users: avg_duration[users].mean()
        )
        plt.figure(figsize=(10, 4))
        avg_per_cohort.plot(marker='o', color='purple')
        plt.title("🧮 Average Retention Duration per Cohort")
        plt.ylabel("Avg. Active Months")
        plt.xlabel("Cohort Month")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "avg_retention_duration_per_cohort.png"))
        plt.close()

        # 🔍 Cumulative Survival Curves (Kaplan-Meier)
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(12, 6))
        for cohort in retention.index[:5]:
            users = df[df['cohort_month'] == cohort]['user_id'].unique()
            durations = avg_duration[users]
            kmf.fit(durations, event_observed=np.ones_like(durations), label=str(cohort))
            kmf.plot_survival_function(ci_show=False)
        plt.title("🔍 Cumulative Survival Curves by Cohort")
        plt.xlabel("Months Since Signup")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cumulative_survival_curves.png"))
        plt.close()

        # 🏷️ Retention by Country
        if 'country' in df.columns and df['country'].notna().any():
            country_retention = df.groupby(['country', 'period_index'])['user_id'].nunique().unstack().fillna(0)
            country_retention = country_retention.divide(country_retention.iloc[:, 0], axis=0)
            plt.figure(figsize=(14, 6))
            for country in country_retention.index[:5]:
                plt.plot(country_retention.columns, country_retention.loc[country], label=country)
            plt.title("🏷️ Retention Trends by Country")
            plt.xlabel("Months Since Signup")
            plt.ylabel("Retention Rate")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "retention_by_country.png"))
            plt.close()

        # Save summary tables to processed data directory
        data_save_dir = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed"
        os.makedirs(data_save_dir, exist_ok=True)

        cohort_sizes.to_csv(os.path.join(data_save_dir, "cohort_sizes.csv"))
        retention.to_csv(os.path.join(data_save_dir, "retention_table.csv"))
        avg_per_cohort.to_csv(os.path.join(data_save_dir, "average_retention_duration.csv"))

        # ✅ Return summary tables
        return {
            'cohort_sizes': cohort_sizes,
            'retention_table': retention,
            'average_retention_duration': avg_per_cohort,
        }

    # ====================== RETENTION METRICS ======================
    @run_with_eda("eda_cohort_retention_rate")
    def cohort_retention_rate(self):
        print("📊 Computing Cohort Retention Rate...")
        df = self.df.copy()

        # Step 1: Convert to monthly cohorts
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

        # Step 2: Calculate difference in months
        df['CohortIndex'] = (df['InvoiceMonth'] - df['CohortMonth']).apply(attrgetter('n'))

        # Step 3: Cohort Analysis
        cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().unstack(0).T
        retention = cohort_data.divide(cohort_data[0], axis=0)

        # Step 4: Save processed data
        data_save_dir = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed"
        os.makedirs(data_save_dir, exist_ok=True)
        cohort_data.to_csv(os.path.join(data_save_dir, "cohort_data.csv"))
        retention.to_csv(os.path.join(data_save_dir, "retention.csv"))

        # Step 5: Assign cohort retention rate (based on cohort index average)
        retention_mean = retention.mean(axis=1)  # Mean across CohortMonths for each CohortIndex
        df['cohort_retention_rate'] = df['CohortIndex'].map(retention_mean)

        # Step 6: Update self.df
        self.df = df.copy()

        return cohort_data, retention,df
 

    #def repeat_purchase_rate(self):
     #   print("📈 Computing Repeat Purchase Rate...")
      #  df = self.df.copy()
       # repeat_customers = df.groupby('CustomerID')['InvoiceNo'].nunique()
       # repeat_rate = (repeat_customers > 1).sum() / repeat_customers.count()
       # return repeat_customers, repeat_rate

    def repeat_purchase_rate(self):
        print("📈 Computing Repeat Purchase Rate...")
        df = self.df.copy()
        repeat_customers = df.groupby('CustomerID')['InvoiceNo'].nunique()
        repeat_rate = (repeat_customers > 1).sum() / repeat_customers.count()

        repeat_customers_df = repeat_customers.to_frame(name='RepeatPurchaseCount')
        repeat_rate_df = pd.DataFrame({'RepeatPurchaseRate': [repeat_rate]})

        return repeat_customers_df, repeat_rate_df


    #def time_to_first_repeat(self):
    #    print("⏱️ Calculating Time to First Repeat Purchase...")
     #   df = self.df.copy()
     #   df = df.sort_values(['CustomerID', 'InvoiceDate'])
     #   time_diff_df = df.groupby('CustomerID')['InvoiceDate'].apply(lambda x: x.iloc[1] - x.iloc[0] if len(x) > 1 else pd.NaT)
     #   avg_time = time_diff_df.dropna().mean()
     #   return time_diff_df, avg_time

    def time_to_first_repeat(self):
            print("⏱️ Calculating Time to First Repeat Purchase...")
            df = self.df.copy()
            df = df.sort_values(['CustomerID', 'InvoiceDate'])
            time_diff_df = df.groupby('CustomerID')['InvoiceDate'].apply(lambda x: x.iloc[1] - x.iloc[0] if len(x) > 1 else pd.NaT)

            avg_time = time_diff_df.dropna().mean()

            time_diff_df = time_diff_df.to_frame(name='TimeToFirstRepeat')
            avg_time_df = pd.DataFrame({'AverageTimeToFirstRepeat': [avg_time]})

            return time_diff_df, avg_time_df


    def purchase_frequency(self):
        print("🔁 Calculating Purchase Frequency...")
        df = self.df.copy()
        #purchases = df.groupby(['CustomerID', pd.Grouper(key='InvoiceDate', freq='M')])['InvoiceNo'].nunique()
        purchases = df.groupby(['CustomerID', pd.Grouper(key='InvoiceDate', freq='ME')])['InvoiceNo'].nunique()
        return purchases.to_frame(name='MonthlyPurchaseCount')#purchases

    def customer_lifetime_value(self):
        print("💰 Calculating Customer Lifetime Value...")
        df = self.df.copy()
        clv = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
        return clv.to_frame(name='CustomerLifetimeValue')#clv

    def average_order_value(self):
        print("🧾 Calculating Average Order Value...")
        df = self.df.copy()
        order_count = df.groupby('CustomerID')['InvoiceNo'].nunique()
        total_spent = df.groupby('CustomerID')['TotalPrice'].sum()
        aov = total_spent / order_count
        return aov.to_frame(name='AverageOrderValue')#aov

    #def churn_rate(self):
    #    print("📉 Calculating Churn Rate...")
    #    df = self.df.copy()
    #    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    #    last_month = df['InvoiceMonth'].max()
    #    retention = df.groupby('CustomerID')['InvoiceMonth'].max()
    #    churned = (retention < last_month).sum() / retention.count()
    #    return retention, churned
    def churn_rate(self):
        print("📉 Calculating Churn Rate...")
        df = self.df.copy()
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        last_month = df['InvoiceMonth'].max()
        retention = df.groupby('CustomerID')['InvoiceMonth'].max()
        churned = (retention < last_month).sum() / retention.count()

        retention_df = retention.to_frame(name='LastInvoiceMonth')
        churn_df = pd.DataFrame({'ChurnRate': [churned]})

        return retention_df, churn_df

    def average_order_size(self):
        print("📦 Calculating Average Order Size...")
        df = self.df.copy()
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)

        # Sum quantity per order
        order_sizes = df.groupby(['CustomerID', 'InvoiceNo'])['Quantity'].sum()

        # Average number of items per order for each customer
        avg_order_size = order_sizes.groupby('CustomerID').mean()
        return avg_order_size.to_frame(name='AverageOrderSize')

    def average_order_time(self):
        print("⏱️ Calculating Average Order Time (days between orders)...")
        df = self.df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Remove duplicated invoices
        df_unique = df.drop_duplicates(subset=['CustomerID', 'InvoiceNo'])

        # Get sorted order dates per customer
        df_unique = df_unique.sort_values(by=['CustomerID', 'InvoiceDate'])

        # Time differences between successive orders
        df_unique['PrevOrderDate'] = df_unique.groupby('CustomerID')['InvoiceDate'].shift(1)
        df_unique['DaysBetweenOrders'] = (df_unique['InvoiceDate'] - df_unique['PrevOrderDate']).dt.days

        avg_order_time = df_unique.groupby('CustomerID')['DaysBetweenOrders'].mean().dropna()
        return avg_order_time.to_frame(name='AverageOrderTime')

    def predict_next_order_date(self):
        print("🔮 Predicting Next Order Dates...")
        df = self.df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Last order date
        last_order = df.groupby('CustomerID')['InvoiceDate'].max()
        avg_order_time_df = self.average_order_time()

        prediction = (last_order + pd.to_timedelta(avg_order_time_df['AverageOrderTime'], unit='D')).to_frame(name='PredictedNextOrderDate')
        return prediction


    def all_retention_metrics(self):
        return {
            'cohort_retention': self.cohort_retention_rate(),
            'repeat_purchase_rate': self.repeat_purchase_rate(),
            'time_to_first_repeat': self.time_to_first_repeat(),
            'purchase_frequency': self.purchase_frequency(),
            'clv': self.customer_lifetime_value(),
            'aov': self.average_order_value(),
            'churn_rate': self.churn_rate(),
            'order_time':self.average_order_time(),
            'average_order_size':self.average_order_size()
        }

    # ====================== RETENTION FACTORS ======================

    def order_size_quantity(self):
        print("📦 Analyzing Order Size...")
        df = self.df.copy()
        first_order = df.groupby('CustomerID').first()['Quantity']
        return first_order.to_frame(name='FirstOrderQuantity')#first_order

    def product_type(self):
        print("🧾 Analyzing Product Types...")
        df = self.df.copy()
        product_retention = df.groupby('StockCode')['CustomerID'].nunique()
        return product_retention.sort_values(ascending=False).to_frame(name='UniqueCustomers')#product_retention.sort_values(ascending=False)

    def price_sensitivity(self):
        print("💲 Analyzing Price Sensitivity...")
        df = self.df.copy()
        df['PriceBucket'] = pd.qcut(df['UnitPrice'], q=4, labels=["Low", "Mid", "High", "Very High"])
        #repeat_by_price = df.groupby('PriceBucket')['CustomerID'].nunique()
        repeat_by_price = df.groupby('PriceBucket', observed=True)['CustomerID'].nunique()

        return repeat_by_price.to_frame(name='UniqueCustomersByPrice')#return repeat_by_price

    def geographic_trends(self):
        print("🌍 Analyzing Geographic Trends...")
        df = self.df.copy()
        customers_by_country = df.groupby('Country')['CustomerID'].nunique()
        #return customers_by_country.sort_values(ascending=False)
        return customers_by_country.sort_values(ascending=False).to_frame(name='UniqueCustomers')


    def purchase_timing(self):
        print("⏰ Analyzing Purchase Timing...")
        df = self.df.copy()
        df['Weekday'] = df['InvoiceDate'].dt.day_name()
        df['Month'] = df['InvoiceDate'].dt.month
        timing_stats = df.groupby(['Weekday', 'Month'])['CustomerID'].nunique()
        return timing_stats.to_frame(name='CustomersPerWeekdayMonth')
        #return timing_stats

    def basket_size(self):
        print("🛒 Analyzing Basket Size and Retention...")
        df = self.df.copy()
        avg_basket = df.groupby('CustomerID')['TotalPrice'].mean()
        #return avg_basket
        return avg_basket.to_frame(name='AverageBasketSize')

    def all_factors(self):
        return {
            'order_size': self.order_size_quantity(),
            'product_type': self.product_type(),
            'price_sensitivity': self.price_sensitivity(),
            'geographic_trends': self.geographic_trends(),
            'purchase_timing': self.purchase_timing(),
            'basket_size': self.basket_size()
        }
    def none(self):
        return pd.DataFrame()

    

def eda_repeat_purchase_rate(self):
    

    df = self.df.copy()
    charts_dir = r"C:\Users\emeka\ecommerce-cohort-analysis\charts"
    data_dir = r"C:\Users\emeka\ecommerce-cohort-analysis\data\processed"
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['CustomerID'] = df['CustomerID'].astype(str)

    # Group by CustomerID
    customer_invoice_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()

    # Identify one-time and repeat customers
    one_time_customers = customer_invoice_counts[customer_invoice_counts == 1].count()
    repeat_customers = customer_invoice_counts[customer_invoice_counts > 1].count()

    # 🥧 Pie chart of one-time vs repeat customers
    plt.figure(figsize=(6, 6))
    plt.pie([one_time_customers, repeat_customers],
            labels=['One-Time', 'Repeat'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['#ff9999', '#66b3ff'])
    plt.title('One-Time vs Repeat Customers')
    plt.savefig(os.path.join(charts_dir, 'one_time_vs_repeat_pie.png'))
    plt.close()

    # 📊 Histogram of purchase counts per customer
    plt.figure(figsize=(8, 6))
    sns.histplot(customer_invoice_counts, bins=30, kde=False)
    plt.xlabel('Number of Purchases')
    plt.ylabel('Number of Customers')
    plt.title('Purchase Count per Customer')
    plt.savefig(os.path.join(charts_dir, 'purchase_count_histogram.png'))
    plt.close()

    # 📈 Trend line of repeat rate over time
    monthly_repeat = (
        df.groupby(['InvoiceMonth', 'CustomerID'])['InvoiceNo']
        .nunique()
        .reset_index()
        .groupby('InvoiceMonth')['CustomerID']
        .apply(lambda x: (x.value_counts() > 1).sum() / x.nunique())
        .reset_index(name='RepeatRate')
    )
    monthly_repeat['InvoiceMonth'] = monthly_repeat['InvoiceMonth'].astype(str)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_repeat, x='InvoiceMonth', y='RepeatRate', marker='o')
    plt.xticks(rotation=45)
    plt.title('Monthly Repeat Rate Over Time')
    plt.ylabel('Repeat Rate')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'monthly_repeat_trend.png'))
    plt.close()

    # 🏷️ Repeat purchase rate by segment (Country)
    segment_df = df.groupby(['Country', 'CustomerID'])['InvoiceNo'].nunique().reset_index()
    segment_df['IsRepeat'] = segment_df['InvoiceNo'] > 1
    repeat_rate_by_country = segment_df.groupby('Country')['IsRepeat'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=repeat_rate_by_country.values, y=repeat_rate_by_country.index)
    plt.xlabel('Repeat Purchase Rate')
    plt.ylabel('Country')
    plt.title('Repeat Rate by Country (Top 10)')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'repeat_by_country.png'))
    plt.close()

    # 💡 Impact of acquisition month on repeat behavior
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    repeat_customers_df = df.groupby(['CohortMonth', 'CustomerID'])['InvoiceNo'].nunique().reset_index()
    repeat_customers_df['IsRepeat'] = repeat_customers_df['InvoiceNo'] > 1
    repeat_rate_by_cohort = repeat_customers_df.groupby('CohortMonth')['IsRepeat'].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=repeat_rate_by_cohort, marker='o')
    plt.xticks(rotation=45)
    plt.ylabel('Repeat Rate')
    plt.title('Repeat Rate by Acquisition Month')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'repeat_by_acquisition_month.png'))
    plt.close()

    # 📉 Churn vs repeat rates side-by-side
    churn_vs_repeat = pd.Series({
        'Repeat Customers': repeat_customers,
        'Churned Customers': one_time_customers
    })

    plt.figure(figsize=(6, 5))
    sns.barplot(x=churn_vs_repeat.index, y=churn_vs_repeat.values, palette='Set2')
    plt.ylabel('Customer Count')
    plt.title('Churn vs Repeat Customers')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'churn_vs_repeat.png'))
    plt.close()

    print("🎉 Repeat purchase EDA charts saved to:", charts_dir)

    # 🔒 Save repeat vs churn data
    churn_vs_repeat_df = churn_vs_repeat.reset_index()
    churn_vs_repeat_df.columns = ['CustomerType', 'Count']
    churn_vs_repeat_df.to_csv(os.path.join(data_dir, 'repeat_vs_churn.csv'), index=False)

    # ✅ Return summary values
    return repeat_customers, one_time_customers



