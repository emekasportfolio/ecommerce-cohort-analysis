from scripts.computation import Computation
from scripts.g_dispatcher import Dispatcher
from scripts.eda_analysis import EDAAnalyst
from scripts.analyze import RegressionModel  # Ensure this import is correct

class Intro:
    def __init__(self, df):
        self.df = df
        self.condition = [
            'no condition', 'Country', 'cohort', 'CustomerID', 'Quantity',
            'UnitPrice', 'TotalPrice', 'Description', 'all condition except no condition'
        ]
        self.method = ['linear regression model', 'logistic regression']
        self.model_mapping = {
            'linear regression model': 'linear',
            'logistic regression': 'logistic'
        }

    def get_conditioned_groups(self):
        df = self.df.copy()

        print("Select a condition from the list below:")
        for cond in self.condition:
            print(f"- {cond}")
        selected = input("> ").strip()

        selected_lower = selected.lower()

        if selected_lower == "no condition":
            return {'All Data': df}

        elif selected_lower == "cohort":
            cohort_type = input("Weekly or Monthly cohort? ").strip().lower()
            if cohort_type == "weekly":
                df['cohort'] = df['InvoiceDate'].dt.to_period('W')
            elif cohort_type == "monthly":
                df['cohort'] = df['InvoiceDate'].dt.to_period('M')
            return dict(tuple(df.groupby('cohort')))

        elif selected_lower in ["Quantity", "UnitPrice","TotalPrice"]:
            col_name = 'TotalPrice' if selected_lower == 'totalprice' else selected.title()
            threshold = float(input(f"Enter a {col_name} threshold: "))
            above = df[df[col_name] >= threshold]
            below = df[df[col_name] < threshold]
            return {
                f'{col_name} >= {threshold}': above,
                f'{col_name} < {threshold}': below
            }

        elif selected in df.columns:
            return dict(tuple(df.groupby(selected)))

        else:
            print("Invalid condition. Using no condition.")
            return {'All Data': df}


    def run(self):
        retention_metrics = [
            "Cohort Retention", "Repeat Purchase Rate", "Customer Lifetime Value",
            "Time to First Repeat", "Purchase Frequency", "Average Order Value",
            "Churn Rate", "Order Time", "Order Size", "All Retention Metrics"
        ]

        retention_method_map = {
            1: 'cohort_retention_rate',
            2: 'repeat_purchase_rate',
            3: 'customer_lifetime_value',
            4: 'time_to_first_repeat',
            5: 'purchase_frequency',
            6: 'average_order_value',
            7: 'churn_rate',
            8: 'average_order_time',
            9: 'average_order_size',
            10: 'all_retention_metrics'
        }

        factor_options = [
            "Geographic Region", "Order Size", "Time of Purchase", "Product Type",
            "Price Sensitivity", "Basket Size", "All Retention Factors", "No Factor"
        ]

        factor_method_map = {
            1: 'geographic_trends',
            2: 'order_size_quantity',
            3: 'purchase_timing',
            4: 'product_type',
            5: 'price_sensitivity',
            6: 'basket_size',
            7: 'all_factors',
            8: 'none'
        }

        comp = Computation(self.df)

        # Step 1: Select retention metrics
        metric_choice = self.get_user_choice(retention_metrics, "\nSelect Retention Metrics to Analyze:")
        selected_metrics = [retention_method_map[idx] for idx in metric_choice]
        if 'all_retention_metrics' in selected_metrics:
            selected_metrics = [v for k, v in retention_method_map.items() if k != 10]

        # Step 2: Select retention factors
        factor_choice = self.get_user_choice(factor_options, "\nSelect Factors Affecting Retention to Analyze:")
        selected_factors = [factor_method_map[idx] for idx in factor_choice]
        if 'all_factors' in selected_factors:
            selected_factors = [v for v in factor_method_map.values() if v not in ['all_factors', 'none']]
        elif 'none' in selected_factors:
            selected_factors = []

        # Step 3: Run selected computations
        self.results = {}
        #for metric in selected_metrics:
         #   print(f"\n📌 Executing Retention Metric: {metric}")
          #  self.results[metric] = getattr(comp, metric)()

        # Step 3: Run selected computations
        self.results = {}
        for metric in selected_metrics:
            print(f"\n📌 Executing Retention Metric: {metric}")
            result = getattr(comp, metric)()

            # 👇 If the metric is cohort_retention_rate, also update self.df
            if metric == 'cohort_retention_rate':
                _, _, updated_df = result
                self.df = updated_df.copy()  # 👈 Ensure Intro has access to the enriched data

            self.results[metric] = result


        for factor in selected_factors:
            if factor != 'none':
                print(f"\n📌 Executing Retention Factor: {factor}")
                self.results[factor] = getattr(comp, factor)()




        # Step 4: Model training
        grouped = self.get_conditioned_groups()
        selected_model = input(f"Select a model:\n{self.method}\n> ").strip().lower()

        if selected_model not in self.model_mapping:
            print(f"Unknown model: {selected_model}")
        else:
            model_type = self.model_mapping[selected_model]
            #for label, data in grouped.items():
             #   print(f"\nRunning for group: {label}")
              #  try:
               #     model = RegressionModel(data, metrics=[selected_metrics], factors=selected_factors, model_type=model_type)
                #    results = model.run()
                 #   print("Results:", results)

            for label, data in grouped.items():
                print(f"\nRunning for group: {label}")

                # Drop NaNs from regression target(s)
                clean_data = data.dropna(subset=[m for m in selected_metrics if m in data.columns])

                if clean_data.empty:
                    print(f"⚠️ Skipping group '{label}' — no valid data after removing NaNs.")
                    continue

                try:
                    model = RegressionModel(clean_data, metrics=[selected_metrics], factors=selected_factors, model_type=model_type)
                    results = model.run()

                    # Optional prediction
                    predict_now = input("Would you like to make a prediction based on user input? (y/n): ").lower()
                    if predict_now == 'y':
                        model.predict(label=label)
                except Exception as e:
                    print(f"Error for group '{label}': {e}")

        print("\n🎉 Analysis complete! Check your results in the returned variables.")

    def get_user_choice(self, options, prompt):
        print(prompt)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")
        selected = input("\nEnter your choices (comma-separated): ")
        try:
            choices = [int(x.strip()) for x in selected.split(',')]
            valid_choices = [i for i in choices if 1 <= i <= len(options)]
            if not valid_choices:
                print("⚠️ Invalid input. Please try again.")
                return self.get_user_choice(options, prompt)
            return valid_choices
        except ValueError:
            print("⚠️ Invalid input. Please enter numbers only.")
            return self.get_user_choice(options, prompt)
