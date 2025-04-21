import itertools

# Retention metrics and corresponding method mappings
retention_metrics = [
    "Cohort Retention",
    "Repeat Purchase Rate",
    "Customer Lifetime Value",
    "Time to First Repeat",
    "Purchase Frequency",
    "Average Order Value",
    "Churn Rate"
]

# Retention factors and corresponding method mappings
factors = [
    "Geographic Region",
    "Order Size",
    "Time of Purchase",
    "Product Type",
    "Price Sensitivity",
    "Basket Size"
]

# Function to generate all combinations of metrics and factors
def generate_combinations():
    # Open a file to write the combinations and corresponding EDA
    with open("eda_combinations.txt", "w") as f:
        # Generate all possible combinations of retention metrics (1 to all 7)
        for r in range(1, len(retention_metrics) + 1):
            for metric_combo in itertools.combinations(retention_metrics, r):
                # For each combination of retention metrics, generate EDA for each combination of factors
                for factor_r in range(0, len(factors) + 1):
                    for factor_combo in itertools.combinations(factors, factor_r):
                        # Write the combination of metrics and factors to the file
                        f.write(f"Retention Metrics: {', '.join(metric_combo)}\n")
                        f.write(f"Factors: {', '.join(factor_combo) if factor_combo else 'None'}\n")
                        f.write("Exploratory Data Analysis:\n")
                        
                        # EDA for retention metrics combination
                        if "Cohort Retention" in metric_combo:
                            f.write("  - Analyze customer retention over time based on cohorts.\n")
                        if "Repeat Purchase Rate" in metric_combo:
                            f.write("  - Assess the repeat purchase behavior of customers.\n")
                        if "Customer Lifetime Value" in metric_combo:
                            f.write("  - Calculate the total value a customer brings over their lifetime.\n")
                        if "Time to First Repeat" in metric_combo:
                            f.write("  - Calculate the time it takes for customers to make their first repeat purchase.\n")
                        if "Purchase Frequency" in metric_combo:
                            f.write("  - Examine the frequency of purchases made by customers.\n")
                        if "Average Order Value" in metric_combo:
                            f.write("  - Analyze the average order value for customers.\n")
                        if "Churn Rate" in metric_combo:
                            f.write("  - Calculate the percentage of customers who stop purchasing over time.\n")
                        
                        # EDA for factors combination
                        if "Geographic Region" in factor_combo:
                            f.write("  - Analyze customer distribution across geographic regions.\n")
                        if "Order Size" in factor_combo:
                            f.write("  - Examine the impact of order size on retention and purchase behavior.\n")
                        if "Time of Purchase" in factor_combo:
                            f.write("  - Investigate how the timing of purchases influences customer behavior.\n")
                        if "Product Type" in factor_combo:
                            f.write("  - Explore how different product types affect customer retention.\n")
                        if "Price Sensitivity" in factor_combo:
                            f.write("  - Analyze how customer purchase behavior varies with price sensitivity.\n")
                        if "Basket Size" in factor_combo:
                            f.write("  - Investigate the impact of basket size on purchase frequency and retention.\n")
                        
                        f.write("\n")
                        f.write("="*50 + "\n")

if __name__ == "__main__":
    generate_combinations()
    print("EDA combinations saved to 'eda_combinations.txt'")
