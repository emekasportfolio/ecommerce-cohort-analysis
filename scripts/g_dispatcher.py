import itertools

class Dispatcher:
    def __init__(self):
        self.metric_base = [
            "cohort_retention_rate",
            "repeat_purchase_rate",
            "customer_lifetime_value",
            "time_to_first_repeat",
            "purchase_frequency",
            "average_order_value",
            "churn_rate"
        ]

        self.factor_base = [
            "geographic_trends",
            "order_size_quantity",
            "purchase_timing",
            "product_type",
            "price_sensitivity",
            "basket_size"
        ]

        self.metrics = self.metric_base + ['all_retention_metrics']
        self.factors = self.factor_base + ['all_factors', 'none']

        self.dispatch_map = {}
        self.build_dispatch_map()

    def expand(self, items, base_items, all_key):
        """
        Helper to expand 'all_retention_metrics' and 'all_factors'
        """
        expanded = []
        for item in items:
            if item == all_key:
                expanded.extend(base_items)
            elif item == 'none':
                continue
            else:
                expanded.append(item)
        return sorted(set(expanded))  # Ensure uniqueness and order

    def build_dispatch_map(self):
        # Generate all valid combinations of metrics (1 or more) and factors (0 or more)
        for r in range(1, len(self.metrics) + 1):
            for metric_combo in itertools.combinations(self.metrics, r):
                expanded_metrics = self.expand(metric_combo, self.metric_base, 'all_retention_metrics')
                for f in range(len(self.factors) + 1):
                    for factor_combo in itertools.combinations(self.factors, f):
                        expanded_factors = self.expand(factor_combo, self.factor_base, 'all_factors')
                        key = (tuple(sorted(expanded_metrics)), tuple(sorted(expanded_factors)))
                        func_name_parts = expanded_metrics + expanded_factors
                        func_name = "_".join(func_name_parts)
                        self.dispatch_map[key] = func_name

    def get_function(self, selected_metrics, selected_factors):
        """
        selected_metrics and selected_factors should be lists (may include 'all_retention_metrics' or 'all_factors').
        """
        expanded_metrics = self.expand(selected_metrics, self.metric_base, 'all_retention_metrics')
        expanded_factors = self.expand(selected_factors, self.factor_base, 'all_factors')
        key = (tuple(sorted(expanded_metrics)), tuple(sorted(expanded_factors)))
        return self.dispatch_map.get(key)
