#  Customer Cohort Analysis & Prediction Tool

This tool helps you analyze customer behavior over time and predict future purchases using cohort-based analytics, clustering, and machine learning.

---

##  Features

- Cohort analysis by time, product, country, and revenue
- Heatmaps and trend charts
- Clustering to identify customer segments
- Machine Learning to predict next purchases
- Saves clean datasets and reports to `charts/` and `outputs/`

---

##  How to Use

### 1. Choose Cohort Granularity
Define how to group the cohort periods:

```
Select cohort granularity ('monthly' or 'weekly'):
> monthly
```

---

### 2. Select Grouping Criteria
Choose how to segment your cohort reports. Use one or more of the options below (comma-separated):

- **a** → Group by **Revenue**
- **b** → Group by **Product** (Description)
- **c** → Group by **Country**
- **d** → Group by **Customer ID**

```
Enter choices (e.g. a,b,d):
> b,c
```

---

### 3. Apply Optional Filters (If Chosen)
If you selected **Product** or **Country**, you can filter to specific values.

The tool will show you a numbered list. You can pick one or more items by entering their indexes (e.g., `1,3,5`).

```
Select index/indices for Country (comma-separated):
> 2,4
```

---

## Output

The tool saves results based on your selections:

###  In `charts/` Directory:
- Heatmap of retention
- Trend over time chart
- Customer segmentation clusters
- Feature importance from ML model

###  In `outputs/` Directory:
- Cleaned cohort data (`.csv`)
- ML classification report (`.txt`)
- Predicted next purchases (`.csv`)

---

## 🔧 Rerun to Explore More!
Try different combinations (e.g., by Product + Country, or only Revenue) to explore more customer insights.

---

##  Requirements
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Install dependencies:
```
pip install -r requirements.txt
```

---

Happy Analyzing! 🤖

 

