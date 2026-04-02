# Customer Behavior & Revenue Intelligence Analysis

An end-to-end data science project that applies exploratory data analysis, feature engineering, and unsupervised machine learning to segment customers and generate actionable business insights from behavioral and demographic data.

---

## Overview

This project analyzes a dataset of 800 customer records to uncover behavioral patterns, identify distinct customer segments, and translate analytical findings into strategic business recommendations. The full pipeline covers data cleaning, EDA, feature engineering, K-Means clustering, dimensionality reduction, and cluster profiling.

---

## Dataset

The dataset contains 800 customer-level records with 11 attributes spanning demographics, financials, and behavioral metrics.

| Feature | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `gender` | Customer gender |
| `annual_income` | Annual income (USD) |
| `spending_score` | Behavioral spending score |
| `purchase_frequency` | Number of purchases per period |
| `avg_transaction_value` | Average value per transaction (USD) |
| `product_category` | Primary product category |
| `tenure_months` | Customer tenure in months |
| `online_engagement_score` | Digital engagement score |
| `returns_ratio` | Proportion of returned purchases |

> **Note:** The dataset is synthetically generated for portfolio and learning purposes.

---

## Project Structure

```
├── data/
│   └── customer_data_v2.csv
├── outputs/
│   └── charts/
│       ├── distribution_characteristics.png
│       ├── correlation_heatmap.png
│       ├── pairplot.png
│       ├── boxplots.png
│       ├── income_vs_spending_score.png
│       ├── elbow_method.png
│       ├── pca_cluster_scatter.png
│       └── cluster_profile_boxplots.png
├── Customer_Behavior___Revenue_Intelligence_Analysis.ipynb
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- Imputed missing values in `annual_income` (median) and `online_engagement_score` (mean), with justification for each strategy
- Applied IQR-based outlier capping on `annual_income` and `avg_transaction_value` to preserve high-value customer records rather than drop them

### 2. Exploratory Data Analysis
- Univariate distributions (histogram + KDE) across all 7 numerical features
- Correlation heatmap to identify feature relationships
- Pairplot for multi-dimensional scatter and density inspection
- Boxplots to profile spread and outlier behavior
- Bivariate scatter of income vs. spending score to motivate segmentation

### 3. Feature Engineering
Three composite features were derived to enrich the modeling:

| Feature | Formula | Business Purpose |
|---|---|---|
| `customer_value` | `purchase_frequency × avg_transaction_value` | Estimates total revenue contribution |
| `risk_score` | `online_engagement_score × returns_ratio` | Identifies high-engagement, high-return risk customers |
| `value_efficiency` | `customer_value / (online_engagement_score + 1)` | Measures how effectively engagement converts to revenue |

### 4. Customer Segmentation (K-Means Clustering)
- Feature selection: `annual_income`, `spending_score`, `online_engagement_score`, `returns_ratio`
- StandardScaler applied to normalize all features before distance-based modeling
- Optimal K determined via the Elbow Method → **K = 4**
- Cluster quality evaluated with **Silhouette Score**
- PCA applied for 2D cluster visualization

---

## Customer Segments

| Cluster | Segment Name | Characteristics |
|---|---|---|
| 0 | Premium Customers | High income, high spending — top revenue contributors |
| 1 | Moderate-Income, Low-Spending | Average income, low engagement — activation opportunity |
| 2 | High-Income, Low-Spending | High income, low spending — high untapped revenue potential |
| 3 | Budget Customers | Low income, low spending — retention focus |

---

## Key Findings

- **High income does not guarantee high spending.** A notable segment of high-income customers exhibit low spending behavior, representing a significant untapped revenue opportunity.
- **Premium customers** (Cluster 0) are the highest-value segment and warrant dedicated retention and loyalty investment.
- **Online engagement does not strongly predict spending.** The weak correlation between `online_engagement_score` and `spending_score` suggests that digital engagement alone is insufficient for revenue growth.
- **Returns risk** is concentrated among a small minority of customers but is disproportionately associated with high engagement profiles.

---

## Strategic Recommendations

1. **Retain Premium Customers** — Invest in loyalty programs, exclusive offers, and personalized experiences for Cluster 0.
2. **Activate High-Income, Low-Spending Customers** — Target Cluster 2 with personalized incentives and behavioral nudges to convert latent income into spending.
3. **Re-engage Moderate Customers** — Use targeted campaigns to increase purchase frequency among Cluster 1.
4. **Monitor High-Risk Customers** — Flag customers with high `risk_score` values for proactive intervention to reduce return rates.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and analysis |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Static visualizations and plot layouts |
| `seaborn` | Statistical visualization (heatmap, boxplot, pairplot) |
| `scikit-learn` | StandardScaler, KMeans, PCA, silhouette_score |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-behavior-analysis.git
   cd customer-behavior-analysis
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Customer_Behavior___Revenue_Intelligence_Analysis.ipynb
   ```

> Make sure the data file is located at `data/customer_data_v2.csv` relative to the notebook.

---

## Limitations

- The dataset is synthetically generated and may not fully reflect real-world customer behavior distributions.
- K-Means assumes spherical clusters and is sensitive to feature scale — alternative algorithms (e.g., DBSCAN, Agglomerative Clustering) were not evaluated.
- The silhouette score (~0.17–0.20) indicates moderate cluster overlap, which is expected in real-world behavioral data.
- Feature selection for clustering excludes the engineered features (`customer_value`, `risk_score`, `value_efficiency`), which may improve segmentation quality if included.

---

## Author

**Roy**
Data Analytics & Data Science Portfolio Project

---

## License

This project is for educational and portfolio purposes. The dataset is synthetic and does not contain any real personal information.
