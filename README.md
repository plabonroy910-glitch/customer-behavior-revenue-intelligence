# Customer Behavior & Revenue Intelligence Analysis

A data-driven customer segmentation project designed to identify high-value customer groups, uncover hidden revenue opportunities, and support strategic decision-making using machine learning.

By analyzing behavioral, financial, and engagement data, this project reveals actionable insights that can help businesses optimize marketing efforts, improve customer retention, and maximize revenue potential.

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
- Feature selection: all 7 features — original attributes (`annual_income`, `spending_score`, `online_engagement_score`, `returns_ratio`) plus engineered features (`customer_value`, `risk_score`, `value_efficiency`)
- StandardScaler applied to normalize all features before distance-based modeling
- Optimal K determined via the Elbow Method → **K = 4**
- Cluster quality evaluated with **Silhouette Score: 0.1471**
- PCA applied for 2D cluster visualization

---

## Cluster Distribution & Business Value

| Cluster | Segment Name | Business Priority |
|--------|-------------|------------------|
| 0 | Premium Customers | High — Core revenue drivers |
| 1 | Moderate Customers | Medium — Growth opportunity |
| 2 | High-Income, Low-Spending | Very High — Untapped revenue |
| 3 | Budget Customers | Low — Retention focus |

---

## Key Findings

- **High income does not guarantee high spending.** A notable segment of high-income customers exhibit low spending behavior, representing a significant untapped revenue opportunity.
- **Premium customers** (Cluster 0) are the highest-value segment and warrant dedicated retention and loyalty investment.
- **Online engagement does not strongly predict spending.** The weak correlation between `online_engagement_score` and `spending_score` suggests that digital engagement alone is insufficient for revenue growth.
- **Returns risk** is concentrated among a small minority of customers but is disproportionately associated with high engagement profiles.
- **Engineered features sharpen segmentation.** Including `customer_value`, `risk_score`, and `value_efficiency` in the clustering model reveals behavioral distinctions — such as high-engagement, high-return customers — that were invisible using raw attributes alone.

---

## Business Decision Framework

Each customer segment is mapped to a concrete business decision, 
data-backed justification, and measurable KPI.

| Cluster | Segment | Business Decision | Expected Outcome | KPI |
|---|---|---|---|---|
| 0 | Premium Customers | Launch Premium Loyalty Program | 15–20% retention increase, CLV growth | Retention rate, CLV, purchase frequency |
| 1 | Moderate, Low-Spending | Re-engagement & Frequency Program | 25–35% increase in purchase frequency | Purchase frequency, reactivation rate |
| 2 | High-Income, Low-Spending | Personalized Conversion Campaigns | Convert 20–30% into active spenders | Conversion rate, revenue per customer |
| 3 | Budget / At-Risk | Satisfaction Monitoring & Return Reduction | 15–25% reduction in return rate | Return rate, CSAT score |

> The combined execution of all four decisions creates a full-funnel 
> customer strategy — protecting existing revenue, activating latent 
> revenue, recovering dormant revenue, and reducing operational cost.
```

---

Also update your **Conclusion cell (Section 14)** by adding this line at the end:
```
A dedicated Business Decision Framework was developed to translate 
each customer segment into a concrete, KPI-driven business action — 
bridging the gap between analytical findings and real-world 
implementation. This ensures the project delivers not just insights, 
but decisions.

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
   git clone https://github.com/plabonroy910-glitch/customer-behavior-revenue-intelligence.git
   cd customer-behavior-revenue-intelligence
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
- Engineered features are derived from existing columns and do not introduce external data. Their inclusion improves behavioral granularity but does not expand the underlying information in the dataset.

---

## Author

**Plabon Roy**  
Aspiring Data Analyst | Business Analytics Enthusiast  
Focused on building data-driven solutions for real-world business problems

---

## License

This project is for educational and portfolio purposes. The dataset is synthetic and does not contain any real personal information.
