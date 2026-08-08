# Space Launch Cost Prediction

Regression model predicting the cost of a space mission from launch metadata — company, country, rocket status, and launch year — using data on 4,300+ historical space missions (1957–2020).

## Problem
Launch cost data (`Rocket` column) was missing for ~78% of records (3,360 of 4,324 rows), so the dataset was filtered to the 964 launches with reported costs. The goal: predict cost from the remaining metadata, and use regularization to understand which features matter and to address overfitting.

## Data
- Source: Space_Corrected.csv — all space missions from 1957 onward
- Final modeling dataset: 964 rows, 27 features after cleaning
- Target: `Rocket` (launch cost, $M)

## Feature Engineering
- Extracted `Year` from launch date
- Extracted `Country` from launch location string
- Bucketed companies with fewer than 10 launches into `Other` to prevent one-hot column explosion
- One-hot encoded `Company_Grouped`, `Country`, `Status Rocket` (with `drop_first=True` to avoid the dummy variable trap)
- Scaled features with `StandardScaler` before fitting Ridge/Lasso

## Models & Results
80/20 train-test split, evaluated on held-out test data:

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression (baseline) | 0.458 | $279.33M |
| Ridge Regression (L2, α=10.0) | 0.395 | $295.23M |
| Lasso Regression (L1, α=1.0) | 0.404 | $292.92M |

## Key Findings
- **Baseline overfitting:** Train R² (0.580) notably exceeds Test R² (0.458), indicating the linear model partly fits noise in the training set.
- **Regularization underperformed at these alpha values:** Both Ridge and Lasso scored lower than baseline on test data. This suggests α=10.0 (Ridge) and α=1.0 (Lasso) over-penalized the model on a relatively small dataset (964 rows), shrinking useful signal along with noise rather than improving generalization.
- **Lasso feature selection:** Of 27 features, Lasso zeroed out 6 — mostly single-country indicators (`Country_Japan`, `Country_New Zealand`, `Country_Pacific Missile Range Facility`, `Country_USA`, `Country_Yellow Sea`) and one low-frequency company (`Company_Grouped_Rocket Lab`). This indicates these specific geographic/company effects had limited independent predictive value once other features were controlled for — useful for interpretability even though it didn't improve raw accuracy here.
- **Takeaway:** Regularization strength needs tuning (e.g. cross-validated alpha search) rather than fixed values — a natural next step.

## Visualizations
- Predicted vs. actual scatter plots for all three models
- Dual-axis R² / RMSE comparison bar chart across models

## Tech Stack
Python, Pandas, scikit-learn, Matplotlib, Seaborn

## Next Steps
- Use `RidgeCV`/`LassoCV` to select alpha via cross-validation instead of fixed values
- Address the 78% missing-cost problem with a larger or supplementary cost dataset
- Try log-transforming cost (`Rocket`) to reduce the influence of high-cost outliers on RMSE
