# Space Launch Cost Prediction

Regression model predicting the cost of a space mission from launch metadata — company, country, rocket status, and launch year — using data on 4,300+ historical space missions (1957–2020).

## Problem
Launch cost data (`Rocket` column) was missing for ~78% of records, so the dataset was filtered to the ~964 launches with reported costs. The goal: predict cost from the remaining metadata, and use regularization to understand which features actually matter.

## Data
- Source: [Space_Corrected.csv](link-to-kaggle-dataset) — all space missions from 1957 onward
- Final dataset: 964 rows, 27 features after cleaning
- Target: `Rocket` (launch cost, $M)

## Feature Engineering
- Extracted `Year` from launch date
- Extracted `Country` from launch location
- Bucketed companies with fewer than 10 launches into `Other` to prevent one-hot explosion
- One-hot encoded `Company_Grouped`, `Country`, `Status Rocket` (with `drop_first=True` to avoid the dummy variable trap)

## Models
Three models were trained and compared on an 80/20 train-test split:

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression (baseline) | [X] | $[X]M |
| Ridge Regression (L2) | [X] | $[X]M |
| Lasso Regression (L1) | [X] | $[X]M |

## Key Findings
- The baseline model showed a Train/Test R² gap of [X], indicating overfitting on the training set
- Regularization (Ridge/Lasso) reduced this gap, trading a small amount of test performance for a more generalizable model
- Lasso zeroed out [X] of 27 features, including [list 2-3 examples, e.g. `Country_Japan`, `Company_Grouped_Rocket Lab`] — suggesting these had little independent predictive value once other features were accounted for

## Visualizations
- Predicted vs. actual scatter plots for all three models
- R² / RMSE comparison bar chart across models

## Tech Stack
Python, Pandas, scikit-learn, Matplotlib, Seaborn

## What I'd improve with more data
Launch cost was missing for most of the dataset — a larger or more complete cost dataset would substantially improve model reliability.
