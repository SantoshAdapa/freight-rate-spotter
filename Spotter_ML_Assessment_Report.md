# Spotter ML Engineer Assessment Report

## 1. Validation Strategy & Data Split

My main goal was to avoid data leakage. Since freight markets are heavily influenced by seasonality and macro trends, doing a standard random train/test split would artificially inflate the validation metrics (because the model would be peeking at future market signals to predict past rates). 

Instead, I went with a strict time-based holdout:
* **Train Set (Jan – Aug 2025):** 38,477 records. I used this exclusively to build target encodings and train the model.
* **Validation Set (Sep – Oct 2025):** 9,523 records. I used this to evaluate how well the model generalizes to future, unseen market conditions.

The final LightGBM ensemble achieved the following metrics on the Sep-Oct holdout:
* **MAE:** $141.19
* **RMSE:** $629.65
* **MAPE:** 6.39%
* **R²:** 0.829

Given the 6.39% MAPE on strict future data, I'm highly confident these predictions will generalize well to the final 12,000 validation loads.

---

## 2. Model Pipeline & Features

To get to that performance, I focused on a few key areas of feature engineering:

1. **Geographic routing:** I calculated Haversine distances and a "circuity ratio" (actual road distance vs. straight-line distance) to proxy routing complexity.
2. **Target Encoding:** I used Bayesian smoothed target encoding for the pickup/delivery cities and specific route pairs. The smoothing factor was critical to stop the trees from overfitting on rare routes.
3. **Signal Engineering:** I built baseline rate-per-mile estimates and added interaction terms (like `distance` × `quote_signal` and `distance` × `market_index`) which gave the model a huge lift.
4. **The Model:** The final model is a simple ensemble of two LightGBM regressors with different random seeds. Averaging them smoothed out the variance nicely.
5. **December Forecast Imputation:** For the December input file, we were missing a few columns present in the training set (coordinates, market index, etc). I handled this by imputing the training set medians and manually mapping the coordinates for Lexington and Fort Wayne so the model could generate the required 31-day forecast.

---

## 3. December Prediction Chart

Here is the forecast for the fixed Lexington-to-Fort Wayne route for December 2025:

![Candidate December](candidate_december.png)

**Thoughts on the output:**
The model establishes a stable baseline rate for the route while reacting dynamically to the temporal features. You can clearly see the expected late-December rate spike as the model accounts for the typical holiday capacity crunch in the freight market.
