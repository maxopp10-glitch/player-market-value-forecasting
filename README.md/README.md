# Player Market Value Forecasting (Time-Aware Machine Learning)

**Applied Machine Learning project** focused on modeling and interpreting **short-term football players’ market value dynamics** using historical Transfermarkt data.

📊 **Problem Type:** Time-Series Regression  
⚽ **Domain:** Football Analytics / Sports Data Science  
🧠 **Models:** Random Forest, Gradient Boosting  
⏱️ **Core Constraint:** Strict temporal causality (no information leakage)

---

## Project Highlights

- **Time-aware forecasting of football players’ market value changes**, modeling the **future percentage change** (`target_pct_change`) using historical Transfermarkt data.
- **Strict temporal validation and leakage control**, with chronological train / validation / test splits and explicit removal of target-derived features.
- **Momentum signals (recent valuation changes)** identified as the strongest short-term predictors across all models and age groups.
- **Age shows a dominant but non-linear effect**, peaking during players’ prime years (~24–30) and declining in later career stages.
- **Realistic and interpretable performance**, reflecting noisy, asymmetric real-world valuation dynamics rather than over-optimistic metrics.

---

## Repository Structure

├── data/
│   ├── raw/transfermarkt/        # Raw source files
│   └── processed/                # Cleaned & feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_diagnostics_and_simulation.ipynb
│
├── reports/
│   └── figures/                  # Saved plots (G1–G6)
│
├── README.md
├── requirements.txt
└── .env

## Executive Summary

This project explores the dynamics of football players’ market value using **time-aware Machine Learning models** applied to historical Transfermarkt data.

The objective is to predict the **future percentage change in a player’s market value** (`target_pct_change`), while rigorously avoiding **information leakage** and preserving **temporal causality**.

Key findings show that:

- **Recent market momentum** (short-term percentage changes) is the strongest predictor of future valuation.
- **Age plays a dominant but non-linear role**, peaking during a player’s prime years (approximately 24–30).
- Absolute market value levels alone provide limited predictive power without temporal context.
- Even with advanced models, large valuation jumps remain inherently difficult to predict, reflecting the noisy and asymmetric nature of the problem.

Random Forest and Gradient Boosting models consistently outperform baselines, while maintaining **realistic performance levels** due to strict leakage controls and time-based splits.

Overall, the project emphasizes **methodological rigor, interpretability, and realistic expectations**, making it a real-world example of applied Machine Learning for time-series forecasting.

---

## Why This Matters

In real-world scenarios, player valuation impacts:
- Investment decisions by football clubs
- Contract negotiations by agents
- Risk assessment in player acquisition

Accurate forecasting of market value changes helps stakeholders:
- Identify undervalued players
- Anticipate performance trends
- Reduce financial risk

This project focuses not only on prediction, but on building a reliable and realistic modeling framework for time-dependent data.

---

## Player Market Value Forecasting (Transfermarkt)

This project applies **Machine Learning to time-series data** to model and predict the **future percentage change in football players’ market value**, using historical data from Transfermarkt.

The goal is not only predictive performance, but also to understand **which factors actually drive player valuation dynamics over time**.

---

## Project Objective

Predict the **future percentage change in market value (`target_pct_change`)** of a player based on:

- Historical market value levels  
- Recent value changes (momentum)  
- Trend and volatility signals  
- Player age  

Beyond prediction, the project aims to answer:

> **What truly drives player valuation dynamics over time?**

---

## Dataset

**Source:** Transfermarkt  

**Main files used:**
- `player_valuations.csv`
- `players.csv`

**Time span:**  
📅 2005-01-10 → 2025-01-09  

**Granularity:**
- Time-series observations per player  
- Each row represents a historical market valuation  

---

## Data Quality Decisions

Special attention was given to data reliability and consistency throughout the pipeline:

- Removal of inconsistent and invalid records  
- Strict enforcement of chronological ordering  
- Prevention of future information leakage  
- Validation of feature construction  

**Key principle:** No model is better than the data it is built on.

---

## Project Pipeline

### 1. Data Ingestion & Cleaning  
Notebook: `01_data_ingestion.ipynb`

- Type casting and validation  
- Temporal ordering by player  
- Removal of invalid records  
- Enforcement of time consistency  

---

### 2. Feature Engineering  
Notebook: `02_feature_engineering.ipynb`

Features derived from **time-series structure**:

#### Lag Features (Temporal Memory)
- `value_lag_1`, `value_lag_3`, `value_lag_6`
- `pct_change_lag_1`, `pct_change_lag_3`, `pct_change_lag_6`

#### Trend & Volatility
- `value_ma_3` (rolling mean)
- `value_std_3` (rolling standard deviation)

#### Demographics
- Player age (derived from birth date)

---

### 3. Time-Aware Data Splitting

Used consistently across all models:

- **Train:** older observations  
- **Validation:** intermediate window  
- **Test:** most recent data  

No shuffling — strict respect for **temporal causality**.

## Step 2 — Model Selection & Final Recommendation

This step documents **how the final predictive model was selected**, **why it was chosen**, and **how the results can be reproduced**, under strict time-aware and leakage-free conditions.

---

### Time-aware evaluation strategy

All models were evaluated using a **strict temporal split**, applied consistently across all experiments:

- **Train:** older observations  
- **Validation:** intermediate time window  
- **Test:** most recent observations  

No random shuffling was applied at any stage.  
This design strictly preserves **temporal causality**, ensuring that models never have access to future information during training.

---

### Baseline models (minimum performance benchmarks)

Two naive baselines were implemented to define a minimum acceptable performance:

1. **Zero-change baseline**  
   - Always predicts a `0%` change in market value.
2. **Last-change baseline**  
   - Predicts the most recent observed percentage change (`pct_change_lag_1`).

These baselines serve as **sanity checks** and help contextualize the performance of more complex models.

---

### Candidate machine learning models

Two non-linear ensemble models were trained and evaluated:

#### Random Forest Regressor
- Captures complex, non-linear feature interactions
- Robust to outliers and heavy-tailed distributions
- Well-suited for tabular, noisy data

#### Gradient Boosting Regressor
- Sequential additive ensemble model
- Strong bias control
- Effective when predictive signals are weak but consistent

Both models were evaluated on **validation and test sets** using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

---

### Information leakage control (critical validation)

Because the target variable represents a **future percentage change**, explicit leakage tests were conducted.

The following features were treated as **forbidden** and removed from the final models:

- `target_pct_change`
- `target_pct_change_capped`
- `next_value`

Results clearly show that:
- Including target-derived features leads to **unrealistically high performance**
- Removing them yields **realistic, stable metrics**, consistent with real-world uncertainty

All reported results and conclusions are based exclusively on **leakage-free models**.

---

### Final model recommendation

**Selected final model:**  
**Gradient Boosting Regressor (leakage-free)**

**Rationale:**
- Performance comparable to Random Forest under strict leakage control
- More stable learning behavior in weak-signal environments
- Feature importance aligns with domain intuition:
  - Short-term momentum (recent value changes)
  - Age as a non-linear structural factor
  - Past valuation trends

This model offers the best trade-off between  
**realism, interpretability, and robustness**.

---

### Reproducibility reference

All experiments related to this step are implemented in:

- `notebooks/03_model_diagnostics_and_simulation.ipynb`

Relevant notebook sections include:
- Temporal splitting
- Baseline evaluation
- Random Forest (with and without leakage)
- Gradient Boosting (with and without leakage)
- Final model comparison


## 4. Final Validation & Business Interpretation

This step consolidates the **final validation**, **model behavior analysis**, and **practical interpretation** of the results, translating technical findings into **real-world insights**.

The objective is not only to assess predictive accuracy, but to clearly define **how the model should — and should not — be used**.

---

### 4.1 Model Behavior Under Uncertainty

Residual diagnostics and error analysis reveal clear structural patterns:

- Prediction errors increase as the magnitude of market value changes grows.
- Small to moderate valuation changes are modeled with significantly higher reliability.
- Extreme valuation jumps exhibit heavy-tailed error behavior, reflecting intrinsic uncertainty.

These effects are visualized through:
- Residual distribution analysis
- Residuals vs predicted values
- Absolute error vs true target
- Error stratification by target quantiles
- Risk–coverage curves based on prediction confidence

This behavior confirms that **player market valuation dynamics are inherently noisy and asymmetric**, especially at the extremes.

---

### 4.2 Risk–Coverage Tradeoff (Model Confidence)

The Risk–Coverage Curve demonstrates a clear and desirable tradeoff:

- Restricting predictions to the most confident subsets significantly reduces absolute error.
- As coverage increases, prediction error grows smoothly and predictably.
- Median error remains consistently lower than mean error, reinforcing the presence of asymmetric risk.

This enables **confidence-aware decision making**, allowing stakeholders to:
- Focus on high-confidence predictions
- Explicitly trade off coverage versus risk
- Avoid overreliance on low-confidence forecasts

---

### 4.3 What the Model Is *Not* Predicting

It is critical to define the model’s limitations explicitly:

- The model does **not** account for unobserved external shocks such as:
  - Transfers
  - Injuries
  - Contract negotiations
  - Media exposure or sudden tactical changes
- Predictions are **not deterministic forecasts**, but conditional expectations given historical patterns.
- The model does **not replace scouting, domain expertise, or human judgment**.

These constraints are intentional and aligned with real-world deployment standards.

---

### 4.4 Recommended Practical Usage

The model is best suited for **decision support**, not decision automation.

Recommended applications include:
- Ranking players by expected short-term valuation momentum
- Identifying undervalued or overperforming profiles
- Supporting recruitment, portfolio monitoring, or market analysis workflows
- Scenario exploration under controlled uncertainty

Used correctly, the model provides **structured, data-driven signals** that complement qualitative evaluation.

---

### 4.5 Final Validation Conclusion

- All machine learning models outperform baseline benchmarks under strict temporal validation.
- Momentum (recent percentage changes) consistently emerges as the dominant predictive signal.
- Age acts as a strong structural factor, peaking during prime years (approximately 24–30).
- Random Forest and Gradient Boosting exhibit comparable performance under leakage-free conditions.
- The prediction task is **inherently noisy, asymmetric, and context-dependent**.

Overall, the project demonstrates **methodological rigor, interpretability, and realistic expectations**, making it suitable as a **real-world applied Machine Learning case study**.

## Models Evaluated

### Baseline Models
- Zero-change prediction  
- Last observed percentage change  

Serve as **minimal performance benchmarks**.

---

### Random Forest Regressor
- Non-linear ensemble model  
- Captures complex interactions  
- Robust to outliers  

Evaluated under multiple conditions:
- With age  
- Without age  
- Without any target-derived features (leakage control)

---

### Gradient Boosting Regressor
- Sequential additive model  
- Strong bias control  
- Effective for weak but consistent signals  

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R²** (Explained variance)

Metrics are reported separately for **validation** and **test** sets.

---

## Residual Analysis

Includes:
- Residual distribution  
- Residuals vs predicted values  
- Absolute error vs true target  
- Error by target quantile  

Clear evidence of:
- Heavy-tailed error distribution  
- Increasing error for extreme valuation changes  
- Structural difficulty in predicting large “value jumps”

---

## Feature Importance Insights

### Most Influential Signals
1. **Momentum (recent value changes)**
2. **Age**
3. **Past value trend**

### Key Insight
- Age is dominant **around the player’s prime (≈24–30 years)**  
- Its marginal effect decreases at later career stages  
- Momentum remains relevant across all age ranges  

---

## Information Leakage Control

Explicit tests were conducted using models:
- **With** target-derived features  
- **Without** target-derived features  

Results confirm:
- Leakage leads to unrealistically high performance  
- Clean models preserve explanatory power with realistic metrics  

---



---

## Limitations

- Valuations are not match-level  
- External shocks are unobserved (injuries, transfers, media exposure)  
- Target distribution is highly skewed  
- Extreme valuation spikes remain difficult to predict  

---

## Future Work

- Quantile regression models  
- Heteroscedastic error modeling  
- League and club context features  
- Performance-based metrics  
- SHAP-based explainability  

## Project Artifacts & Reproducibility

All experiments, evaluations, and visual diagnostics generated in this project are fully reproducible and stored in a structured manner.

### 📊 Generated Figures

The following plots were generated during model validation and diagnostics and are saved under:
- **G1 — Target Distribution**
  - `G1_target_distribution.png`
- **G2 — Residual Distribution**
  - `G2_residual_distribution.png`
- **G3 — Residuals vs Predicted Values**
  - `G3_residuals_vs_predicted.png`
- **G4 — Absolute Error vs True Target**
  - `G4_abs_error_vs_real.png`
- **G5 — Error by Target Quantile**
  - `G5_error_by_quantile.png`
- **G6 — Risk–Coverage Curve (Uncertainty Analysis)**
  - `G6_risk_coverage_curve.png`

These visualizations support:
- Error distribution analysis
- Detection of heteroscedasticity
- Model bias diagnostics
- Risk-aware evaluation under uncertainty

---

### 🔁 Reproducibility Notes

- All data splits are **time-aware**, preserving temporal causality
- No random shuffling is used at any stage
- Feature leakage was explicitly tested and controlled
- Results can be fully reproduced by running the notebooks in sequence:
### 🧪 Environment

- Python version and dependencies are listed in `requirements.txt`
- The project was developed using Jupyter Notebooks and scikit-learn
- File paths and outputs follow a consistent project structure



## Model Usage Notes

This project is designed as an **analytical and forecasting support tool**, not as a deterministic valuation engine.

### ✅ Appropriate Use Cases

The model is suitable for:

- Analyzing **drivers of short-term player valuation dynamics**
- Understanding how **recent market momentum** and **career stage** influence valuation changes
- Supporting **scenario analysis** and **risk-aware insights**
- Educational and research purposes involving **time-series Machine Learning**

---

## Production Considerations

If deployed in a real-world environment, the following improvements would be implemented:

- Automated data pipelines (ETL / ELT)
- Continuous model retraining with new data
- Monitoring for model drift
- Integration with external data sources (injuries, transfers, news)

This highlights the gap between experimental models and production-ready systems, and the steps required to bridge it.

---

### ⚠️ Limitations and Non-Recommended Uses

The model **should NOT** be used for:

- Directly setting player prices or transfer fees
- Contract negotiations or financial commitments without expert judgment
- Predicting exact future market values
- Long-term multi-season forecasting without additional contextual features

Large valuation jumps are often driven by **exogenous events** (injuries, transfers, media exposure, tactical role changes), which are **not observable** in the dataset.

---

### 🧠 Interpretation Guidance

- Predictions represent **expected percentage change tendencies**, not guaranteed outcomes
- Model confidence decreases for **extreme valuation changes**
- Outputs should always be interpreted **in context**, alongside qualitative football knowledge

This design choice prioritizes **realism, robustness, and interpretability** over over-optimistic predictive claims.

---

## Author

This project was developed as an **applied Machine Learning and time-series study**, with a strong focus on **methodological rigor, interpretability, and real-world constraints**.
