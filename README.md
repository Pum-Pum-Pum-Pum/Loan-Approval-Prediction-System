# Loan Approval Prediction System

End-to-end machine learning project for loan approval prediction with a strong focus on **EDA**, **business-aware model evaluation**, **cost-based thresholding**, **A/B testing**, **API serving**, and **post-deployment drift monitoring**.

## 1. Project Goal
This project simulates a realistic lending-use-case where a model is built to predict loan approval outcomes from applicant profile and financial information.

The main goal is not only to train a classifier, but to demonstrate the full workflow expected in practical data science and applied ML roles:
- exploratory data analysis
- preprocessing pipelines
- baseline and tree-based modeling
- threshold tuning
- hypothesis testing
- business-cost-based model selection
- API deployment with FastAPI
- inference logging
- drift monitoring

---

## 2. Business Context
In lending, model selection is not just about maximizing accuracy.
Different types of mistakes have different business consequences:
- **False Positive (FP):** approving a risky applicant
- **False Negative (FN):** rejecting a good applicant

This project explicitly demonstrates that:
- the best metric is not always the best business choice
- threshold selection is a policy decision
- deployment and monitoring matter as much as modeling

---

## 3. Dataset
Dataset used:
- `data/raw/train.csv`
- `data/raw/test.csv`

Features include:
- demographic variables
- education and employment information
- income information
- loan amount and term
- credit history
- property area

Target:
- `Loan_Status` (`Y` / `N`)

---

## 4. Project Structure
```text
.
 data/raw/                         # raw training and test data
 notebooks/                        # EDA notebook
 artifacts/                        # saved model + inference logs
 src/models/                       # model training, evaluation, deployment scripts
    train_baseline.py
    compare_models.py
    tune_tree_models.py
    cost_based_threshold.py
    hypothesis_testing.py
    ab_test_simulation.py
    save_model.py
    api_app.py
    drift_checking.py
    linear_regression_scratch.py
    logistic_regression_scratch.py
 README.md
```

---

## 5. Key Workflow Covered
### A. Exploratory Data Analysis
Performed in notebook:
- data inspection
- missing value analysis
- distribution analysis
- outlier analysis
- categorical target relationships
- correlation review
- segment analysis

### B. Hypothesis Testing
Implemented in:
- `src/models/hypothesis_testing.py`

Includes:
- chi-square tests
- t-tests
- ANOVA

### C. Baseline Modeling
Implemented in:
- `src/models/train_baseline.py`

Includes:
- weighted logistic regression
- threshold analysis
- confusion matrix interpretation
- business-cost tradeoff

### D. Model Comparison
Implemented in:
- `src/models/compare_models.py`
- `src/models/tune_tree_models.py`

Compared:
- Logistic Regression
- Decision Tree
- Random Forest
- tuned tree-based models via GridSearchCV

### E. Cost-Based Threshold Optimization
Implemented in:
- `src/models/cost_based_threshold.py`

Shows how threshold choice changes:
- precision
- recall
- false positives
- false negatives
- total business cost

### F. A/B Test Simulation
Implemented in:
- `src/models/ab_test_simulation.py`

Simulates a controlled experiment and evaluates:
- uplift
- z-statistic
- p-value
- statistical significance

### G. FastAPI Deployment
Implemented in:
- `src/models/save_model.py`
- `src/models/api_app.py`

Exposes:
- `GET /health`
- `POST /predict`

### H. Inference Logging
Prediction requests and responses are logged to:
- `artifacts/inference_logs/predictions.jsonl`

### I. Drift Monitoring
Implemented in:
- `src/models/drift_checking.py`

Includes:
- PSI for numeric drift
- categorical distribution shift comparison
- production-style drift summary

---

## 6. Main Modeling Insight
A key lesson from this project:

> The model with the best raw metric is not always the best deployment model.

In this project:
- Logistic Regression remained a strong and interpretable candidate
- threshold tuning materially changed business behavior
- cost-based evaluation gave a different answer from metric maximization
- deployment considerations favored a simpler, explainable model

---

## 7. Deployment Choice
The deployed API currently uses:
- **weighted Logistic Regression**
- stored threshold = **0.60**

Why?
- competitive performance
- easier interpretability
- clean threshold story
- better explainability for lending use cases
- simpler deployment path

This model acts as the **champion model**.
A tree-based model can later be introduced as a **challenger model**.


## Future Improvements
Possible next improvements:
- add challenger model deployment path
- persist inference logs to SQLite/PostgreSQL
- add score drift monitoring
- build automated retraining trigger logic
- package monitoring into scheduled jobs
- add unit tests for API and preprocessing
