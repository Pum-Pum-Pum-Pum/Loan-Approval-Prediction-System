import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


RANDOM_STATE = 35
CV_FOLDS = 5
SCORING = 'roc_auc'


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    return df


def get_feature_groups() -> tuple[list[str], list[str]]:
    numeric_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'TotalIncome',
    ]
    categorical_features = [
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'Property_Area',
    ]
    return numeric_features, categorical_features


def build_tree_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])


def build_decision_tree_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_tree_preprocessor()),
        ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_tree_preprocessor()),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def evaluate_on_validation(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    return {
        'accuracy': accuracy_score(y_valid, y_pred),
        'precision': precision_score(y_valid, y_pred, zero_division=0),
        'recall': recall_score(y_valid, y_pred, zero_division=0),
        'f1': f1_score(y_valid, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_valid, y_prob),
        'confusion_matrix': confusion_matrix(y_valid, y_pred),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / 'data' / 'raw' / 'train.csv'

    df = load_data(train_path)
    df = build_features(df)

    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status'].map({'N': 0, 'Y': 1})

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    dt_pipeline = build_decision_tree_pipeline()
    rf_pipeline = build_random_forest_pipeline()

    dt_param_grid = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [3, 4, 5, 6, None],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 3, 5, 10],
    }

    rf_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [4, 6, 8, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 3, 5],
        'classifier__max_features': ['sqrt', 'log2', None],
    }

    dt_search = GridSearchCV(
        estimator=dt_pipeline,
        param_grid=dt_param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print('Running GridSearchCV for Decision Tree...')
    dt_search.fit(X_train, y_train)

    print('Running GridSearchCV for Random Forest...')
    rf_search.fit(X_train, y_train)

    dt_valid_metrics = evaluate_on_validation(dt_search.best_estimator_, X_valid, y_valid)
    rf_valid_metrics = evaluate_on_validation(rf_search.best_estimator_, X_valid, y_valid)

    print('\nBEST DECISION TREE PARAMETERS:')
    print(dt_search.best_params_)
    print(f"Best CV ROC-AUC: {dt_search.best_score_:.4f}")
    print('Validation Metrics:')
    for key, value in dt_valid_metrics.items():
        if key != 'confusion_matrix':
            print(f'  {key}: {value:.4f}')
    print('  confusion_matrix:')
    print(dt_valid_metrics['confusion_matrix'])

    print('\nBEST RANDOM FOREST PARAMETERS:')
    print(rf_search.best_params_)
    print(f"Best CV ROC-AUC: {rf_search.best_score_:.4f}")
    print('Validation Metrics:')
    for key, value in rf_valid_metrics.items():
        if key != 'confusion_matrix':
            print(f'  {key}: {value:.4f}')
    print('  confusion_matrix:')
    print(rf_valid_metrics['confusion_matrix'])

    comparison_df = pd.DataFrame([
        {
            'model': 'tuned_decision_tree',
            'best_cv_roc_auc': dt_search.best_score_,
            'valid_accuracy': dt_valid_metrics['accuracy'],
            'valid_precision': dt_valid_metrics['precision'],
            'valid_recall': dt_valid_metrics['recall'],
            'valid_f1': dt_valid_metrics['f1'],
            'valid_roc_auc': dt_valid_metrics['roc_auc'],
        },
        {
            'model': 'tuned_random_forest',
            'best_cv_roc_auc': rf_search.best_score_,
            'valid_accuracy': rf_valid_metrics['accuracy'],
            'valid_precision': rf_valid_metrics['precision'],
            'valid_recall': rf_valid_metrics['recall'],
            'valid_f1': rf_valid_metrics['f1'],
            'valid_roc_auc': rf_valid_metrics['roc_auc'],
        },
    ]).sort_values(by='best_cv_roc_auc', ascending=False)

    print('\nTUNED MODEL COMPARISON:')
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
