import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    return df


def build_preprocessor() -> ColumnTransformer:
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

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])


def build_model(class_weight=None) -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)),
    ])


def get_metrics(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    return {
        'accuracy': accuracy_score(y_valid, y_pred),
        'precision': precision_score(y_valid, y_pred),
        'recall': recall_score(y_valid, y_pred),
        'f1': f1_score(y_valid, y_pred),
        'roc_auc': roc_auc_score(y_valid, y_prob),
        'confusion_matrix': confusion_matrix(y_valid, y_pred),
        'classification_report': classification_report(y_valid, y_pred),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f'\n{name}')
    print('-' * len(name))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print('\nConfusion Matrix:')
    print(metrics['confusion_matrix'])
    print('\nClassification Report:')
    print(metrics['classification_report'])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / 'data' / 'raw' / 'train.csv'

    df = load_data(train_path)
    df = build_features(df)

    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status'].map({'N': 0, 'Y': 1})

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline_model = build_model(class_weight=None)
    weighted_model = build_model(class_weight='balanced')

    baseline_model.fit(X_train, y_train)
    weighted_model.fit(X_train, y_train)

    baseline_metrics = get_metrics(baseline_model, X_valid, y_valid)
    weighted_metrics = get_metrics(weighted_model, X_valid, y_valid)

    print_metrics('Baseline Logistic Regression', baseline_metrics)
    print_metrics('Weighted Logistic Regression', weighted_metrics)

    comparison = pd.DataFrame([
        {
            'model': 'baseline_logistic',
            'accuracy': baseline_metrics['accuracy'],
            'precision': baseline_metrics['precision'],
            'recall': baseline_metrics['recall'],
            'f1': baseline_metrics['f1'],
            'roc_auc': baseline_metrics['roc_auc'],
        },
        {
            'model': 'weighted_logistic',
            'accuracy': weighted_metrics['accuracy'],
            'precision': weighted_metrics['precision'],
            'recall': weighted_metrics['recall'],
            'f1': weighted_metrics['f1'],
            'roc_auc': weighted_metrics['roc_auc'],
        },
    ])

    print('\nMetric Comparison:')
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
