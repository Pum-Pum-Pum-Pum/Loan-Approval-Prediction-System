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


def build_pipeline() -> Pipeline:
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

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ])


def evaluate_model(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> None:
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    print(f'Accuracy : {accuracy_score(y_valid, y_pred):.4f}')
    print(f'Precision: {precision_score(y_valid, y_pred):.4f}')
    print(f'Recall   : {recall_score(y_valid, y_pred):.4f}')
    print(f'F1-score : {f1_score(y_valid, y_pred):.4f}')
    print(f'ROC-AUC  : {roc_auc_score(y_valid, y_prob):.4f}')
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_valid, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_valid, y_pred))


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

    model = build_pipeline()
    model.fit(X_train, y_train)
    evaluate_model(model, X_valid, y_valid)

if __name__ == '__main__':
    main()
