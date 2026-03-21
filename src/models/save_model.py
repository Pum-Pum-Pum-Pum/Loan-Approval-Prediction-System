import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MODEL_THRESHOLD = 0.60


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
        ('log', FunctionTransformer(__import__('numpy').log1p, feature_names_out='one-to-one')),
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


def build_model() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')),
    ])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / 'data' / 'raw' / 'train.csv'
    artifact_dir = project_root / 'artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / 'loan_approval_model.joblib'

    df = pd.read_csv(train_path)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status'].map({'N': 0, 'Y': 1})

    model = build_model()
    model.fit(X, y)

    joblib.dump({'model': model, 'threshold': MODEL_THRESHOLD}, model_path)
    print(f'Model saved to: {model_path}')


if __name__ == '__main__':
    main()
