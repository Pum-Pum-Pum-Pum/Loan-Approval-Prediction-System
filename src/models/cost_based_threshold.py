import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
FP_COST = 10
FN_COST = 3


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


def build_weighted_logistic() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')),
    ])


def evaluate_thresholds(y_true: pd.Series, y_prob: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fp * FP_COST + fn * FN_COST

        rows.append({
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'fp_cost': fp * FP_COST,
            'fn_cost': fn * FN_COST,
            'total_cost': total_cost,
        })
    return pd.DataFrame(rows).sort_values(by='total_cost')


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

    model = build_weighted_logistic()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_valid)[:, 1]
    threshold_df = evaluate_thresholds(y_valid, y_prob, THRESHOLDS)

    print(f'Business Cost Setup -> FP cost: {FP_COST}, FN cost: {FN_COST}')
    print('\nCOST-BASED THRESHOLD ANALYSIS:')
    print(threshold_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    best_row = threshold_df.iloc[0]
    print('\nBEST THRESHOLD BY TOTAL COST:')
    print(best_row.to_string())


if __name__ == '__main__':
    main()
