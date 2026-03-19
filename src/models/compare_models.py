import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RANDOM_STATE = 42


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


def build_logistic_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups()

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


def build_models() -> dict:
    return {
        'logistic_regression': Pipeline(steps=[
            ('preprocessor', build_logistic_preprocessor()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        'decision_tree': Pipeline(steps=[
            ('preprocessor', build_tree_preprocessor()),
            ('classifier', DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', build_tree_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> dict:
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    valid_prob = model.predict_proba(X_valid)[:, 1]

    return {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'valid_accuracy': accuracy_score(y_valid, valid_pred),
        'precision': precision_score(y_valid, valid_pred),
        'recall': recall_score(y_valid, valid_pred),
        'f1': f1_score(y_valid, valid_pred),
        'roc_auc': roc_auc_score(y_valid, valid_prob),
        'confusion_matrix': confusion_matrix(y_valid, valid_pred),
        'fitted_model': model,
    }


def get_feature_importance(model: Pipeline) -> pd.DataFrame | None:
    classifier = model.named_steps['classifier']
    if not hasattr(classifier, 'feature_importances_'):
        return None

    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values(by='importance', ascending=False)

    return importance_df.head(10)


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

    models = build_models()
    results = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_train, X_valid, y_train, y_valid)
        results.append({
            'model': model_name,
            'train_accuracy': metrics['train_accuracy'],
            'valid_accuracy': metrics['valid_accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
        })

        print(f'\n{model_name.upper()}')
        print('-' * len(model_name))
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Valid Accuracy: {metrics['valid_accuracy']:.4f}")
        print(f"Precision     : {metrics['precision']:.4f}")
        print(f"Recall        : {metrics['recall']:.4f}")
        print(f"F1-score      : {metrics['f1']:.4f}")
        print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")
        print('Confusion Matrix:')
        print(metrics['confusion_matrix'])

        importance_df = get_feature_importance(metrics['fitted_model'])
        if importance_df is not None:
            print('\nTop Feature Importances:')
            print(importance_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    comparison_df = pd.DataFrame(results).sort_values(by='roc_auc', ascending=False)
    print('\nMODEL COMPARISON')
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
