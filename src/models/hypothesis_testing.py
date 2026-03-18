import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


CATEGORICAL_COLS = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
]

NUMERIC_COLS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
]

ANOVA_GROUP_COLS = ['Property_Area', 'Education']
ANOVA_NUMERIC_COLS = ['ApplicantIncome', 'LoanAmount']


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def run_chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_COLS:
        subset = df[[col, 'Loan_Status']].copy()
        subset[col] = subset[col].fillna('Missing')
        contingency = pd.crosstab(subset[col], subset['Loan_Status'])
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        rows.append({
            'feature': col,
            'chi2_stat': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant_0.05': p_value < 0.05,
        })
    return pd.DataFrame(rows).sort_values(by='p_value')


def run_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    approved = df[df['Loan_Status'] == 'Y']
    rejected = df[df['Loan_Status'] == 'N']

    for col in NUMERIC_COLS:
        y_vals = approved[col].dropna()
        n_vals = rejected[col].dropna()
        t_stat, p_value = ttest_ind(y_vals, n_vals, equal_var=False)
        rows.append({
            'feature': col,
            'mean_approved': y_vals.mean(),
            'mean_rejected': n_vals.mean(),
            't_stat': t_stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
        })
    return pd.DataFrame(rows).sort_values(by='p_value')


def run_anova_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ANOVA_GROUP_COLS:
        for num_col in ANOVA_NUMERIC_COLS:
            subset = df[[group_col, num_col]].dropna()
            groups = [g[num_col].values for _, g in subset.groupby(group_col) if len(g) > 1]

            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)
                rows.append({
                    'group_col': group_col,
                    'numeric_col': num_col,
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'significant_0.05': p_value < 0.05,
                })
    return pd.DataFrame(rows).sort_values(by='p_value')


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / 'data' / 'raw' / 'train.csv'

    df = load_data(train_path)

    chi_df = run_chi_square_tests(df)
    ttest_df = run_t_tests(df)
    anova_df = run_anova_tests(df)

    print('\nCHI-SQUARE TESTS: categorical feature vs Loan_Status')
    print(chi_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    print('\nT-TESTS: numeric feature vs Loan_Status')
    print(ttest_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    print('\nANOVA TESTS: numeric feature across categorical groups')
    print(anova_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


if __name__ == '__main__':
    main()
