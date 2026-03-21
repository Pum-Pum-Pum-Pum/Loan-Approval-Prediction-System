import pandas as pd
import numpy as np
from pathlib import Path


NUMERIC_COLS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
]

CATEGORICAL_COLS = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
]

PSI_THRESHOLD_MODERATE = 0.1
PSI_THRESHOLD_HIGH = 0.25


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.linspace(0, 100, buckets + 1)
    bins = np.percentile(expected, breakpoints)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_counts / max(1, expected_counts.sum())
    actual_perc = actual_counts / max(1, actual_counts.sum())

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi >= PSI_THRESHOLD_HIGH:
        return 'high_drift'
    if psi >= PSI_THRESHOLD_MODERATE:
        return 'moderate_drift'
    return 'stable'


def create_shifted_inference_sample(train_df: pd.DataFrame, sample_size: int = 200, random_state: int = 42) -> pd.DataFrame:
    sample = train_df.sample(n=sample_size, random_state=random_state).copy()

    sample['ApplicantIncome'] = sample['ApplicantIncome'] * 1.35
    sample['LoanAmount'] = sample['LoanAmount'].fillna(sample['LoanAmount'].median()) * 1.20
    sample['Credit_History'] = sample['Credit_History'].fillna(1)
    sample.loc[sample.sample(frac=0.20, random_state=random_state).index, 'Credit_History'] = 0
    sample.loc[sample.sample(frac=0.25, random_state=random_state + 1).index, 'Property_Area'] = 'Rural'

    return sample


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / 'data' / 'raw' / 'train.csv'

    train_df = pd.read_csv(train_path)
    current_df = create_shifted_inference_sample(train_df)

    print('DRIFT CHECKING REPORT')
    print('---------------------')
    print('Reference data : training data')
    print('Current data   : simulated shifted inference batch')

    psi_rows = []
    for col in NUMERIC_COLS:
        psi_value = compute_psi(train_df[col], current_df[col])
        psi_rows.append({
            'feature': col,
            'psi': psi_value,
            'status': classify_psi(psi_value),
        })

    psi_df = pd.DataFrame(psi_rows).sort_values(by='psi', ascending=False)

    print('\nNUMERIC DRIFT (PSI):')
    print(psi_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print('\nCATEGORICAL DISTRIBUTION SHIFTS:')
    for col in CATEGORICAL_COLS:
        ref_dist = train_df[col].fillna('Missing').value_counts(normalize=True)
        cur_dist = current_df[col].fillna('Missing').value_counts(normalize=True)
        dist_df = pd.concat([ref_dist, cur_dist], axis=1).fillna(0)
        dist_df.columns = ['reference_pct', 'current_pct']
        dist_df['abs_diff'] = (dist_df['current_pct'] - dist_df['reference_pct']).abs()
        print(f'\nFeature: {col}')
        print(dist_df.sort_values(by='abs_diff', ascending=False).to_string(float_format=lambda x: f'{x:.4f}'))

    high_drift_features = psi_df[psi_df['status'] == 'high_drift']['feature'].tolist()
    moderate_drift_features = psi_df[psi_df['status'] == 'moderate_drift']['feature'].tolist()

    print('\nDRIFT SUMMARY:')
    print(f'High drift features     : {high_drift_features}')
    print(f'Moderate drift features : {moderate_drift_features}')

    if high_drift_features:
        print('\nRecommended action: investigate upstream data changes, monitor model quality, and consider retraining if drift persists.')
    elif moderate_drift_features:
        print('\nRecommended action: continue monitoring closely and compare model performance on recent data.')
    else:
        print('\nRecommended action: no strong drift signal detected from current checks.')


if __name__ == '__main__':
    main()
