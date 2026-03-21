from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'artifacts' / 'loan_approval_model.joblib'

app = FastAPI(title='Loan Approval Prediction API', version='1.0.0')


class LoanApplication(BaseModel):
    Gender: str = Field(..., examples=['Male'])
    Married: str = Field(..., examples=['Yes'])
    Dependents: str = Field(..., examples=['0'])
    Education: str = Field(..., examples=['Graduate'])
    Self_Employed: str = Field(..., examples=['No'])
    ApplicantIncome: float = Field(..., ge=0)
    CoapplicantIncome: float = Field(..., ge=0)
    LoanAmount: float = Field(..., gt=0)
    Loan_Amount_Term: float = Field(..., gt=0)
    Credit_History: float = Field(..., ge=0, le=1)
    Property_Area: str = Field(..., examples=['Urban'])


@app.on_event('startup')
def load_artifact() -> None:
    artifact = joblib.load(MODEL_PATH)
    app.state.model = artifact['model']
    app.state.threshold = artifact['threshold']


@app.get('/health')
def health_check():
    return {'status': 'ok', 'model_loaded': hasattr(app.state, 'model')}


@app.post('/predict')
def predict_loan(application: LoanApplication):
    payload = application.model_dump()
    payload['TotalIncome'] = payload['ApplicantIncome'] + payload['CoapplicantIncome']

    df = pd.DataFrame([payload])
    probability = app.state.model.predict_proba(df)[0, 1]
    prediction = int(probability >= app.state.threshold)

    return {
        'prediction': prediction,
        'prediction_label': 'Y' if prediction == 1 else 'N',
        'approval_probability': round(float(probability), 6),
        'threshold_used': app.state.threshold,
    }
