from pydantic import BaseModel, Field

class LoanApplication(BaseModel):
    person_age: int = Field(..., gt=0, description="Applicant's age in years")
    person_income: int = Field(..., gt=0, description="Annual income")
    person_home_ownership: str = Field(..., description="Home ownership status (e.g., RENT, OWN, MORTGAGE)")
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_intent: str = Field(..., description="Intent of loan (e.g., PERSONAL, EDUCATION, MEDICAL, VENTURE)")
    loan_grade: str = Field(..., description="Loan grade (e.g., A, B, C, D, E, F, G)")
    loan_amnt: int = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., description="Loan interest rate")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Percentage of income")
    cb_person_default_on_file: str = Field(..., description="Historical default (Y/N)")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")

class PredictionResponse(BaseModel):
    prediction: str
    approval_probability: float
    risk_level: str
    explainability: dict = None
