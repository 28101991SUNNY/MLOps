from typing import Union
import pickle

from fastapi import FastAPI

app = FastAPI()


#loading the ML model
model_pickle = open("./classifier.pkl", "rb")
clf = pickle.load(model_pickle)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def prediction(loan_req: dict):

    print(loan_req)

    if loan_req['Gender'] == "Male":
        gender = 0
    else:
        gender = 1
    
    if loan_req['Married'] == "Unmarried":
        marital_status = 0
    else:
        marital_status = 1
        
    if loan_req['Credit_History'] == "Uncleared Debts":
        credit_history = 0
    else:
        credit_history = 1
    
    applicant_income = loan_req['ApplicantIncome']
    loan_amt = loan_req['LoanAmount'] / 1000

    input_data = [[gender, marital_status, applicant_income, loan_amt, credit_history]]
    
    ##generate inference
    prediction = clf.predict(input_data)

    if prediction == 0:
        pred = "Rejected"
    else:
        pred = "Approved"

    return {"loan_approval_status": pred}
