import pandas as pd
import streamlit as st
import datetime
import pickle
import warnings




st.write("""
    # Credit Line Extension Classification App:
""")



loan_amnt = st.slider("Loan Amount",
                      min_value=1000, max_value=50000)
int_rate = st.slider("Interest Rate",
                      min_value=2, max_value=30)
installment = st.slider("installment",
                      min_value=12, max_value=2000)
annual_inc =  st.slider("Anual Income",
                      min_value=10000, max_value=300000)
dti =   st.slider("debt-to-income ratio",
                      min_value=0, max_value=42)
open_acc =   st.slider("open_acc",
                      min_value=1, max_value=30)
pub_rec =    st.number_input("Public Record",
                      min_value=0, max_value=1)
revol_bal =  st.number_input("Revolving Balance",
                      min_value=0, max_value=80000)
revol_util = st.number_input("Revolving Util",
                      min_value=0, max_value=150)
total_acc = st.number_input("Total Accounts",
                      min_value=1, max_value=80)
mort_acc =  st.number_input("Mortgage Accounts",
                      min_value=0, max_value=1)
pub_rec_bankruptcies =  st.number_input("Public Record Bankruptcies",
                      min_value=0, max_value=1)




term = st.selectbox("Select term",
                            [' 36 months', ' 60 months'])


grade = st.selectbox("Select Grade",
                            ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

sub_grade = st.selectbox("Select Sub Grade",
                            ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2',
                              'C3', 'C4', 'C5', 'D1', 'D2', 'D3',
                              'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2',
                                'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'])

home_ownership = st.selectbox("Select Home Ownership",
                            ['MORTGAGE', 'OTHER', 'OWN', 'RENT', 'All'])

purpose = st.selectbox("Select Purpose",
                            ['car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
                              'house', 'major_purchase', 'medical', 'moving', 'other',
                              'renewable_energy', 'small_business', 'vacation', 'wedding', 'All'])

application_type = st.selectbox("Select Application Type",
                            ['DIRECT_PAY', 'INDIVIDUAL', 'JOINT', 'All'])



Zip_Code = st.selectbox("Select Zip Code",
                            ['00813', '05113', '11650', '22690', '29597', '30723', '48052', '70466', '86630', '93700', 'All'])









encode_dict = {


    "term" : {' 36 months': 0, ' 60 months': 1},
    "grade" : {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
    "sub_grade": {'A1': 0,  'A2': 1,  'A3': 2,  'A4': 3,  'A5': 4,  'B1': 5,  'B2': 6,  'B3': 7,  'B4': 8,
                        'B5': 9,  'C1': 10,  'C2': 11,  'C3': 12,  'C4': 13,  'C5': 14,  'D1': 15,  'D2': 16,  'D3': 17,
                          'D4': 18,  'D5': 19,  'E1': 20,  'E2': 21,  'E3': 22,  'E4': 23,  'E5': 24,  'F1': 25,  'F2': 26,
                            'F3': 27,  'F4': 28,  'F5': 29,  'G1': 30,  'G2': 31,  'G3': 32,  'G4': 33,  'G5': 34},
    "home_ownership":{'MORTGAGE': 0.1697808373138522,  'OTHER': 0.22033898305084745,  'OWN': 0.204155258495732,  'RENT': 0.2311396828241197,  'All': 0.1974347064918361},
    "purpose":{'car': 0.1402081977878985,  'credit_card': 0.16630877656645743,  'debt_consolidation': 0.20816406441192306,  'educational': 0.0,
                     'home_improvement': 0.17161480877934512,  'house': 0.21132516053706946,  'major_purchase': 0.17779173865242656,
                       'medical': 0.22043840691571473,  'moving': 0.2465564738292011,  'other': 0.21785846609855503,
                         'renewable_energy': 0.23555555555555555,  'small_business': 0.3146894318516338,  'vacation': 0.18967280163599182,
                           'wedding': 0.14316469321851452,  'All': 0.1974347064918361},
    "application_type":{'DIRECT_PAY': 0.32653061224489793,  'INDIVIDUAL': 0.1974310492914062, 
                             'JOINT': 0.09967845659163987,'All': 0.1974347064918361},
    
    "Zip_Code":{'00813': 0.0,  '05113': 0.0,  '11650': 1.0,  '22690': 0.19461165454393445,  '29597': 0.0,
                      '30723': 0.1963303898700433,  '48052': 0.2014581006763733,  '70466': 0.1969254960379037,  '86630': 1.0,
                        '93700': 1.0,  'All': 0.1974347064918361}

}










# making a function to predict 

def model_predict(loan_amnt,
                   term, 
                  int_rate, 
                  installment,
                   grade ,
                   sub_grade,
                   home_ownership,
                   annual_inc, 
                   purpose,
                   dti , 
                   open_acc, 
                   pub_rec, 
                   revol_bal, 
                   revol_util ,
                   total_acc , 
                   application_type , 
                   mort_acc,
                   pub_rec_bankruptcies, 
                   Zip_Code):
    
    with open("DTC_Pipe_line_model.pkl", 'rb') as file:

        DecisionTreeModel = pickle.load(file)
    
    input_features = [[loan_amnt, term, int_rate, installment,
                   grade, sub_grade, home_ownership, annual_inc, 
                   purpose, dti, open_acc, pub_rec, revol_bal, revol_util,
                   total_acc, application_type, mort_acc,
                     pub_rec_bankruptcies, Zip_Code]]
    
    return DecisionTreeModel.predict(input_features)









if st.button("Predict Loan Defulter"):

    term = encode_dict['term'][term]
    grade = encode_dict['grade'][grade]
    sub_grade = encode_dict['sub_grade'][sub_grade]
    home_ownership = encode_dict['home_ownership'][home_ownership]
    purpose = encode_dict['purpose'][purpose]
    application_type = encode_dict['application_type'][application_type]
    Zip_Code = encode_dict["Zip_Code"][Zip_Code]
    

    deafault = model_predict(loan_amnt,
                             term,
                             int_rate,
                             installment,
                             grade,
                             sub_grade,
                             home_ownership,
                             annual_inc,
                             purpose,
                             dti,open_acc,
                             pub_rec,
                             revol_bal,
                             revol_util,
                             total_acc,
                             application_type,
                             mort_acc,
                             pub_rec_bankruptcies,
                             Zip_Code)
    
    if deafault == 1:
        ans = "Default"
    else:
        ans = "Extend CreditLine"

    st.text("Predicted : "+ str(ans))

