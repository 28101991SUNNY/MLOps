import pickle
import flask

from flask import Flask, request


app = Flask(__name__)   # create an instance of this Flask class.




@app.route("/ping", methods=["GET"])
              # using the route() decorator to tell Flask what URL should strigger our function
def ping():
    return {"Messege":"Running Model successful!"}








# loading the model
model_pickle = open('DTC_Pipe_line_model.pkl', 'rb')
clf = pickle.load(model_pickle)






@app.route("/predict", methods=['POST'])
def prediction():


    loan_req = request.get_json()
    

    encode_dict = {
        "term" : {' 36 months': 0, ' 60 months': 1},
        "grade" : {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
        "sub_grade": {'A1': 0,  'A2': 1,  'A3': 2,  'A4': 3,  'A5': 4,  'B1': 5,  'B2': 6,  'B3': 7,  'B4': 8,                            'B5': 9,  'C1': 10,  'C2': 11,  'C3': 12,  'C4': 13,  'C5': 14,  'D1': 15,  'D2': 16,  'D3': 17,                            'D4': 18,  'D5': 19,  'E1': 20,  'E2': 21,  'E3': 22,  'E4': 23,  'E5': 24,  'F1': 25,  'F2': 26,                                'F3': 27,  'F4': 28,  'F5': 29,  'G1': 30,  'G2': 31,  'G3': 32,  'G4': 33,  'G5': 34},
        "home_ownership":{'MORTGAGE': 0.1697808373138522,  'OTHER': 0.22033898305084745,  'OWN': 0.204155258495732,  'RENT': 0.2311396828241197,  'All': 0.1974347064918361},
        "purpose":{'car': 0.1402081977878985,  'credit_card': 0.16630877656645743,  'debt_consolidation': 0.20816406441192306,  'educational': 0.0,                        'home_improvement': 0.17161480877934512,  'house': 0.21132516053706946,  'major_purchase': 0.17779173865242656,                        'medical': 0.22043840691571473,  'moving': 0.2465564738292011,  'other': 0.21785846609855503,                            'renewable_energy': 0.23555555555555555,  'small_business': 0.3146894318516338,  'vacation': 0.18967280163599182,                            'wedding': 0.14316469321851452,  'All': 0.1974347064918361},
        "application_type":{'DIRECT_PAY': 0.32653061224489793,  'INDIVIDUAL': 0.1974310492914062,      'JOINT': 0.09967845659163987,'All': 0.1974347064918361},
        "Zip_Code":{'00813': 0.0,  '05113': 0.0,  '11650': 1.0,  '22690': 0.19461165454393445,  '29597': 0.0,                        '30723': 0.1963303898700433,  '48052': 0.2014581006763733,  '70466': 0.1969254960379037,  '86630': 1.0,                            '93700': 1.0,  'All': 0.1974347064918361}
    }


    term = encode_dict['term'][loan_req["term"]]
    grade = encode_dict['grade'][loan_req["grade"]]
    sub_grade = encode_dict['sub_grade'][loan_req["sub_grade"]]
    home_ownership = encode_dict['home_ownership'][loan_req["home_ownership"]]
    purpose = encode_dict['purpose'][loan_req["purpose"]]
    application_type = encode_dict['application_type'][loan_req["application_type"]]
    Zip_Code = encode_dict["Zip_Code"][loan_req["Zip_Code"]]


    loan_amnt = loan_req["loan_amnt"]
    int_rate = loan_req["int_rate"]
    installment =loan_req["installment"]
    annual_inc = loan_req["annual_inc"]
    dti = loan_req["dti"]
    open_acc = loan_req["open_acc"]
    pub_rec = loan_req["pub_rec"]
    revol_bal = loan_req["revol_bal"]
    revol_util = loan_req["revol_util"]
    total_acc = loan_req["total_acc"]
    mort_acc = loan_req["mort_acc"]
    pub_rec_bankruptcies = loan_req["pub_rec_bankruptcies"]

    input_features = [[loan_amnt, term, int_rate, installment,
                   grade, sub_grade, home_ownership, annual_inc, 
                   purpose, dti, open_acc, pub_rec, revol_bal, revol_util,
                   total_acc, application_type, mort_acc,
                     pub_rec_bankruptcies, Zip_Code]]
    
    
    
    prediction = clf.predict(input_features)

    if prediction == 0:
        pred = "Extend Credit Line"
    else:
        pred = "Default"

    return {"Status":pred}



@app.route("/get_params", methods=['GET'])
def get_application_params():

    parameters = {
                    "loan_amnt":5000,
                    "term":"",
                    "int_rate":5, 
                    "installment":65,
                    "grade":"A", 
                    "sub_grade":"A1", 
                    "home_ownership":"RENT", 
                    "annual_inc":117700, 
                    "purpose":"vacation", 
                    "dti":26, 
                    "open_acc":16, 
                    "pub_rec":0, 
                    "revol_bal":3669, 
                    "revol_util":41,
                    "total_acc":25, 
                    "application_type":"INDIVIDUAL", 
                    "mort_acc":0,
                    "pub_rec_bankruptcies":0, 
                    "Zip_Code" :"22690"
        }
    return parameters


