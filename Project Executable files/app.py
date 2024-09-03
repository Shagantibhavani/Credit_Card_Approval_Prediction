from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model
with open("creditcardmodel10.pkl", 'rb') as handle:
 model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template("index1.html")

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        #Reading the inputs given by the user
        CODE_GENDER = int(request.form['CODE_GENDER'])
        FLAG_OWN_CAR = int(request.form['FLAG_OWN_CAR'])
        FLAG_OWN_REALTY = int(request.form['FLAG_OWN_REALTY'])
        AMT_INCOME_TOTAL = int(request.form['AMT_INCOME_TOTAL'])
        NAME_INCOME_TYPE = (request.form['NAME_INCOME_TYPE'])
        NAME_EDUCATION_TYPE = int(request.form['NAME_EDUCATION_TYPE'])
        NAME_FAMILY_STATUS = int(request.form['NAME_FAMILY_STATUS'])
        NAME_HOUSING_TYPE = int(request.form['NAME_HOUSING_TYPE'])
        DAYS_BIRTH = int(request.form['DAYS_BIRTH'])
        DAYS_EMPLOYED = int(request.form['DAYS_EMPLOYED'])
        CNT_FAM_MEMBERS = int(request.form['CNT_FAM_MEMBERS'])
        paid_off = int(request.form['paid_off'])
        no_of_pastdues = int(request.form['no_of_pastdues'])
        no_loan = int(request.form['no_loan'])

        # Create a dictionary with the input features
        data={
            "CODE_GENDER": [CODE_GENDER],
            "FLAG_OWN_CAR": [FLAG_OWN_CAR],
            "FLAG_OWN_REALTY": [FLAG_OWN_REALTY],
            "AMT_INCOME_TOTAL": [AMT_INCOME_TOTAL],
            "NAME_INCOME_TYPE": [NAME_INCOME_TYPE],
            "NAME_EDUCATION_TYPE": [NAME_EDUCATION_TYPE],
            "NAME_FAMILY_STATUS": [NAME_FAMILY_STATUS],
            "NAME_HOUSING_TYPE": [NAME_HOUSING_TYPE],
            "DAYS_BIRTH": [DAYS_BIRTH],
            "DAYS_EMPLOYED": [DAYS_EMPLOYED],
            "CNT_FAM_MEMBERS": [CNT_FAM_MEMBERS],
            "paid_off": [paid_off],
            "no_of_pastdues": [no_of_pastdues],
            "no_loan": [no_loan]
        }

        # Create DataFrame
        x = pd.DataFrame(data)

        # Ensure the order of columns matches the training data
        feature_order = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
            'CNT_FAM_MEMBERS', 'paid_off', 'no_of_pastdues', 'no_loan'
        ]
        x = x[feature_order]

        # Predictions using the loaded model file
        pred = model.predict(x) 
        if pred == 0:
          prediction = "Eligible " 
        else:
          prediction = " Not Eligible "

        # Showing the prediction results in a UI
        return render_template("Result.html", prediction=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
