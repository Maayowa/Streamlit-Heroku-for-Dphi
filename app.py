import streamlit as st
import json
import pickle
import numpy as np

__data_columns = None
__model = None

def load_saved_artifacts():
    print('Loading saved artifact...start')
    global __data_columns
    global __model
    global __scaler

    jsonfile = r"features.json"
    modelfile = r"model.pkl"
    scalefile = r"scaler.pkl"

    filename =  open(jsonfile, 'r')
    modelfile = open(modelfile, 'rb')
    scaler = open(scalefile, 'rb')

    __data_columns = json.load(filename)['column_names']
    __model = pickle.load(modelfile)
    __scaler = pickle.load(scaler)

    print("Loading saved artifacts...done")


def loan_app_prediction(gender, married, dependents,
                        education, self_employed, app_income,
                        coapp_income, loan_amount, loan_amount_term,
                        credit_history,prop_area):

    features = [gender, married, dependents,
                education, self_employed, app_income,
                coapp_income, loan_amount, loan_amount_term,
                credit_history,prop_area]

    instance = np.zeros(len(__data_columns))

    features = __scaler.transform([features])

    instance[:] = features

    pred = __model.predict([instance])

    if pred:
        result =  "Approve Loan Application"
    else:
        result = "Reject Loan Application"

    return result




def main():
    st.title("Loan Status Prediction")

    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "font-family: "Roboto", sans-serif;
  font-size: 21px;
  margin-bottom: 8px;
  text-align: center"> Streamlit Loan Status Prediction App </h2>
  </div>
  """

    st.markdown(html_temp, unsafe_allow_html = True)

    gnd = ('Male', 'Female')
    gndval = list(range(len(gnd)))
    gender = st.selectbox("Gender", gndval, format_func = lambda x: gnd[x])
    marrd = ("Not married","Married")
    marrdval = list(range(len(marrd)))
    married = st.selectbox("Married", marrdval, format_func = lambda x: marrd[x])
    depndt = ("0", "1","2", "3+")
    depndtval = list(range(len(depndt)))
    dependents = st.selectbox("Number of Dependents", depndtval, format_func = lambda x: depndt[x])
    edctn = ('Graduate', "Not Graduate")
    edctnval = list(range(len(edctn)))
    education = st.selectbox("Level of Education", edctnval, format_func = lambda x: edctn[x])
    slfmp = ("Self-Employed", "Not Self-Employed")
    slfmpval = list(range(len(slfmp)))
    selfemp =  st.selectbox("Is Applicant Self-Employed", slfmpval, format_func = lambda x : slfmp[x])
    cdht = ("Outstanding Debts","Cleared debts")
    cdhtval = list(range(len(cdht)))
    credhist = st.selectbox("Customer's Credit History", cdhtval, format_func = lambda x: cdht[x])
    prpar = ("Rural", "SemiUrban", "Urban")
    prparval = list(range(len(prpar)))
    prop_area = st.selectbox("Property Area", prparval, format_func = lambda x: prpar[x])
    appincome = st.text_input("Applicant's Income", "Enter a value")
    coappincome = st.text_input("Coapplicant's Income", "Enter a value")
    loan_amount = st.text_input("Loan Amount", "Enter a value")
    loan_amount_term = st.text_input("Loan Term", "Enter a value in days")


    result = ""
    if st.button("Check Status"):
        result = loan_app_prediction(gender, married, dependents,
                                    education, selfemp, appincome,
                                    coappincome, loan_amount, loan_amount_term,
                                    credhist,prop_area)
    st.success("Loan Application Status: \t{}".format(result))
    if st.button("About"):
        st.text("Learn with me! Iwakin Oluwabunmi, 2020.")
        st.text("Built with Streamlit")

if __name__ == "__main__":
    load_saved_artifacts()
    main()
