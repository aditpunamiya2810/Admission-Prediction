import numpy as np
import pickle
import pandas as pd
import streamlit as st 


pickle_in_lr = open(r"C:\Users\Adit Punamiya\OneDrive\Desktop\BTech AI\Sem 4\Machine Learning\Lab 4\lr_model.pkl","rb")
lr_model=pickle.load(pickle_in_lr)
pickle_in_scaler = open(r"C:\Users\Adit Punamiya\OneDrive\Desktop\BTech AI\Sem 4\Machine Learning\Lab 4\scalar.pkl","rb")
scaler=pickle.load(pickle_in_scaler)


def predict_admit(GRE_Score, Toefl_score,Univ_Rating,SOP, LOR, CGPA, Research):
    test_values = np.array([[GRE_Score, Toefl_score,Univ_Rating,SOP, LOR, CGPA, Research]])
    test_values_scaled=scaler.transform(test_values)
    prediction=lr_model.predict(test_values_scaled)
    #test_values = np.array([[337, 118,4,4.5,4.5,9.65,1]])]])
    #print(prediction)
    return prediction



def main():
    st.title("Admission Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Admission Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    GRE_Score = st.text_input("GRE Score (out of 340)","Type Here")
    Toefl_score = st.text_input("Toefl Score (out of 120)"," ")
    Univ_Rating = st.text_input("University Rating (out of 5)"," ")
    SOP = st.text_input("SOP (out of 5)"," ")
    LOR = st.text_input("LOR (out of 5)"," ")
    CGPA = st.text_input("CGPA (out of 10)"," ")
    Research = st.text_input("Research (1 for yes 0 for no)"," ")
    result=""
    if st.button("Predict"):
        result=predict_admit(GRE_Score, Toefl_score,Univ_Rating,SOP, LOR, CGPA, Research)
        result = np.round(result[0]*100,2)
    st.success('The admission chances are {}%'.format(result))
    if st.button("About"):
        st.text("My First Linear Regression model")
        st.text("Built with Streamlit")

if __name__=='__main__':
    # r = predict_admit(337, 118,4,4.5,4.5,9.65,1)
    main()

    