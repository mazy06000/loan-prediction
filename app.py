import json
import time
import streamlit as st
import os
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import webbrowser
import lightgbm as lgb
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(layout='centered')

def predict_fn(x):
  # Function necessary to get probabilities to be used by LIME
  # As described in here: https://github.com/marcotcr/lime/issues/51
  preds = st.session_state["model"].predict(x).reshape(-1, 1)
  p0 = 1 - preds
  return np.hstack((p0, preds))

if "model" not in st.session_state:
    model = lgb.Booster(model_file='lgbm_loan_prediction.txt')
    st.session_state["model"] = model

if "individus" not in st.session_state:
    st.session_state["individus"] = pd.read_csv('individus.csv')
    st.session_state["X_train"] = pd.read_csv('X_train.csv')

header = st.container()
parameters = st.container()
characteristic = st.container()

with header:
    st.markdown("""<div style="display:flex;justify-content:center;"><h1>LOAN PREDICTION</h1></div>""", unsafe_allow_html=True)
    

with parameters:

    st.header("Prediction")
    client = st.selectbox('Select a client', st.session_state["individus"].index)

    if st.button('Predict'):
        client_data = st.session_state["individus"].iloc[client]
        percent = round(st.session_state["model"].predict(client_data)[0]*100)

        

        if percent < 50:
            st.markdown(f"""<div style="display:flex;align-items: center;flex-direction: column;">
            <h3 style="color:green; text-align: center;">{percent}%</h3>
            <p>chance of having a credit default</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: green;
                    }
                </style>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"""<div style="display:flex;align-items: center;flex-direction: column;">
            <h3 style="color:red; text-align: center;">{percent}%</h3>
            <p>chance of having a credit default</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: red;
                    }
                </style>""",
                unsafe_allow_html=True,
            )

        my_bar = st.progress(0)
        for percent_complete in range(percent):
            time.sleep(0.001)
            my_bar.progress(percent_complete + 1)

        explainer_lime = LimeTabularExplainer(st.session_state["X_train"].to_numpy(), 
                        feature_names=st.session_state["X_train"].columns, 
                        discretize_continuous=False 
                        )
        lime = explainer_lime.explain_instance(client_data, predict_fn)
        
        components.html(lime.as_html(), height=600)

with characteristic:
    st.header("Characteristic")
    left, right = st.columns(2)
    left.subheader("About Model")
    left.markdown('<div><b>Model:</b> LightGBM </div>',
                  unsafe_allow_html=True)

    right.subheader("Model performance")
    right.markdown('<div><b>Test AUC:</b> 74%</div>', unsafe_allow_html=True)


# with credit:
#     st.markdown("""<div style="text-align: center; margin-top: 25px;"">
#     By <button onClick='window.location.href="https://www.mohamed-mazy.com"'>Mohamed Mazy</button></div>""", unsafe_allow_html=True)
