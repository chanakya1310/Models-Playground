import streamlit as st
from sklearn.linear_model import LogisticRegression


def logisticReg_param_selector():

    solver = st.selectbox("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
    max_iter = st.number_input("max iterations", 10, 1000, 100, 50)
    multi_class = st.selectbox("multi-class", ["auto", "ovr", "multinomial"])

    params = {"solver": solver, "max_iter": max_iter, "multi_class": multi_class}
    model = LogisticRegression(**params)

    return model
