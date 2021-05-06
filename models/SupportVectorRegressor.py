import time

import streamlit as st
from sklearn.svm import SVR


def svr_param_selector(X_train, y_train):

    kernel = st.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"])
    gamma = st.selectbox("gamma", ["scale", "auto"])
    C = st.number_input("C", 1.0, 1000.0, 1.0, 1.0)
    max_iter = st.number_input("max_iter", -1, 1000, -1, 1)
    epsilon = st.number_input("epsilon", 0.0, 100.0, 0.1, 0.1)
    params = {
        "kernel": kernel,
        "gamma": gamma,
        "C": C,
        "max_iter": max_iter,
        "epsilon": epsilon,
    }

    model = SVR(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
