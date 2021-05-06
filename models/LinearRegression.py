import time

import streamlit as st
from sklearn.linear_model import LinearRegression


def linearReg_param_selector(X_train, y_train):

    model = LinearRegression()
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
