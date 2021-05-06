import time

import streamlit as st
from sklearn.ensemble import RandomForestRegressor


def rfr_param_selector(X_train, y_train):

    criterion = st.selectbox("criterion", ["mse", "mae"])
    n_estimators = st.number_input("n_estimators", 50, 1000, 100, 10)
    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
