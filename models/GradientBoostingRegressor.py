import time

import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor


def gbr_param_selector(X_train, y_train):

    loss = st.selectbox("loss", ["ls", "lad", "huber", "quantile"])
    learning_rate = st.number_input("learning_rate", 0.0, 10.0, 0.1, 0.1)
    n_estimators = st.number_input("n_estimators", 1, 2000, 100, 10)
    criterion = st.selectbox("criterion", ["friedman_mse", "mse", "mae"])
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_depth = st.number_input("max_depth", 1, 20, 3, 1)
    random_state = st.number_input("random_state", 0, 1000, 0, 1, key="xgboost")
    max_features = st.selectbox("max_features", ["auto", "sqrt", "log2"])

    params = {
        "loss": loss,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "max_depth": max_depth,
        "random_state": random_state,
        "max_features": max_features,
    }

    model = GradientBoostingRegressor(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
