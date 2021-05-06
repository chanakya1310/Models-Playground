from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import time


def dtr_param_selector(X_train, y_train):

    criterion = st.selectbox("criterion", ["mse", "friedman_mse", "mae", "poisson"])
    max_depth = st.number_input("max depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeRegressor(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
