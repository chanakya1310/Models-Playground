import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def rf_param_selector():

    criterion = st.selectbox("criterion", ["gini", "entropy"])
    n_estimators = st.number_input("n_estimators", 1, 1000, 100, 10)
    max_depth = st.number_input("max depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", ["auto", "sqrt", "log2"])
    random_state = st.number_input("random_state", 0, 1000, 0, 1)

    params = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "random_state": random_state,
    }

    model = RandomForestClassifier(**params)

    return model
