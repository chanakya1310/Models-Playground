import time

import streamlit as st
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def ada_param_selector(X_train, y_train):

    base_estimator = st.selectbox("base_estimator", ["Decision Tree Classifier"])
    base_estimator1 = DecisionTreeClassifier(max_depth=1)
    if base_estimator == "Decision Tree Classifier":

        criterion = st.selectbox("criterion for " + base_estimator, ["gini", "entropy"])
        max_depth = st.number_input("max_depth for " + base_estimator, 1, 50, 5, 1)
        min_samples_split = st.number_input(
            "min_samples_split for " + base_estimator, 1, 20, 2, 1
        )
        max_features = st.selectbox(
            "max_features for " + base_estimator, [None, "auto", "sqrt", "log2"]
        )

        paramsDT = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
        }

        base_estimator1 = DecisionTreeClassifier(**paramsDT)

    base_estimator = base_estimator1
    n_estimators = st.number_input("n_estimators", 1, 1000, 100, 10)
    learning_rate = st.number_input("learning_rate", 0.0, 10.0, 0.1, 0.1)
    random_state = st.number_input("random_state", 0, 1000, 0, 1, key="ada")

    params = {
        "base_estimator": base_estimator1,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": random_state,
    }

    model = AdaBoostClassifier(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return model, duration
