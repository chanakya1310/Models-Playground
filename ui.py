import streamlit as st
import numpy as np
import pandas as pd

from models.DecisionTree_Classification import dt_param_selector
from models.LinearRegression import linearReg_param_selector
from models.LogisticRegression import logisticReg_param_selector
from models.randomForest_Classification import rf_param_selector
from models.GradientBoostingClassifier import gbc_param_selector
from models.AdaBoost import ada_param_selector


def introduction():
    st.title("**Welcome to Models Playground**")
    st.subheader(
        """
        This is a place where you can train your machine learning models right from the browser
        """
    )
    st.markdown(
        """
    - 🗂️ Upload a dataset
    - ⚙️ Pick a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics on train and test data
    - 🩺 Diagnose possible overitting and experiment with other settings
    -----
    """
    )


def dataset_upload():
    dataset_container = st.sidebar.beta_expander("Upload a Dataset", True)
    with dataset_container:
        dataset = st.file_uploader("Upload Dataset", type=["csv"])
        dataset_details = {}
        if dataset is not None:
            dataset_details = {
                "Dataset Name": dataset.name,
                "Dataset": dataset.type,
                "FileSize": dataset.size,
            }

    return dataset, dataset_details


def dataset_columns():
    dataset, dataset_details = dataset_upload()
    columns_container = st.sidebar.beta_expander("Dataset Columns", False)
    if dataset:
        dataset = pd.read_csv(dataset)
        columns_checked = []
        with columns_container:
            for i in dataset.columns:
                st.checkbox(i)


def model_selector():
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    model = ""
    model_type = ""
    with model_training_container:
        problem_type = st.selectbox("Type of Problem", ("Regression", "Classification"))

        if problem_type == "Regression":

            model_type = st.selectbox(
                "Choose a model",
                (
                    "Linear Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor",
                    "Support Vector Regression",
                ),
            )

            if model_type == "Linear Regression":
                model = linearReg_param_selector()

        elif problem_type == "Classification":

            model_type = st.selectbox(
                "Choose a model",
                (
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                    "AdaBoost Classifier",
                ),
            )

            if model_type == "Logistic Regression":
                model = logisticReg_param_selector()

            elif model_type == "Decision Tree Classifier":
                model = dt_param_selector()

            elif model_type == "Random Forest Classifier":
                model = rf_param_selector()

            elif model_type == "Gradient Boosting Classifier":
                model = gbc_param_selector()

            elif model_type == "AdaBoost Classifier":
                model = ada_param_selector()

    return model_type, model

    # st.write(dataset)
    # st.write(dataset.columns)


introduction()
dataset_columns()
model_selector()
