import streamlit as st
import numpy as np
import pandas as pd
import time
from models.DecisionTree_Classification import dt_param_selector
from models.LinearRegression import linearReg_param_selector
from models.LogisticRegression import logisticReg_param_selector
from models.randomForest_Classification import rf_param_selector
from models.GradientBoostingClassifier import gbc_param_selector
from models.AdaBoost import ada_param_selector
from sklearn.model_selection import train_test_split


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
        if dataset is not None:
            result = []
            X = y = 0
            dataset = pd.read_csv(dataset)
            dependent_column = st.text_input("Enter the Dependent Variable")
            if dependent_column:
                y = dataset[dependent_column]
                X = dataset.drop(dependent_column, axis=1)
                result.append(X)
                result.append(y)
                return result


def split_data(result):
    split_container = st.sidebar.beta_expander("Data Splitting")
    with split_container:
        train_size_in_percent = st.number_input("Train Size in %", 0, 100, 80, 1)

        test_size = 1 - (float(train_size_in_percent) / 100)
        train_size = float(train_size_in_percent) / 100
        random_state = st.number_input("random_state", 0, 1000, 0, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            result[0], result[1], train_size=train_size, random_state=random_state
        )

        st.write("Shape of X train: ", X_train.shape)
        st.write("Shape of X test: ", X_test.shape)

        return X_train, X_test, y_train, y_test


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


def main():
    introduction()
    result = dataset_upload()
    if result is not None:
        X_train, X_test, y_train, y_test = split_data(result)
    model_selector()


if __name__ == "__main__":
    main()
