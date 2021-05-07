import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from functions import img_to_bytes
from models.AdaBoost import ada_param_selector
from models.DecisionTree_Classification import dt_param_selector
from models.DecisionTreeRegressor import dtr_param_selector
from models.GradientBoostingClassifier import gbc_param_selector
from models.GradientBoostingRegressor import gbr_param_selector
from models.LinearRegression import linearReg_param_selector
from models.LogisticRegression import logisticReg_param_selector
from models.randomForest_Classification import rf_param_selector
from models.RandomForestRegressor import rfr_param_selector
from models.SupportVectorRegressor import svr_param_selector
from models.utils import model_imports


def introduction():
    st.title("**Welcome to Models Playground**")
    st.subheader(
        """
        This is a place where you can train your machine learning models right from the browser
        """
    )
    st.markdown(
        """
    - 🗂️ Upload a **pre-processed** dataset
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
            problem_type = st.selectbox(
                "Type of Problem", ("Regression", "Classification")
            )
            result.append(problem_type)
            dependent_column = st.text_input("Enter the Dependent Variable")
            if dependent_column:
                result.append(dependent_column)
                y = dataset[dependent_column]
                X = dataset.drop(dependent_column, axis=1)
                result.append(dataset)
                result.append(X)
                result.append(y)
                return result


def split_data(result):
    split_container = st.sidebar.beta_expander("Data Splitting", True)
    with split_container:
        train_size_in_percent = st.number_input("Train Size in %", 0, 100, 80, 1)

        test_size = 1 - (float(train_size_in_percent) / 100)
        train_size = float(train_size_in_percent) / 100
        random_state = st.number_input("random_state", 0, 1000, 0, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            result[3], result[4], train_size=train_size, random_state=random_state
        )

        st.write("Shape of X train: ", X_train.shape)
        st.write("Shape of X test: ", X_test.shape)

        return X_train, X_test, y_train, y_test, test_size, random_state


def scale_data(result, X_train, X_test, y_train, y_test):
    scale_container = st.sidebar.beta_expander("Data Scaling", True)
    with scale_container:
        st.write("Select Scaling Method")
        standardScaler = st.checkbox("StandardScaler")
        minmaxscaler = st.checkbox("MinMaxScaler")
        columns = st.text_input(
            "Enter the columns to be scaled/normalized separated by comma"
        )
        columnss = []
        col_name = ""
        t = 0

        for col in columns:
            if col != "," and col != " ":
                col_name += col
            if col == "," or t == len(columns) - 1:
                columnss.append(col_name)
                col_name = ""
            t += 1
        # dataset = result[2]
        # st.write(X_train[columnss])
        # st.write(columnss)

        if standardScaler:
            for col in columnss:
                scaler = StandardScaler()
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
            st.write(X_test)

        if minmaxscaler:
            for col in columnss:
                scaler = MinMaxScaler()
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

            st.write(X_test)


introduction()
result = dataset_upload()
if result:
    X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
    scale_data(result, X_train, X_test, y_train, y_test)
