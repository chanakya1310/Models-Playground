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
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
            # name = dataset.name
            result = []
            X = y = 0
            dfname = dataset.name
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
                result.append(dfname)
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


def scale_data(result, X_train, X_test):
    scale_container = st.sidebar.beta_expander("Data Scaling", True)
    with scale_container:
        st.write("Select Scaling Method")
        standardScaler = st.checkbox("StandardScaler")
        minmaxscaler = st.checkbox("MinMaxScaler")
        none = st.checkbox("None")
        columns = st.text_input(
            "Enter the columns to be scaled/normalized separated by comma"
        )
        columnss = []
        col_name = ""
        t = 0

        if len(columns) == 0:
            return X_train, X_test
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

        if none:
            return X_train, X_test

        if standardScaler:
            for col in columnss:
                scaler = StandardScaler()
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
            st.write(X_test)
            return X_train, X_test

        if minmaxscaler:
            for col in columnss:
                scaler = MinMaxScaler()
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

            st.write(X_test)
            return X_train, X_test

        if minmaxscaler and standardScaler:
            st.write("Please select only one")


def model_selector(problem_type, X_train, y_train):
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    model = ""
    model_type = ""
    with model_training_container:

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
                model, duration = linearReg_param_selector(X_train, y_train)

            elif model_type == "Decision Tree Regressor":
                model, duration = dtr_param_selector(X_train, y_train)

            elif model_type == "Random Forest Regressor":
                model, duration = rfr_param_selector(X_train, y_train)

            elif model_type == "Gradient Boosting Regressor":
                model, duration = gbr_param_selector(X_train, y_train)

            elif model_type == "Support Vector Regression":
                model, duration = svr_param_selector(X_train, y_train)

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
                model, duration = logisticReg_param_selector(X_train, y_train)

            elif model_type == "Decision Tree Classifier":
                model, duration = dt_param_selector(X_train, y_train)

            elif model_type == "Random Forest Classifier":
                model, duration = rf_param_selector(X_train, y_train)

            elif model_type == "Gradient Boosting Classifier":
                model, duration = gbc_param_selector(X_train, y_train)

            elif model_type == "AdaBoost Classifier":
                model, duration = ada_param_selector(X_train, y_train)

    return model_type, model, duration, problem_type


def generate_snippet(
    model,
    model_type,
    dataset,
    test_size,
    random_state,
    dependent_column,
    problem_type,
    name,
):

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if problem_type == "Classification":
        dataset_import = f"""df = pd.read_csv(str({name}))"""
        snippet = f"""

        {model_import}

        import pandas as pd

        from sklearn.metrics import accuracy_score, f1_score 

        from sklearn.model_selection import train_test_split

        {dataset_import}

        dependent_column = str({dependent_column})

        y = df[dependent_column]

        X = df.drop(dependent_column, axis = 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = {round(test_size, 2)}, random_state = {random_state}
        )

        ## Type in the code for MinMaxScaler or StandardScaler

        model = {model_text_rep}

        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)

        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)

        test_f1 = f1_score(y_test, y_test_pred)
        """

        return snippet

    elif problem_type == "Regression":
        dataset_import = "df = pd.read_csv('file_path')"
        snippet = f"""

        {model_import}

        import pandas as pd

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        from sklearn.model_selection import train_test_split

        {dataset_import}

        dependent_column = str({dependent_column})

        y = df[dependent_column]

        X = df.drop(dependent_column, axis = 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = {round(test_size, 2)}, random_state = {random_state}
        )

        model = {model_text_rep}

        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)

        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

        test_mse = mean_squared_error(y_test, y_test_pred)

        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        """

        return snippet


def evaluate_model(model, X_train, y_train, X_test, y_test, duration, problem_type):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    if problem_type == "Classification":
        train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
        train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

        test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
        test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)
        return (
            model,
            train_accuracy,
            train_f1,
            test_accuracy,
            test_f1,
        )
    elif problem_type == "Regression":
        train_mse = np.round(mean_squared_error(y_train, y_train_pred), 3)
        train_rmse = np.round(
            mean_squared_error(y_train, y_train_pred, squared=False), 3
        )

        test_mse = np.round(mean_squared_error(y_test, y_test_pred), 3)
        test_rmse = np.round(mean_squared_error(y_test, y_test_pred, squared=False), 3)

        return (model, train_mse, train_rmse, test_mse, test_rmse)


def footer():
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=30 height=30>](https://github.com/chanakya1310/Machine-Learning-Models-Dashboard) <small> ML Playground 0.1.0 | May 2021</small>""".format(
            img_to_bytes("./images/github-mark.png")
        ),
        unsafe_allow_html=True,
    )
