import streamlit as st
import numpy as np
import pandas as pd
import time
from models.utils import model_imports
from models.DecisionTree_Classification import dt_param_selector
from models.LinearRegression import linearReg_param_selector
from models.LogisticRegression import logisticReg_param_selector
from models.randomForest_Classification import rf_param_selector
from models.GradientBoostingClassifier import gbc_param_selector
from models.AdaBoost import ada_param_selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from plotly.subplots import make_subplots
import plotly.graph_objs as go


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


def model_selector(problem_type, X_train, y_train):
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    model = ""
    model_type = ""
    with model_training_container:
        # problem_type = st.selectbox("Type of Problem", ("Regression", "Classification"))

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
                model, duration = logisticReg_param_selector(X_train, y_train)

            elif model_type == "Decision Tree Classifier":
                model, duration = dt_param_selector(X_train, y_train)

            elif model_type == "Random Forest Classifier":
                model, duration = rf_param_selector(X_train, y_train)

            elif model_type == "Gradient Boosting Classifier":
                model, duration = gbc_param_selector(X_train, y_train)

            elif model_type == "AdaBoost Classifier":
                model, duration = ada_param_selector(X_train, y_train)

    return model_type, model, duration


def generate_snippet(
    model, model_type, dataset, test_size, random_state, dependent_column
):

    model_text_rep = repr(model)
    model_import = model_imports[model_type]
    dataset_import = "df = pd.read_csv('file_path')"
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


def evaluate_model(model, X_train, y_train, X_test, y_test, duration):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)
    # st.write(f"Train F1 is {train_f1}")
    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)
    # st.write(f"Test F1 is {test_f1}")
    return (
        model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
    )  # duration


# def plot_metrics(model, train_accuracy, test_accuracy, train_f1, test_f1):
#     fig = make_subplots(
#         rows=2,
#         cols=1,
#         specs=[[{"colspan": 1}, None], [{"type": "indicator"}, {"type": "indicator"}]],
#         # row_heights=[0.7, 0.30],
#     )

#     fig.add_trace(
#         go.indicator(
#             mode="gauge+number+delta",
#             value=test_accuracy,
#             title={"text": f"Accuracy (test)"},
#             domain={"x": [0, 1], "y": [0, 1]},
#             gauge={"axis": {"range": [0, 1]}},
#             delta={"reference": train_accuracy},
#         ),
#         row=1,
#         col=1,
#     )

#     fig.add_trace(
#         go.indicator(
#             mode="gauge+number+delta",
#             value=test_f1,
#             title={"text": f"F1 score (test)"},
#             domain={"x": [0, 1], "y": [0, 1]},
#             gauge={"axis": {"range": [0, 1]}},
#             delta={"reference": train_f1},
#         ),
#         row=2,
#         col=1,
#     )
#     return fig


# def main():
#     introduction()
#     result = dataset_upload()
#     if result is not None:
#         dependent_column = result[0]
#         X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
#         model_type, model = model_selector()
#         st.write(model)
#         if model:
#             (
#                 model,
#                 train_accuracy,
#                 train_f1,
#                 test_accuracy,
#                 test_f1,
#                 duration,
#             ) = train_model(model, X_train, y_train, X_test, y_test)
#             plot_metrics(model, train_accuracy, test_accuracy, train_f1, test_f1)
#             snippet = generate_snippet(
#                 model, model_type, result[0], test_size, random_state, dependent_column
#             )
#             st.write(snippet)


# if __name__ == "__main__":
#     main()
