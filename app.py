import time

import streamlit as st
from datetime import datetime

from functions import get_model_tips, get_model_url, local_css, plot_metrics
from ui import (
    dataset_upload,
    evaluate_model,
    footer,
    generate_snippet,
    introduction,
    model_selector,
    split_data,
    scale_data,
)

st.set_page_config(
    page_title="Models Playground", layout="wide", page_icon="./images/mlground.png"
)

hyperparameters = []


def sidebar_controllers(result):
    if result is not None:
        view = st.checkbox("View the Dataset")
        if view:
            st.write(result[2])
        dependent_column = result[1]
        X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
        X_train, X_test = scale_data(result, X_train, X_test)
        model_type, model, duration, problem_type = model_selector(
            result[0], X_train, y_train
        )
        if model:
            (model, train_accuracy, train_f1, test_accuracy, test_f1,) = evaluate_model(
                model, X_train, y_train, X_test, y_test, duration, problem_type
            )

            snippet = generate_snippet(
                model,
                model_type,
                result[0],
                test_size,
                random_state,
                dependent_column,
                problem_type,
                result[-1],
            )
            footer()
            # st.write(snippet)
            return (
                model_type,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                duration,
                train_accuracy,
                train_f1,
                test_accuracy,
                test_f1,
                snippet,
                problem_type,
            )


def body(
    x_train,
    x_test,
    y_train,
    y_test,
    model,
    model_type,
    duration,
    train_accuracy,
    train_f1,
    test_accuracy,
    test_f1,
    snippet,
    problem_type,
    name,
):
    local_css("css/style.css")
    col1, col2 = st.beta_columns((2, 1))
    with col1:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    with col2:
        relative_metrics = st.empty()
        add_placeholder = st.empty()
        show_placeholder = st.empty()
        plot_placeholder = st.empty()
        models_placeholder = st.empty()

    model_url = get_model_url(model_type)
    if problem_type == "Classification":
        metrics = {
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
        }

    elif problem_type == "Regression":
        metrics = {
            "train_mse": train_accuracy,
            "train_rmse": train_f1,
            "test_mse": test_accuracy,
            "test_rmse": test_f1,
        }

    model_tips = get_model_tips(model_type)

    fig = plot_metrics(metrics, problem_type)

    relative_metrics.warning(
        f"Increase or Decrease is with respect to Training Dataset"
    )
    plot_placeholder.plotly_chart(fig, True)
    if add_placeholder.button("Click to record these Hyperparmaters"):
        t0 = datetime.now()
        with open("data.txt", "a") as f:
            f.write("\n\n")
            f.write("Trained at: " + str(t0))
            f.write("\n")
            f.write("Dataset Name: " + str(name))
            f.write("\n")
            model = str(model).strip()
            f.write(model)
            for i in metrics:
                f.write("\n")
                f.write(str(i))
                f.write(" ")
                f.write(str(metrics[i]))
                f.write("")

    if show_placeholder.button("Click to view all models"):
        f = open("data.txt", "r")
        final = ""
        for x in f:
            final = "\t" + final + str(x) + "\n"
        final = final + "\n"
        models_placeholder.code(final)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    code_header_placeholder.header("**Retrain the same model in Python**")
    snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


if __name__ == "__main__":
    introduction()
    result = dataset_upload()
    if result:
        (
            model_type,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            duration,
            train_accuracy,
            train_f1,
            test_accuracy,
            test_f1,
            snippet,
            problem_type,
        ) = sidebar_controllers(result)
        if train_f1:
            body(
                X_train,
                X_test,
                y_train,
                y_test,
                model,
                model_type,
                duration,
                train_accuracy,
                train_f1,
                test_accuracy,
                test_f1,
                snippet,
                problem_type,
                result[5],
            )
