import streamlit as st
import time
from ui import (
    introduction,
    dataset_upload,
    split_data,
    model_selector,
    evaluate_model,
    generate_snippet,
)
from functions import get_model_url, get_model_tips, plot_metrics, local_css

st.set_page_config(
    page_title="Playground",
    layout="wide",  # page_icon="./images/flask.png"
)


def sidebar_controllers(result):
    if result is not None:
        dependent_column = result[1]
        X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
        model_type, model, duration = model_selector(result[0], X_train, y_train)
        if model:
            (
                model,
                train_accuracy,
                train_f1,
                test_accuracy,
                test_f1,
                # duration,
            ) = evaluate_model(model, X_train, y_train, X_test, y_test, duration)
            # plot_metrics(model, train_accuracy, test_accuracy, train_f1, test_f1)
            snippet = generate_snippet(
                model, model_type, result[0], test_size, random_state, dependent_column
            )
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
):
    local_css("css/style.css")
    col1, col2 = st.beta_columns((2, 1))

    with col2:
        plot_placeholder = st.empty()

    with col1:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    model_url = get_model_url(model_type)

    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }

    model_tips = get_model_tips(model_type)

    fig = plot_metrics(metrics)

    plot_placeholder.plotly_chart(fig, True)
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
            )
