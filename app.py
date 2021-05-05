import streamlit as st
import time
from ui import (
    introduction,
    dataset_upload,
    split_data,
    model_selector,
    train_model,
    generate_snippet,
)


def sidebar_controllers():
    introduction()
    result = dataset_upload()
    if result is not None:
        dependent_column = result[0]
        time.sleep(1)
        X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
        model_type, model = model_selector(result[0])
        st.write(model)
        if model:
            (
                model,
                train_accuracy,
                train_f1,
                test_accuracy,
                test_f1,
                duration,
            ) = train_model(model, X_train, y_train, X_test, y_test)
            # plot_metrics(model, train_accuracy, test_accuracy, train_f1, test_f1)
            snippet = generate_snippet(
                model, model_type, result[0], test_size, random_state, dependent_column
            )
            st.write(snippet)
            return (model_type, model, X_train, y_train, X_test, y_test)


sidebar_controllers()
