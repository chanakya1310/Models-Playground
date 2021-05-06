import base64
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from models.utils import model_infos, model_urls


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def plot_metrics(metrics, problem_type):

    if problem_type == "Classification":
        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "indicator"}], [{"type": "indicator"}]],
            row_heights=[0.7, 0.30],
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics["test_accuracy"],
                title={"text": f"Accuracy (test)"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 1]}},
                delta={"reference": metrics["train_accuracy"]},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics["test_f1"],
                title={"text": f"F1 score (test)"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 1]}},
                delta={"reference": metrics["train_f1"]},
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=700,
        )

        return fig

    elif problem_type == "Regression":
        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{"type": "indicator"}], [{"type": "indicator"}]],
            row_heights=[0.7, 0.30],
        )

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics["test_mse"],
                domain={"row": 1, "column": 1},
                delta={"reference": metrics["train_mse"], "increasing.color": "red"},
                title={"text": f"MSE (test)"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics["test_rmse"],
                domain={"row": 2, "column": 1},
                delta={"reference": metrics["train_rmse"], "increasing.color": "red"},
                title={"text": f"RMSE (test)"},
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=700,
        )

        return fig
