from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

from plotly.subplots import make_subplots
import plotly.graph_objs as go

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


def plot_metrics(metrics):

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
