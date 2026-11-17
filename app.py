"""Streamlit entry point for the current ECG project dashboard."""
from pathlib import Path

import streamlit as st

from project_overview import render

st.set_page_config(page_title="HeartPrint | ECG", layout="wide")

render(Path(__file__).resolve().parent)
