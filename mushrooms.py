import streamlit as st
import boto3
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices, dmatrix
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

st.set_page_config(
     page_title="Mushroom Classification",
     page_icon="https://cdn-icons-png.flaticon.com/512/3656/3656824.png",
     layout="wide",
)

title = "Mushroom Classification"
subtitle = "Project by Pedro Hang"
icon_url = "https://cdn-icons-png.flaticon.com/512/3656/3656824.png"  # Replace with your icon URL

custom_css = """
<style>
.title-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center; /* Center horizontally */
    margin-top: 50px; /* Adjust to move the container down */
    text-align: center; /* Center text */
}

.title-row {
    display: flex;
    align-items: center;
}

.title-row img {
    width: 150px;  /* Adjust size as needed */
    height: 150px; /* Adjust size as needed */
    margin-right: 20px;
}

.title-row h1 {
    font-size: 3em;  /* Adjust size as needed */
    margin: 0;
}

.subtitle {
    font-size: 1.5em;  /* Adjust size as needed */
    margin-top: 10px;
}
</style>
"""

# Display the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Display the title with the icon
title_html = f"""
<div class="title-container">
    <div class="title-row">
        <img src="{icon_url}" alt="Icon">
        <h1>{title}</h1>
    </div>
    <div class="subtitle">{subtitle}</div>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)


