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

description = """
<h5>The following analysis and modelling was conducted on top of a Dataset from one of the UCI Machine Learning repositories.</h5>

##### Description from the original source <a href="https://archive.ics.uci.edu/dataset/73/mushroom">UCI Machine Learning Repository</a>:
<h5>"This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible or definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy."</h5>
<h5>In general, the terms used in the Dataset are very straightforward, with no in-depth knowledge required in order to understand them individually as they are directly related to physical characteristics of the mushrooms. I have previously changed the naming of the categories under each feature for better understanding.</h5>
<h5>The main goal of this project is to perform an in-depth analysis of the problem in order to facilitate the development of a model that can most optimally predict whether a mushroom is poisonous or edible, based on the several features presented.</h5>
"""

# Custom CSS for styling
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
}

.description {
    font-size: 1em;  /* Adjust size as needed */
    margin-top: 100px;
    max-width: 800px; /* Set max width to limit text width */
    line-height: 1.6; /* Improve readability */
    text-align: justify; /* Justify text */
}
</style>
"""

# Display the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Display the title with the icon, subtitle, and description
title_html = f"""
<div class="title-container">
    <div class="title-row">
        <img src="{icon_url}" alt="Icon">
        <h1>{title}</h1>
    </div>
    <div class="subtitle">{subtitle}</div>
    <div class="description">{description}</div>
</div>
"""

st.markdown(title_html, unsafe_allow_html=True)

st.sidebar.title("Filters")
st.sidebar.markdown("Here you can customize some filters for the dataset")



bucket_name = 'dataset-content-pedrohang'
object_key = 'mushrooms/mushrooms.csv'

s3_uri = f's3://{bucket_name}/{object_key}'
df = pd.read_csv(s3_uri, storage_options={'anon': True})

st.markdown("---")

st.title("Original Dataframe")

st.dataframe(df, use_container_width=True)