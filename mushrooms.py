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
import plotly.graph_objects as go


st.set_page_config(
     page_title="Mushroom Classification",
     page_icon="https://cdn-icons-png.flaticon.com/512/3656/3656824.png",
     layout="wide",
)

title = "Mushroom Classification"
subtitle = "Project by Pedro Hang"
icon_url = "https://cdn-icons-png.flaticon.com/512/3656/3656824.png"  # Replace with your icon URL
crispdm = "Business Understanding"

description = """
<h5>The following analysis and modelling was conducted on top of a Dataset from one of the UCI Machine Learning repositories.</h5>
<h5>Description from the original source <a href="https://archive.ics.uci.edu/dataset/73/mushroom" style="color:#FF855C;">UCI Machine Learning Repository</a>:</h5>
<h5>"This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the <a href="https://en.wikipedia.org/wiki/Agaricus" style="color:#FF855C;">Agaricus</a> and <a href="https://en.wikipedia.org/wiki/Lepiota" style="color:#FF855C;">Lepiota</a> Family (pp. 500-525). Each species is identified as definitely edible or definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy."</h5>
<h5>In general, the terms used in the Dataset are very straightforward, with no in-depth knowledge required in order to understand them individually as they are directly related to physical characteristics of the mushrooms. I have previously changed the naming of the categories under each feature for better understanding.</h5>
<h5>The main goal of this project is to perform an in-depth analysis of the problem in order to facilitate the development of a model that can most optimally predict whether a mushroom is poisonous or edible, based on the several features presented.</h5>
<h5>It is important to highlight that the project was developed following <a href="https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining" style="color:#FF855C;">crisp-DM</a> (Cross Industry Standard Process for Data Mining) methodology.</h5>
"""
st.markdown('<a name="top"></a>', unsafe_allow_html=True)
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
    margin-top: 40px;
    max-width: 800px; /* Set max width to limit text width */
    line-height: 1.6; /* Improve readability */
    text-align: justify; /* Justify text */
}

.crispdm{
    font-size: 2em;  /* Adjust size as needed */
    margin-top: 50px;
    max-width: 800px; /* Set max width to limit text width */
    text-align: left; /* Justify text */
    color: #FF855C;
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
    <div class="crispdm">{crispdm}</div>
    <div class="description">{description}</div>
</div>
"""

st.markdown(title_html, unsafe_allow_html=True)

image_path = "https://cdn-icons-png.flaticon.com/512/3656/3656824.png"  # Update with the correct path if necessary

# Sidebar setup with icon
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{image_path}" width="25" height="25" style="margin-right: 10px;">
    <h1 style="margin: 0;">Mushroom Classification</h1>
</div>
""", unsafe_allow_html=True) ####### TOPICS STRUCTURE
st.sidebar.markdown("<h3 style='margin-bottom: 10px;'>Topics:</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#top' style='color:#FF855C; text-decoration: none; margin-bottom: 5px;'> - Business Understanding</a><br>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#original-dataset' style='color:#FF855C; text-decoration: none; margin-bottom: 5px;'> - Original Dataset</a><br>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#modified-dataset' style='color:#FF855C; text-decoration: none; margin-bottom: 5px;'> - Modified Dataset</a><br>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#cramers-v-matrix' style='color:#FF855C; text-decoration: none; margin-bottom: 5px;'> - Cramér's V Matrix</a><br>", unsafe_allow_html=True)





bucket_name = 'dataset-content-pedrohang'
object_key = 'mushrooms/mushrooms.csv'

s3_uri = f's3://{bucket_name}/{object_key}'
df = pd.read_csv(s3_uri, storage_options={'anon': True})




st.markdown("---")

st.title("Original Dataset")

st.dataframe(df, use_container_width=True)

st.markdown("""
<h3 style="color:#FF855C;">Data Preparation</h3>
            """, unsafe_allow_html=True)

st.markdown("""
##### The original dataset presented above was preliminarily used to train a simple [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree), and since the accuracy for that model was always 100%, I have decided to eliminate several features in order to make the predictions more challenging.
""")

df = df.drop(columns=['odor', 'spore-print-color', 'ring-type', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'bruises', 'gill-size', 'stalk-color-above-ring', 'stalk-color-below-ring', 'population', 'veil-color', 'cap-surface', 'gill-spacing'])
most_frequent_value = df['stalk-root'].mode()[0]
df_mushrooms = df.fillna({'stalk-root': most_frequent_value}) # Alternative C: Fill missing values with the mode of 'stalk-root'
df_mushrooms['poisonous'] = df_mushrooms['poisonous'].map({'e': 0, 'p': 1})
df_mushrooms.drop(columns=['veil-type'], inplace=True)

st.markdown("""
The features that have been removed from the dataset were:

- **odor**
- **spore-print-color**
- **ring-type**
- **gill-color**
- **stalk-surface-above-ring**
- **stalk-surface-below-ring**
- **bruises**
- **gill-size**
- **stalk-color-above-ring**
- **stalk-color-below-ring**
- **population**
- **veil-color**
- **cap-surface**
- **gill-spacing**
- **veil-type**
""")

st.markdown("""
The target variable "poisonous" has been turned into a dummy variable, presenting a value of 1 for a mushroom that is poisonous and a value of 0 for a mushroom that is edible.
""")

st.markdown("---")

st.title("Modified Dataset")
st.dataframe(df_mushrooms, use_container_width=True)



# Function to get distinct categories for each categorical variable
def get_distinct_categories(df):
    categories = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[column].unique().tolist()
        valid_values = [str(value) for value in unique_values if isinstance(value, str)]
        categories[column] = valid_values
    return categories

# Get distinct categories
distinct_categories = get_distinct_categories(df_mushrooms)

# Display the distinct categories in a compact format using columns
st.markdown("### Distinct Categories for Each Variable")

col1, col2, col3 = st.columns(3)

for i, (variable, categories) in enumerate(distinct_categories.items()):
    color_style = "color:#FF855C;"
    font_size_style = "font-size:14px;"
    formatted_variable = f"<span style='{color_style} {font_size_style}'>{variable.capitalize()}</span>"
    
    category_list = f"<ul style='{font_size_style}'>"
    for category in categories:
        category_list += f"<li>{category}</li>"
    category_list += "</ul>"

    if i % 3 == 0:
        with col1:
            st.markdown(f"#### {formatted_variable}", unsafe_allow_html=True)
            st.markdown(category_list, unsafe_allow_html=True)
    elif i % 3 == 1:
        with col2:
            st.markdown(f"#### {formatted_variable}", unsafe_allow_html=True)
            st.markdown(category_list, unsafe_allow_html=True)
    else:
        with col3:
            st.markdown(f"#### {formatted_variable}", unsafe_allow_html=True)
            st.markdown(category_list, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<a name="cramers-v-matrix"></a>', unsafe_allow_html=True)
st.markdown("### Cramér's V Matrix")
st.markdown("##### Cramér's V matrix is a statistical tool used to measure the strength of association between pairs of categorical variables in a dataset. It extends the concept of Cramér's V, which ranges from 0 (no association) to 1 (perfect association), to a matrix format where each cell represents the Cramér's V value for a specific pair of variables. This matrix helps in identifying and visualizing the relationships between multiple categorical variables, providing insights into their interdependencies.")
st.markdown("Below is the formula for calculating Cramér's V:")

st.latex(r'''V = \sqrt{\frac{\chi^2}{n \times (\min(r, k) - 1)}}''')
st.markdown("""
- **V**: Cramér's V, the measure of association between two categorical variables.
- **$\chi^2$**: The Chi-Square statistic, calculated from the contingency table of the two categorical variables.
- **n**: The total number of observations in the dataset.
- **r**: The number of rows in the contingency table.
- **k**: The number of columns in the contingency table.
- **$min(r, k)$**: The minimum of the number of rows and columns in the contingency table, used to adjust the degrees of freedom.
""")


# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n == 0 or min(r, k) == 1:
        return 0  # Return 0 if there is no data or if the table is degenerate
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Load your dataset
features = df_mushrooms.columns

# Initialize Cramér's V matrix
cramers_v_matrix = pd.DataFrame(index=features, columns=features)

# Calculate Cramér's V for each pair of categorical features
for feature1 in features:
    for feature2 in features:
        confusion_matrix = pd.crosstab(df_mushrooms[feature1], df_mushrooms[feature2])
        cramers_v_matrix.loc[feature1, feature2] = cramers_v(confusion_matrix)

# Convert to float
cramers_v_matrix = cramers_v_matrix.astype(float)

# Plot the Cramér's V matrix using Plotly
fig = go.Figure(data=go.Heatmap(
    z=cramers_v_matrix.values,
    x=cramers_v_matrix.columns,
    y=cramers_v_matrix.index,
    colorscale='Viridis',  # Change to a valid color scale
    text=cramers_v_matrix.values,
    texttemplate="%{text:.2f}",
    hoverinfo='z'
))

fig.update_layout(
    xaxis_nticks=36,
    height=800,
    width=1000
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)

st.markdown("""<h5>This Matrix is very useful for identifying Correlations or problems with <a style='color: #FF855C' href="https://en.wikipedia.org/wiki/Multicollinearity">Multicolilinearity</a></h5>""", unsafe_allow_html=True)
