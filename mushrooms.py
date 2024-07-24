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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import GridSearchCV
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.figure_factory as ff


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
<h5>It is important to highlight that the project was developed following <a href="https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining" style="color:#FF855C;">crisp-DM</a> (Cross Industry Standard Process for Data Mining) methodology, and it follows iteration cycles that span over Business Understanding, Data Understanding, Data Preparation, Modelling and Evaluation, with emphasis on experimenting with a handful of ML models in order to see possible improvements in accuracy. </h5>
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

@st.cache_data
def load_data_from_s3(bucket_name, object_key):
    s3_uri = f's3://{bucket_name}/{object_key}'
    df = pd.read_csv(s3_uri, storage_options={'anon': True})
    return df

df = load_data_from_s3(bucket_name, object_key)


st.markdown("---")

st.title("Original Dataset")

st.dataframe(df, use_container_width=True)

st.markdown("""
<h3 style="color:#FF855C;">Data Preparation</h3>
            """, unsafe_allow_html=True)

st.markdown("""
##### The original dataset presented above was preliminarily used to train a simple <a style='color: #FF855C' href="https://en.wikipedia.org/wiki/Decision_tree">Decision Tree</a>, and since the accuracy for that model was always 100%, I have decided to eliminate several features in order to make the predictions more challenging.
""", unsafe_allow_html=True)






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
@st.cache_data
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

@st.cache_data
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

st.markdown("""<h5>This Matrix is very useful for identifying Correlations or problems with <a style='color: #FF855C' href="https://en.wikipedia.org/wiki/Multicollinearity">Multicollinearity</a></h5>""", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<h3 style="color:#FF855C;">Modelling</h3>
            """, unsafe_allow_html=True)
st.markdown('''##### For the modelling, we will be approaching the problem by implementing the simplest models first and then we will progressively move into more complex approaches. For this first case, we will be implementing a <a style='color: #FF855C' href="https://en.wikipedia.org/wiki/Decision_tree">Decision Tree</a>.''' , unsafe_allow_html=True)
st.markdown('''##### ''')


######################################################### code
df_dummies = pd.get_dummies(df_mushrooms.drop('poisonous', axis=1))
X = df_dummies
y = df_mushrooms['poisonous']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
clf = DecisionTreeClassifier(random_state=89)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = sk_confusion_matrix(y_test, y_pred)
########################################################## code



st.write("""
##### Model Description
In this analysis, a simple classification tree was employed to train the mushroom dataset. 
The goal was to predict whether a mushroom is poisonous or edible based on its various features. 
A decision tree is a powerful machine learning algorithm that works well for classification tasks. 

##### Model Training
The `DecisionTreeClassifier` from the `scikit-learn` library was used for this purpose. The model was trained 
using the default settings, which means no hyperparameter tuning was performed. The decision tree was 
constructed using a subset of the dataset, with 70% of the data used for training and 30% reserved for testing.

<h3 style="color:#FF855C;">Evaluation</h3>
After training the model, its performance was evaluated on the test set. The resulting confusion matrix and accuracy score are displayed below. The confusion matrix provides a clear visual 
representation of the model's performance, showing the counts of true positives, true negatives, false positives, 
and false negatives.
""", unsafe_allow_html=True)




# Plotting the confusion matrix with Plotly
z = conf_matrix
x = ['Predicted: Edible', 'Predicted: Poisonous']
y = ['Actual: Edible', 'Actual: Poisonous']

# Creating annotations for the heatmap
z_text = [[str(y) for y in x] for x in z]

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues', showscale=True)

# Adding title and labels
fig.update_layout(
                  xaxis_title='Predicted Label',
                  yaxis_title='True Label',
                  title_x=0.5)  # Center the title

# Adjusting layout to make the plot square
fig.update_layout(
    width=900,
    height=600,
    margin=dict(l=50, r=50, t=150, b=50)
)

# Center the plot in Streamlit
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.plotly_chart(fig)

# Display accuracy in the middle with a larger font
st.markdown(
    f"""
    <div style="text-align:center; font-size:24px; font-weight:bold;">
        Accuracy: {accuracy:.6%}
    </div>
    """,
    unsafe_allow_html=True
)

st.write("<br><br>", unsafe_allow_html=True)

st.write("""
##### Although we got a pretty decent accuracy score for a simple decision tree, we still need to improve it due to the impact that a misleading prediction could cause for this specific case. Since the Business and Data Understanding phases, it is crucial for us to evaluate the impact of potential misleading predictions in our final model. Some domains such as Healthcare, Finances or even problems like ours cannot afford many mistakes in predictions.
""", unsafe_allow_html=True)


st.markdown("---")

st.write("""
<h3 style="color:#FF855C;">Modelling</h3>
After training the model in a simple classification tree, it is time for us to take a step further and experiment with higher complexity
models. Here, we are leveraging the data preprocessing done in the previous steps in order to create a <a href="https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,Decision%20trees" style="color:#FF855C;">Random Forest</a> model.
""", unsafe_allow_html=True)

st.write("<br><br>", unsafe_allow_html=True)

st.write("""
##### Model Training
The `RandomForestClassifier` from the `scikit-learn` library was used for this purpose. Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks. The model was trained on 70% of the dataset and tested on the remaining 30%.

""", unsafe_allow_html=True)

st.write("""
<h3 style="color:#FF855C;">Evaluation</h3>
After training the model, its performance was evaluated on the test set. The resulting confusion matrix and accuracy score are displayed below. The confusion matrix provides a clear visual 
representation of the model's performance, showing the counts of true positives, true negatives, false positives, 
and false negatives.
""", unsafe_allow_html=True)





################################################################################# CODE
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def random_forest_classifier(df, target_column, test_size=0.3, random_state_split=99, random_state_clf=99):
    df_dummies = pd.get_dummies(df.drop(target_column, axis=1))
    X = df_dummies
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state_split)
    rf_clf = RandomForestClassifier(random_state=random_state_clf)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    report = classification_report(y_test, y_pred_rf)
    conf_matrix = sk_confusion_matrix(y_test, y_pred_rf)
    
    return accuracy, report, conf_matrix
################################################################################# CODE


accuracy, report, conf_matrix = random_forest_classifier(df_mushrooms, 'poisonous')

# Plotting the confusion matrix with Plotly
z = conf_matrix
x = ['Predicted: Edible', 'Predicted: Poisonous']
y = ['Actual: Edible', 'Actual: Poisonous']

# Creating annotations for the heatmap
z_text = [[str(y) for y in x] for x in z]

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues', showscale=True)

# Adding title and labels
fig.update_layout(
                  xaxis_title='Predicted Label',
                  yaxis_title='True Label',
                  title_x=0.5)  # Center the title

# Adjusting layout to make the plot square
fig.update_layout(
    width=900,
    height=600,
    margin=dict(l=50, r=50, t=150, b=50)
)

# Center the plot in Streamlit
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.plotly_chart(fig)

# Display accuracy in the middle with a larger font
st.markdown(
    f"""
    <div style="text-align:center; font-size:24px; font-weight:bold;">
        Accuracy: {accuracy:.6%}
    </div>
    """,
    unsafe_allow_html=True
)

st.write("<br><br>", unsafe_allow_html=True)

st.write("""
##### Since we did not observe any change for the accuracy metric using the Random Forest model, we can try a last method in order to train models based on trees: Gridsearch and Cross-Validation in order to tune our hyperparameters.
""", unsafe_allow_html=True)

st.markdown("---")

st.write("""
<h3 style="color:#FF855C;">Modelling</h3>
Grid search is a method to find the optimal hyperparameters for a model by exhaustively searching through a specified parameter grid. Cross-validation is a technique to assess the model's performance by dividing the dataset into training and validation sets multiple times. These techniques help improve model accuracy by ensuring the chosen parameters generalize well to unseen data. Here we are using GridSearchCV from Scikit-Learn to perform these techniques.
Additionally, we are also running a cross a k-fold cross-validation algorithm, that basically experiments with different train/test splits and then, using an aggregation function, finds an optimal value for the split. Below is an image that explains how K-fold works.
""", unsafe_allow_html=True)

st.write("<br><br>", unsafe_allow_html=True)

st.write("""
<img src="https://user-images.githubusercontent.com/26833433/258589390-8d815058-ece8-48b9-a94e-0e1ab53ea0f6.png" style="height:400px; display: block; margin: auto;">

 """, unsafe_allow_html=True)


st.write("<br><br>", unsafe_allow_html=True)
#################################################################################
@st.cache_data
def perform_grid_search(X_train, y_train):
    rf_clf2 = RandomForestClassifier()
    params_grid = {
        'max_depth': [12],
        'min_samples_leaf': [1],
        'n_estimators': list(range(0, 300, 10))
    }
    grid_rf = GridSearchCV(estimator=rf_clf2,
                           param_grid=params_grid,
                           scoring='accuracy',
                           cv=5)
    grid_rf.fit(X_train, y_train)
    return grid_rf
grid_rf = perform_grid_search(X_train, y_train)
results = grid_rf.cv_results_
mean_test_scores = results['mean_test_score']
n_estimators = list(range(0, 300, 10))
######################################################################################

st.write("""

Down below you are able to see a plot showing the accuracy of a Random Forest model as a function of the number of estimators. Each point represents the mean test accuracy obtained using cross-validation for a specific number of estimators, ranging from 0 to 300 in increments of 10. This plot helps to identify the optimal number of estimators for the highest accuracy, and it is possible to notice that, no matter the complexity that we imply to our model, it keeps its performance stable over time, therefore we can conclude that, using these algorithms, we are probably not going to see any improvements with the accuracy metric.
""", unsafe_allow_html=True)



# Create the plot
fig = go.Figure()

# Add the line
fig.add_trace(go.Scatter(
    x=n_estimators, 
    y=mean_test_scores, 
    mode='lines+markers',
    name='Test Accuracy',
    marker=dict(symbol='circle')
))

# Update layout
fig.update_layout(
    title='Random Forest Accuracy vs. Number of Estimators',
    xaxis_title='Number of Estimators',
    yaxis_title='Accuracy',
    legend_title='Legend',
    width=800,
    height=400,
    template='plotly_white'
)

# Add grid
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Center the plot using st.columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")  # Empty column for spacing

with col2:
    st.plotly_chart(fig)  # Centered column with the plot

with col3:
    st.write("")  # Empty column for spacing


st.write("<br><br>", unsafe_allow_html=True)


st.write("""
#### We have tried several techniques, from simple clf trees to more advanced logistic regression with l2 regularization (Notebook) and cross-validation. Even with all these advanced techniques, a simple classification tree presented the best "score/effort" relationship, but we had to try other models in order to reach that conclusion.
""", unsafe_allow_html=True)