
import matplotlib as matplotlib
import numpy as np
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objs as go
from ipywidgets import HBox, VBox
from jedi.api.refactoring import inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly as py
import sklearn as sk
import voila as vo
from sklearn import metrics
from sklearn import svm
import plotly.offline as pyo
import pandas as pd
import plotly.express as px
pyo.init_notebook_mode()

print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)
print("Matplotlib version: ", matplotlib.__version__)
print("Plotly version: ", py.__version__)
print("Scikit-learn version: ", sk.__version__)
print("Voila version: ", vo.__version__)

# Data
# In this section, data is collected and later prepared to support visualizations
# and the machine learning algorithm.
#
# Collection
# In this stage the application's dataset retrieved from Kaggle is converted from a CSV file to a Pandas DataFrame.
# Information about the DataFrame is then output for inspection.

df = pd.read_csv('C:\\Users\\hrogers\\PycharmProjects\\C964_Capstone-Diabetes-ML-Predicator\\diabetes_data.csv')
df.info()

# Preparation
# In this stage, the data is prepared for use in the application.
# As seen below, the dataset that has been collected has already been well prepared.
# Their is no need for preparing data parsing, cleaning, or featurization, or data wrangling.
# With the data prepared,it can effectively be used for the descriptive and non-descriptive portions
# of the application.application

df.head(8)

# Data wrangling refers to the process of cleaning, structuring, and enriching raw data to make it more suitable
# for analysis or training machine learning models. Based on the diabetes dataset, that data is already
# structured and clean.
#
# However, it's essential to perform some checks to ensure the dataset is ready for analysis.
#
# Here are some steps I am using inspect and prepare my dataset:
# python

df.isnull().sum()

df.dtypes

df.describe()

df.duplicated().sum()

# Assuming 'df' is your diabetes dataset DataFrame
fig = px.pie(df, names='Outcome', title="Diabetes Outcome Distribution", color_discrete_sequence=["green", "red"], category_orders={'Outcome': [0, 1]})
fig.show()








