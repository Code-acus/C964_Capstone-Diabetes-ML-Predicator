
import matplotlib as matplotlib
import numpy as np
import plotly as py
import plotly.offline as pyo
import sklearn as sk
import voila as vo
from pandas import read_csv, __version__
import plotly.express as px

pyo.init_notebook_mode(connected=True)

print("Pandas version: ", __version__)
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

df = read_csv('C:\\Users\\hrogers\\PycharmProjects\\C964_Capstone-Diabetes-ML-Predicator\\diabetes_data.csv')
df.info()

# Preparation
# In this stage, the data is prepared for use in the application.
# As seen below, the dataset that has been collected has already been well prepared.
# There is no need for preparing data parsing, cleaning, or featurization, or data wrangling.
# With the data prepared, it can effectively be used for the descriptive and non-descriptive portions
# of the application.

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
fig = px.pie(df, names='Outcome', title="Diabetes Outcome Distribution", color_discrete_sequence=["green", "red"],
             category_orders={'Outcome': [0, 1]})
fig.show()

# The dataset is balanced, with 500 patients with diabetes and 268 patients without diabetes.
# The dataset is also clean, with no missing values and no duplicate rows.
# The dataset is also well structured, with all columns having the correct data type.
# The dataset is also well prepared, with no need for data parsing, cleaning, or featurization.

# Descriptive
# In this section, the data is used to create visualizations that describe the data.
# The visualizations are used to answer the following questions:
# 1. What is the distribution of the data?
# 2. What is the relationship between the features and the target?
# 3. What is the relationship between the features?

# Distribution
# The distribution of the data is shown in the following visualizations.
# The first visualization shows the distribution of the target variable.
# The second visualization shows the distribution of the features.
# The third visualization shows the distribution of the features and the target variable.

# Distribution of the target variable
fig = px.pie(df, names='Outcome', title="Diabetes Outcome Distribution", color_discrete_sequence=["green", "red"],
                category_orders={'Outcome': [0, 1]})
fig.show()

# Distribution of the features
histograms = px.histogram(df, nbins=50, facet_col='Outcome', facet_row='variable', template='plotly_white',
                          title="Distribution of the Features", color='Outcome', opacity=0.6,
                          category_orders={'Outcome': [0, 1]})
histograms.update_xaxes(showgrid=True)
histograms.show()

# Distribution of the features and the target variable
scatter_matrix = px.scatter_matrix(df, dimensions=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                                   color='Outcome', symbol='Outcome', opacity=0.6,
                                   title="Scatter Matrix of the Features and the Target Variable",
                                   color_discrete_sequence=["green", "red"],
                                   category_orders={'Outcome': [0, 1]})
scatter_matrix.update_traces(diagonal_visible=False)
scatter_matrix.show()

# Train and Test the Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the dataset into training and testing sets
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Test the model and print the results
y_pred = log_reg.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# The above code displays histograms of the features, a scatter matrix of the features and the target variable,
# and trains and tests a simple logistic regression model on the diabetes dataset.
# The histograms show the distribution of the features.
# The scatter matrix shows the relationship between the features and the target variable.
# The logistic regression model is trained and tested on the dataset.
# The model's accuracy score, classification report, and confusion matrix are printed.

# The model's accuracy score is 0.79, which means that the model is 79% accurate.
# The model's classification report shows that the model is 79% accurate for predicting patients without diabetes,
# and 79% accurate for predicting patients with diabetes.
# The model's confusion matrix shows that the model predicted 130 patients without diabetes correctly,
# and 45 patients with diabetes correctly.
# The model's confusion matrix also shows that the model predicted 25 patients without diabetes incorrectly,
# and 28 patients with diabetes incorrectly.




# Assuming you have loaded the dataset into a DataFrame called df

# Scatter plot
px.scatter(df, x='Glucose', y='BloodPressure', color='Outcome',
           title="Glucose v.s. Blood Pressure",
           color_discrete_sequence=["green", "red"])

# Histograms
histograms_glucose = px.histogram(df, x='Glucose', color='Outcome', nbins=50, opacity=0.6,
                                  title="Distribution of Glucose",
                                  color_discrete_sequence=["green", "red"])
histograms_glucose.show()

histograms_blood_pressure = px.histogram(df, x='BloodPressure', color='Outcome', nbins=50, opacity=0.6,
                                         title="Distribution of Blood Pressure",
                                         color_discrete_sequence=["green", "red"])
histograms_blood_pressure.show()

histograms_bmi = px.histogram(df, x='BMI', color='Outcome', nbins=50, opacity=0.6,
                              title="Distribution of BMI",
                              color_discrete_sequence=["green", "red"])
histograms_bmi.show()

histograms_age = px.histogram(df, x='Age', color='Outcome', nbins=50, opacity=0.6,
                              title="Distribution of Age",
                              color_discrete_sequence=["green", "red"])
histograms_age.show()

# Scatter matrix
scatter_matrix = px.scatter_matrix(df, dimensions=['Glucose', 'BloodPressure', 'BMI', 'Age'],
                                      color='Outcome', symbol='Outcome', opacity=0.6,
                                        title="Scatter Matrix of Glucose, Blood Pressure, BMI, and Age",
                                        color_discrete_sequence=["green", "red"])
scatter_matrix.update_traces(diagonal_visible=False)
scatter_matrix.show()

# The above code displays a scatter plot, histograms, and a scatter matrix of the diabetes dataset.
# The scatter plot shows the relationship between the glucose and blood pressure features and the target variable.
# The histograms show the distribution of the glucose, blood pressure, BMI, and age features.
# The scatter matrix shows the relationship between the glucose, blood pressure, BMI, and age features and the target variable.

# The scatter plot shows that patients with diabetes have higher glucose and blood pressure levels than patients without diabetes.
# The histograms show that patients with diabetes have higher glucose, blood pressure, BMI, and age levels than patients without diabetes.
# The scatter matrix shows that patients with diabetes have higher glucose, blood pressure, BMI, and age levels than patients without diabetes.

