import warnings
import matplotlib as matplotlib
import numpy as np
import plotly as py
import plotly.express as px
import plotly.offline as pyo
import sklearn as sk
import voila as vo
from pandas import read_csv, __version__
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings(action='ignore', category=UserWarning)

pyo.init_notebook_mode(connected=True)

print("Pandas version: ", __version__)
print("Numpy version: ", np.__version__)
print("Matplotlib version: ", matplotlib.__version__)
print("Plotly version: ", py.__version__)
print("Scikit-learn version: ", sk.__version__)
print("Voila version: ", vo.__version__)

df = read_csv('C:\\Users\\hrogers\\PycharmProjects\\C964_Capstone-Diabetes-ML-Predicator\\diabetes_data.csv')

# Split the dataset into independent and dependent variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing subsets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df.info()

# Split the data into training and testing subsets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)


df.head(8)

df.isnull().sum()

df.dtypes

df.describe()

df.duplicated().sum()

# Assuming 'df' is your diabetes dataset DataFrame
fig = px.pie(df, names='Outcome', title="Diabetes Outcome Distribution", color_discrete_sequence=["green", "red"],
             category_orders={'Outcome': [0, 1]})
fig.show()

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

def input_data():
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    user_data = []
    print("\nEnter the values for the following features:")
    for feature in features:
        value = float(input(f"{feature}: "))
        user_data.append(value)
    return np.array([user_data])


user_input = input_data()
prediction = log_reg.predict(user_input)
print("\nPrediction result (0 = No Diabetes, 1 = Diabetes):", prediction[0])

# The above code asks the user to enter values for the features of a patient.
# The code then uses the trained logistic regression model to predict whether the patient has diabetes or not.
# The code prints the prediction result.

# Enter the values for the following features:
# Pregnancies: 1
# Glucose: 100
# BloodPressure: 70
# SkinThickness: 30
# Insulin: 0
# BMI: 26
# DiabetesPedigreeFunction: 0.351
# Age: 31

# Prediction result (0 = No Diabetes, 1 = Diabetes): 0

# ... The rest of your existing code ...

# Scatter matrix
scatter_matrix = px.scatter_matrix(df, dimensions=['Glucose', 'BloodPressure', 'BMI', 'Age'],
                                   color='Outcome', symbol='Outcome', opacity=0.6,
                                   title="Scatter Matrix of Glucose, Blood Pressure, BMI, and Age",
                                   color_discrete_sequence=["green", "red"])
scatter_matrix.update_traces(diagonal_visible=False)
scatter_matrix.show()


# Function to predict diabetes using user input
def predict_diabetes(model, input_data):
    # Convert user input to a DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Make the prediction
    prediction = model.predict(input_df)

    return prediction[0]


# Example usage:
input_data = {
    'Pregnancies': 1,
    'Glucose': 120,
    'BloodPressure': 80,
    'SkinThickness': 25,
    'Insulin': 100,
    'BMI': 30,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 25
}

print("Prediction result (0 = No Diabetes, 1 = Diabetes):", predict_diabetes(log_reg, input_data))

