import ff as ff
import matplotlib as matplotlib
import numpy as np
import plotly as py
import plotly.express as px
import plotly.offline as pyo
import sklearn as sk
import voila as vo
from pandas import read_csv, __version__, compat

pyo.init_notebook_mode()

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
# fig = px.pie(df, names='Outcome', title="Diabetes Outcome Distribution", color_discrete_sequence=["green", "red"],
#                 category_orders={'Outcome': [0, 1]})
# fig.show()
#
# # Distribution of the features
# fig = px.histogram(df, x="Pregnancies", title="Pregnancies Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="Glucose", title="Glucose Distribution", color_discrete_sequence=["green"])
# fig.show()
#
#
# fig = px.histogram(df, x="BloodPressure", title="BloodPressure Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="SkinThickness", title="SkinThickness Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="Insulin", title="Insulin Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="BMI", title="BMI Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="DiabetesPedigreeFunction", title="DiabetesPedigreeFunction Distribution",
#                    color_discrete_sequence=["green"])
# fig.show()
#
# fig = px.histogram(df, x="Age", title="Age Distribution", color_discrete_sequence=["green"])
# fig.show()
#
# # Distribution of the features and the target variable
# fig = px.histogram(df, x="Pregnancies", title="Pregnancies Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="Glucose", title="Glucose Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="BloodPressure", title="BloodPressure Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="SkinThickness", title="SkinThickness Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="Insulin", title="Insulin Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="BMI", title="BMI Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="DiabetesPedigreeFunction", title="DiabetesPedigreeFunction Distribution", color="Outcome",
#                    color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.histogram(df, x="Age", title="Age Distribution", color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# # Relationship between the features and the target variable
# fig = px.scatter(df, x="Pregnancies", y="Outcome", title="Pregnancies vs Outcome",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="Outcome", title="Glucose vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="Outcome", title="BloodPressure vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="SkinThickness", y="Outcome", title="SkinThickness vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Insulin", y="Outcome", title="Insulin vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BMI", y="Outcome", title="BMI vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="DiabetesPedigreeFunction", y="Outcome", title="DiabetesPedigreeFunction vs Outcome",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Age", y="Outcome", title="Age vs Outcome", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# # Relationship between the features
# fig = px.scatter(df, x="Pregnancies", y="Glucose", title="Pregnancies vs Glucose", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="BloodPressure", title="Pregnancies vs BloodPressure", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="SkinThickness", title="Pregnancies vs SkinThickness", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="Insulin", title="Pregnancies vs Insulin", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="BMI", title="Pregnancies vs BMI", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="DiabetesPedigreeFunction", title="Pregnancies vs DiabetesPedigreeFunction", color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Pregnancies", y="Age", title="Pregnancies vs Age", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="BloodPressure", title="Glucose vs BloodPressure", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="SkinThickness", title="Glucose vs SkinThickness", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="Insulin", title="Glucose vs Insulin", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="BMI", title="Glucose vs BMI", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="DiabetesPedigreeFunction", title="Glucose vs DiabetesPedigreeFunction",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Glucose", y="Age", title="Glucose vs Age", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="SkinThickness", title="BloodPressure vs SkinThickness", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="Insulin", title="BloodPressure vs Insulin", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="BMI", title="BloodPressure vs BMI", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="DiabetesPedigreeFunction",
#                  title="BloodPressure vs DiabetesPedigreeFunction", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BloodPressure", y="Age", title="BloodPressure vs Age", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="SkinThickness", y="Insulin", title="SkinThickness vs Insulin", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="SkinThickness", y="BMI", title="SkinThickness vs BMI", color="Outcome",
#                 color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="SkinThickness", y="DiabetesPedigreeFunction",
#                  title="SkinThickness vs DiabetesPedigreeFunction",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="SkinThickness", y="Age", title="SkinThickness vs Age",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Insulin", y="BMI", title="Insulin vs BMI", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Insulin", y="DiabetesPedigreeFunction", title="Insulin vs DiabetesPedigreeFunction",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="Insulin", y="Age", title="Insulin vs Age", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BMI", y="DiabetesPedigreeFunction", title="BMI vs DiabetesPedigreeFunction",
#                  color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="BMI", y="Age", title="BMI vs Age", color="Outcome", color_discrete_sequence=["green", "red"])
# fig.show()
#
# fig = px.scatter(df, x="DiabetesPedigreeFunction", y="Age", title="DiabetesPedigreeFunction vs Age", color="Outcome",
#                  color_discrete_sequence=["green", "red"])
# fig.show()

# Load the diabetes dataset
data = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
1,103,30,38,83,43.3,0.183,33,0
1,115,70,30,96,34.6,0.529,32,1"""

# Create a DataFrame
# df = read_csv(compat.StringIO(data), index_col=0)

# Generate a scatter matrix plot
fig2 = px.scatter_matrix(df,
                         dimensions=["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction",
                                     "Age"],
                         color="Outcome",
                         title="Diabetes Feature Relationships",
                         color_discrete_sequence=["green", "red"],
                         category_orders={'Outcome': [0, 1]}
                         )
fig2.update_layout(
    width=950,
    height=950,
)
fig2.show()

# Correlation Matrix
corr = df.corr()
fig = px.imshow(corr, title="Correlation Matrix")
fig.show()

# Heatmap
fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Viridis')
fig.update_layout(
    width=950,
    height=950,
    title_text='Correlation Matrix',
    xaxis_title="Features",
    yaxis_title="Features",
)
fig.show()

# Histogram
fig = px.histogram(df, x="Age", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Box Plot
fig = px.box(df, x="Outcome", y="Age", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Violin Plot
fig = px.violin(df, y="Age", x="Outcome", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Density Plot
fig = px.density_contour(df, x="Age", y="BMI", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Density Heatmap
fig = px.density_heatmap(df, x="Age", y="BMI", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Histogram 2D
fig = px.histogram_2d(df, x="Age", y="BMI", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Histogram 2D Contour
fig = px.histogram_2d_contour(df, x="Age", y="BMI", color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Parallel Coordinates
fig = px.parallel_coordinates(df, color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Parallel Categories
fig = px.parallel_categories(df, color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()

# Scatter Plot Matrix
fig = px.scatter_matrix(df, dimensions=["Age", "BMI"], color="Outcome", color_discrete_sequence=["green", "red"])
fig.show()


