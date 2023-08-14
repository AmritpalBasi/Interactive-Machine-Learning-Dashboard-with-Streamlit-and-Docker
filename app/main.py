import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Introduction
st.title('Machine Learning Exploration with different models and datasets')
st.write("---")
st.write("### Introduction")
st.write(
    "In this project you will be able to toggle through different datasets and models for exploration. There are currently 3 different models and 3 different datasets to chose from. Project inspired from [Patrick Loeber](https://www.youtube.com/watch?v=Klqn--Mu2pE&t=10s)")


dataset_name = st.sidebar.selectbox(
    'Select Dataset', ("Iris", "Breast Cancer", "Wine"))


classifier_name = st.sidebar.selectbox(
    'Select Classifier', ("KNN", "SVM", "Random Forest"))

# st.write(f"### {dataset_name} dataset with a {classifier_name} classifier")
st.write(f"### {dataset_name} dataset")

# Loading the data


@st.cache_data
def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    function will load the corresponding dataset selected from sklearns library
    '''
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    return data.data, data.target


X, y = load_dataset(dataset_name)

st.write("Number of rows: ", X.shape[0])
st.write("Number of columns: ", X.shape[1])
st.write("Number of unique features: ", len(np.unique(y)))

#  Descriptions of each dataset which will be displayed to the user

IRIS_DESCRIPTION = """
The Iris dataset is one of the most famous datasets in the field of machine learning and statistics. It's used for multiclass classification tasks and is often considered a starting point for learning about classification algorithms. The dataset contains samples from three different species of Iris flowers: Iris setosa, Iris versicolor, and Iris virginica.
Each sample is described by four features: sepal length, sepal width, petal length, and petal width. The task with the Iris dataset is to classify each sample into one of the three species based on these features. The Iris dataset is commonly used for educational purposes due to its simplicity and clarity.
"""

BREAST_CANCER_DESCRIPTION = """
The Breast Cancer dataset is a binary classification problem designed to diagnose breast cancer. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe various properties of the cell nuclei present in the image, including attributes like texture, area, and smoothness.
This dataset is frequently used to build models that can predict whether a breast mass is benign or malignant based on the extracted features. It's an important example in the medical domain, demonstrating the application of machine learning in aiding medical diagnosis.
"""

WINE_DESCRIPTION = """
The Wine dataset is a classic multiclass classification problem often used for testing various machine learning algorithms. It consists of three classes representing three different types of wine. Each class corresponds to a specific grape variety. The dataset contains 13 attributes that describe various chemical properties of the wines.
The goal in using the Wine dataset is to predict the class of the wine based on its chemical attributes. It's commonly used to evaluate the performance of classification algorithms and to showcase their ability to distinguish between multiple classes.
"""


def dataset_information(dataset_name):
    """
    function will provide an explanation on the chosen dataset in the streamlit framework
    """

    if dataset_name == 'Iris':
        st.write(IRIS_DESCRIPTION)
    elif dataset_name == 'Breast Cancer':
        st.write(BREAST_CANCER_DESCRIPTION)
    else:
        st.write(WINE_DESCRIPTION)

    return None


dataset_information(dataset_name)

n_rows = st.slider("Select number of rows to be shown", 5, len(X))


def show_data_sample(X: np.ndarray, y: np.ndarray, n) -> pd.DataFrame:
    '''
    function will return the top n rows of data for the user 
    '''
    dataframe = pd.DataFrame(X)
    dataframe['target'] = y

    st.write(dataframe.head(n))

    return None


show_data_sample(X, y, n_rows)


def add_parameter_ui(classifier_name: str) -> dict:
    '''
    function will return a dictionary containing parameter values selected for each model
    '''
    parameters = dict()

    if classifier_name == 'KNN':
        K = st.sidebar.slider('K value', 1, 15)
        parameters['K'] = K
    elif classifier_name == 'SVM':
        C = st.sidebar.slider('C value', 0.01, 10.00)
        parameters['C'] = C
    else:
        max_depth = st.sidebar.slider('Max tree depth', 2, 15)
        n_estimators = st.sidebar.slider('Number of estimators', 1, 100)
        parameters['max_depth'] = max_depth
        parameters['n_estimators'] = n_estimators

    return parameters


parameters = add_parameter_ui(classifier_name)


def get_classifier(classifer_name, parameters):
    '''
    function will create the model selected by user with chosen paramters
    '''
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=parameters['K'])
    elif classifier_name == 'SVM':
        classifier = SVC(C=parameters['C'])
    else:
        classifier = RandomForestClassifier(
            n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'], random_state=1000)

    return classifier


classifier = get_classifier(classifier_name, parameters)

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1000)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)

st.write("---")
st.write(f"### Model selected : {classifier_name} ")
st.write("Model accuracy : ", round(accuracy, 3))


def classifier_information(classifier_name):
    '''
    Function writes information based on the selected classifier 
    '''

    if classifier_name == 'KNN':
        st.write("### K-Nearest Neighbors (KNN)")
        st.write("K-Nearest Neighbors is a simple classification algorithm that assigns a label to an unknown data point based on the majority class of its k-nearest neighbors in the training dataset. The distance metric used (often Euclidean distance) determines the 'closeness' of data points.")
        st.write("**Relevant Formula:**")
        st.latex(
            r"Prediction(q) = \arg\max_{c} \sum_{p \in \text{neighbors}(q)} \delta(c, \text{class}(p))")

    elif classifier_name == 'SVM':
        st.write("### Support Vector Machines (SVM)")
        st.write("Support Vector Machines are powerful classifiers that aim to find a hyperplane that best separates different classes while maximizing the margin between them. SVMs can also handle non-linear separations through the use of kernel functions.")
        st.write("**Relevant Formula:**")
        st.latex(
            r"\text{Minimize } \|w\|^2 \text{ subject to } y_i (w \cdot x_i + b) \geq 1 \text{ for } i = 1, \ldots, n")

    else:
        st.write("### Random Forests")
        st.write("Random Forests is an ensemble learning method that constructs multiple decision trees during training and combines their outputs to improve generalization and robustness. Each tree is trained on a bootstrapped sample of the training data and makes predictions independently.")
        st.write("**Relevant Formula:**")
        st.latex(r"\hat{y} = \frac{1}{n} \sum_{i=1}^n T_i(x)")


classifier_information(classifier_name)

st.write("---")
st.write("### PCA Plot")
# Plot
pca = PCA(2)
x_projected = pca.fit_transform(X)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel = 'Principle Component 1'
plt.ylabel = 'Principle Component 2'
plt.colorbar()

# plt.show
st.pyplot(fig)
