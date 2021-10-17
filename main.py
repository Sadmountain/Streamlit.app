from logging import critical
from math import e
from os import name
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


#Page layout
st.set_page_config(page_title='Banknote predictions')

df = pd.read_csv('data/banknote_authentication.csv')

#Preprocessing
# Normalizing the values of all columns
for column in df.columns:
    min_value = df[column].min()
    max_value = df[column].max()
    df[column] = df[column].apply(lambda x: (x-min_value)/(max_value-min_value))

# Seperate the class label from the dataframe
y = df['class']
X = df.drop(['class'], 1)

# Create the test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

header = st.container()
dataset = st.container()
models = st.container()


with header:
    st.title('Banknote predictions')


with dataset:
    st.header('Data introduction')

    checkbox = st.sidebar.checkbox("Reveal data")
    print(checkbox)

    if checkbox:
        st.dataframe(df.style.highlight_max(axis=1))
    
    #create scatterplot
    st.sidebar.subheader("Scatterplot parameters")
    #add select widget
    select_boxX = st.sidebar.selectbox(label = 'X axis', options= df.columns)
    select_boxY = st.sidebar.selectbox(label = 'Y axis', options= df.columns)

    #create scatterplot
    fig = sns.relplot(x=select_boxX, y=select_boxY, data=df, hue=df['class'])
    st.pyplot(fig)

    st.text("---"*200)

    
    
    if st.checkbox("show visualization, which shows the characteristics of images of class 1, compared to images of class 0"):
        dimensions = ['variance', 'skewness', 'curtosis', 'entropy']

        fig = px.scatter_matrix(
            df,
            dimensions=dimensions,
            color='class',
            labels={'class': 'Different<br>classes'},
            title="Characteristics of different classes"
        )

        st.write(fig)

    if st.checkbox("Show KNN scatterplot"):
        y = df['class']
        X = df.drop(['class'], 1)


        
         # Define the model
        knn = KNeighborsClassifier(n_neighbors=2)

        # Train the model
        knn.fit(X_train.values, y_train.values)

        # Make predictions
        knn_predictions = knn.predict(X_test.values)

        # Visualize the kNN scatter plot
        # Source: https://plotly.com/python/knn-classification/
        fig = px.scatter(
            X_test, x='variance', y='skewness',
            color=knn_predictions, color_continuous_scale='RdBu',
            symbol=y_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
            labels={'symbol': 'label', 'color': 'Score of <br>first class'},
            title="K-Nearest Neighbor (Training data)"
        )
        fig.update_traces(marker_size=12, marker_line_width=1.5)
        fig.update_layout(legend_orientation='h')
        st.write(fig)

with models:
    st.header('Models')
    #create models
    st.sidebar.subheader("Create models")
    seed = st.sidebar.slider('Seed', 1, 200)
    classifier_name = st.sidebar.selectbox('Select Classifier', ('SVM', 'KNN', 'Decision tree', 'LogisticRegression', 'naive_bayes', 'Random_forest'))

    #Algorithm
    def add_parameter(name_of_clf):
        params = dict()
        if name_of_clf== 'SVM':
            c = st.sidebar.slider('C', 0.01, 10.0)
            params['C']=c
            kernel = st.sidebar.selectbox('kernel',('linear', 'poly', 'rbf', 'sigmoid'))
            params['kernel'] = kernel
        elif name_of_clf=='KNN':
            k = st.sidebar.slider('k', 1, 15)
            params['K'] = k
        elif name_of_clf=='LogisticRegression':
            penalty = st.sidebar.selectbox('penalty', ('none', 'l1', 'l2', 'elasticnet'))
            params['penalty'] = penalty
        else:
            name_of_clf == 'Decision tree'
            criterion = st.sidebar.selectbox('criterion',( "gini", "entropy"))
            params['criterion']= criterion
        return params
    

    params = add_parameter(classifier_name)

    def get_classifier(name_of_clf, params):
        clf=None
        if name_of_clf=='SVM':
            clf= SVC(C=params['C'])
            clf = SVC(kernel = params['kernel'])
        elif name_of_clf=='KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        elif name_of_clf=='Decision tree':
            clf = DecisionTreeClassifier(criterion = params['criterion'])
        elif name_of_clf=='LogisticRegression':
            clf = LogisticRegression(penalty = params['penalty'])
        elif name_of_clf=='naive_bayes':
            clf = GaussianNB()
        elif name_of_clf=='Random_forest':
            clf = RandomForestClassifier(criterion = params['criterion'])
        else:
            st.warning('Select algorithm')
        return clf
    

    clf=get_classifier(classifier_name, params)

    # Create the test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    st.write(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    st.write('Classifier_name:', classifier_name)
    st.write('Accuracy for model:', accuracy)







