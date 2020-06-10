import numpy as np 
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from utilities import *
from dataprep import *

def ridge(X, y, X_test):
    # Creats a sklearn RidgeClassifier model, fits it and makes a prediction
    clf = linear_model.RidgeClassifier()    
    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring="f1")
    print(scores)
    clf.fit(X, y)
    return clf.predict(X_test)

def nearest_neighbours(X, y, X_test, n_neighbours=10, algorithm='auto'):
    # Creats a sklearn KNeighborsClassifier model, fits it and makes a prediction
    model = KNeighborsClassifier(n_neighbours, algorithm=algorithm)
    scores = model_selection.cross_val_score(model, X, y, cv=3, scoring="f1")
    print(scores)
    model.fit(X, y)
    return model.predict(X_test)

def dessision_tree(X, y, X_test, max_depth=None):
    # Creats a sklearn DecisionTreeClassifier model, fits it and makes a prediction
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=99) 
    scores = model_selection.cross_val_score(model, X, y, cv=3, scoring="f1")
    print(scores)
    model.fit(X, y)
    return model.predict(X_test)

if __name__ == '__main__':
    # Reading datasets as Pandas DataFrames
    train_df = pd.read_csv("data/train.csv").fillna('nan')
    test_df = pd.read_csv("data/test.csv").fillna('nan')

    # In this case I use only text of the train set tweets
    X = clean(train_df['text'])
    y = train_df.target
    X_test = clean(test_df['text'])

    X = lemm(X)
    X_test = lemm(X_test)

    vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(X) # Fitting a vectorizer using train data not to get any values from test data absent in train data
    
    # Implementing a tfidf to a series of data
    #X = vectorizer.transform(X)
    #X_test = vectorizer.transform(X_test)

    count_vectorizer = feature_extraction.text.CountVectorizer(ngram_range=(1, 3))
    count_vectorizer.fit(X)

    X_ngram = vectorizer.transform(X)
    X_ngram_test = vectorizer.transform(X_test)
    
    #plot(X_2gram, y)
    output(test_df.id, ridge(X_ngram, y, X_ngram_test), 'ridge_submission_tfidf_1-2grams')
