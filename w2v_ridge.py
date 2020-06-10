import pandas as pd 
from tensorflow import keras
import numpy as np 
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection, metrics
from utilities import *
import re
from real_or_not_vectorizer import ridge

def get_vectors(data_set):
    vectors = []
    for i, vector in data_set.iterrows():
        vectors.append(vector.to_numpy())
    return np.array(vectors)

if __name__ == '__main__':

    train_data = pd.read_csv('vectors_25d_train.csv', dtype='float32')
    test_data = pd.read_csv('vectors_25d_test.csv', dtype='float32')
    target = pd.read_csv('data/train.csv')
    test_df = pd.read_csv("data/test.csv").fillna('nan')

    # Preparing data for training
    X = get_vectors(train_data) 
    y = target.target
    X_test = get_vectors(test_data)

    #plot(X, y)

    output(ridge(X, y, X_test), 'ridge_submission_w2v')