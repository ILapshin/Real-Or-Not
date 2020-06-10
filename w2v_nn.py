import pandas as pd 
from tensorflow import keras
import numpy as np 
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection, metrics
from utilities import *
import re
from real_or_not_vectorizer import ridge

def get_vectors(data_set):
    # Returns a Numpy array of vectors from a Pandas Series
    vectors = []
    for i, vector in data_set.iterrows():
        vectors.append(vector.to_numpy())
    return np.array(vectors)

def Model(input_shape):
    # Creates a neural network model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_shape,)),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fit(model, X, y, to_continue=False, load_weights=False, save_weights=False, load_path='model/weights.h5', save_path='model/weights.h5'):
    # Fits a passed model with X train datasets and y result vector
    if load_weights:
        model.load_weights(load_path)
    if not load_weights or load_weights and to_continue:
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
        hist = model.fit(X, y, validation_split=0.1, epochs=200, callbacks=[callback])
        print(hist.history)
    if save_weights:
        model.save_weights(save_path)

if __name__ == '__main__':

    train_data = pd.read_csv('vectors_25d_train.csv', dtype='float32')
    test_data = pd.read_csv('vectors_25d_test.csv', dtype='float32')
    target = pd.read_csv('data/train.csv')
    test_df = pd.read_csv("data/test.csv").fillna('nan')

    # Preparing data for training
    X = get_vectors(train_data) 
    y = to_categorical(target.target.to_numpy())
    X_test = get_vectors(test_data)
    print(type(X), X.shape[1])
    model = Model(X.shape[1])

    fit(model, X, y, save_weights=True, save_path='model/weights_1020_1024_1024_w2v_25d_sum.h5')

    output(np.argmax(model.predict(X_test), axis=-1), 'prediction_1020_1024_1024_w2v_25d_sum')