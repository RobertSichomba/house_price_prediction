# src/data_preprocessing.py
import numpy as np
from keras.datasets import boston_housing

def load_and_preprocess_data():
    # Load Boston Housing data
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    # Normalize the data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data -= mean
    train_data /= std
    test_data -= mean
    test_data /= std

    return train_data, train_targets, test_data, test_targets
