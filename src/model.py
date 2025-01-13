# src/model.py
from keras import models, layers

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Single output for regression
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
