# =============================================================================
# PROBLEM C1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Desired loss (MSE) < 1e-4
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


def solution_C1():
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ], dtype=float)

    # YOUR CODE HERE
    model = Sequential()
    model.add(Dense(526, input_dim=1))
    model.add(Dense(256))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(X, Y, epochs=150, batch_size=16, verbose=1)

    print(model.predict([-2.0, 10.0]))
    return model

# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C1()
    model.save("model_C1.h5")