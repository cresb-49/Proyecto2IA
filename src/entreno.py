import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from perceptron import Perceptron

def entrenar_red(df, feature_x, feature_y, learning_rate=0.01, epocas=100, split=80):
    X = df[[feature_x, feature_y]].values
    y = df['target'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = 1 - (split / 100)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    perceptron = Perceptron(learning_rate=learning_rate, epocas=epocas)
    errores = perceptron.entrenar(X_train, y_train)
    y_pred = perceptron.predecir(X_test)
    accuracy = np.mean(y_pred == y_test)

    return errores, accuracy, scaler, perceptron