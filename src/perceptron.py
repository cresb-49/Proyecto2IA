import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epocas=100):
        self.lr = learning_rate
        self.epocas = epocas
        self.pesos = None
        self.sesgo = None

    def activacion(self, entradas):
        z = np.dot(self.pesos, entradas)
        return 1 if z + self.sesgo > 0 else 0

    def entrenar(self, X, y):
        self.pesos = np.random.uniform(-1, 1, size=X.shape[1])
        self.sesgo = np.random.uniform(-1, 1)
        errores_por_epoca = []

        for _ in range(self.epocas):
            error_total = 0
            for xi, yi in zip(X, y):
                pred = self.activacion(xi)
                error = yi - pred
                error_total += error**2

                self.pesos += self.lr * error * xi
                self.sesgo += self.lr * error

            errores_por_epoca.append(error_total)
        return errores_por_epoca

    def predecir(self, X):
        return np.array([self.activacion(xi) for xi in X])
