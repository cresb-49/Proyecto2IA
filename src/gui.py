from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QMessageBox, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
import numpy as np

from carga import cargar_y_guardar_dataset
from entreno import entrenar_red


class GraficaWindow(QWidget):
    def __init__(self, figure, title="Gráfica"):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        self.setLayout(layout)
        self.show()



class RedNeuronalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador con Red Neuronal")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        self.error_window = None
        self.frontera_window = None

    def setup_ui(self):
        layout = QVBoxLayout()

        param_group = QGroupBox("Parámetros de Entrenamiento")
        grid = QGridLayout()

        self.eta_input = QLineEdit("0.01")
        self.epocas_input = QLineEdit("100")
        self.split_input = QLineEdit("80")

        grid.addWidget(QLabel("Tasa de aprendizaje (η):"), 0, 0)
        grid.addWidget(self.eta_input, 0, 1)

        grid.addWidget(QLabel("Épocas máximas:"), 1, 0)
        grid.addWidget(self.epocas_input, 1, 1)

        grid.addWidget(QLabel("Porcentaje de entrenamiento:"), 2, 0)
        grid.addWidget(self.split_input, 2, 1)

        self.train_btn = QPushButton("Entrenar")
        self.train_btn.clicked.connect(self.entrenar)
        grid.addWidget(self.train_btn, 3, 0, 1, 2)

        self.load_data_btn = QPushButton("Cargar Datos")
        self.load_data_btn.clicked.connect(self.cargar_datos)
        grid.addWidget(self.load_data_btn, 4, 0, 1, 2)

        param_group.setLayout(grid)
        layout.addWidget(param_group)

        self.accuracy_label = QLabel("Accuracy: ---")
        layout.addWidget(self.accuracy_label)

        self.setLayout(layout)
    
    def mostrar_frontera_decision(self, perceptron, scaler, feature_x, feature_y):
        X = self.df[[feature_x, feature_y]].values
        y = self.df['target'].values

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)

        Z = perceptron.predecir(grid_scaled).reshape(xx.shape)
        Z = Z.reshape(xx.shape)

        fig2 = Figure(figsize=(5, 4))
        ax = fig2.add_subplot(111)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')

        colores = ['red' if t == 0 else 'green' for t in y]
        ax.scatter(X[:, 0], X[:, 1], c=colores, edgecolors='k')

        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title("Frontera de decisión")
        if self.frontera_window:
            self.frontera_window.close()
        self.frontera_window = GraficaWindow(fig2, title="Frontera de decisión")

    def entrenar(self):
        if not hasattr(self, 'df'):
            QMessageBox.warning(self, "Datos no cargados", "Primero debes cargar los datos.")
            return

        try:
            eta = float(self.eta_input.text())
            epocas = int(self.epocas_input.text())
            split = float(self.split_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Verifica que todos los parámetros sean válidos.")
            return

        x_col = self.combo1.currentText()
        y_col = self.combo2.currentText()

        errores, accuracy, scaler, perceptron = entrenar_red(
            self.df,
            feature_x=x_col,
            feature_y=y_col,
            learning_rate=eta,
            epocas=epocas,
            split=split
        )

        fig1 = Figure(figsize=(5, 4))
        ax = fig1.add_subplot(111)
        ax.plot(errores, label="Error")
        ax.set_title("Error vs Épocas")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error")
        ax.legend()
        if self.error_window:
            self.error_window.close()
        self.error_window = GraficaWindow(fig1, title="Error vs Épocas")

        self.mostrar_frontera_decision(perceptron , scaler, x_col, y_col)

        self.accuracy_label.setText(f"Accuracy: {accuracy*100:.2f}%")

    def cargar_datos(self):
        ruta = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer_data.csv")

        if not os.path.exists(ruta):
            QMessageBox.critical(self, "Error", "El archivo de datos no existe. Asegúrate de haber ejecutado carga.py.")
            cargar_y_guardar_dataset()
            QMessageBox.information(self, "Dataset", "Dataset creado. Por favor, vuelve a cargar los datos.")
            return

        self.df = pd.read_csv(ruta)
        self.columnas = list(self.df.columns[:-1])

        QMessageBox.information(self, "Datos cargados", f"Datos cargados correctamente con {len(self.df)} filas.")

        if not hasattr(self, 'combo1'):
            self.combo1 = QComboBox()
            self.combo2 = QComboBox()
            self.combo1.addItems(self.columnas)
            self.combo2.addItems(self.columnas)

            combo_layout = QGridLayout()
            combo_layout.addWidget(QLabel("Característica X:"), 0, 0)
            combo_layout.addWidget(self.combo1, 0, 1)
            combo_layout.addWidget(QLabel("Característica Y:"), 1, 0)
            combo_layout.addWidget(self.combo2, 1, 1)

            combo_box = QGroupBox("Características a usar")
            combo_box.setLayout(combo_layout)

            self.layout().insertWidget(1, combo_box)