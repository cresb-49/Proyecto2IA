from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QGridLayout, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Dummy para probar la interfaz
def dummy_train(eta, epochs, train_split):
    print(f"Entrenando con η={eta}, épocas={epochs}, entrenamiento={train_split}%")
    return [0.9, 0.7, 0.5, 0.3, 0.2]  # Ejemplo de error

class RedNeuronalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificador con Red Neuronal")
        self.setMinimumSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Grupo de parámetros
        param_group = QGroupBox("Parámetros de Entrenamiento")
        grid = QGridLayout()

        self.eta_input = QLineEdit("0.01")
        self.epochs_input = QLineEdit("100")
        self.split_input = QLineEdit("80")

        grid.addWidget(QLabel("Tasa de aprendizaje (η):"), 0, 0)
        grid.addWidget(self.eta_input, 0, 1)

        grid.addWidget(QLabel("Épocas máximas:"), 1, 0)
        grid.addWidget(self.epochs_input, 1, 1)

        grid.addWidget(QLabel("Porcentaje de entrenamiento:"), 2, 0)
        grid.addWidget(self.split_input, 2, 1)

        self.train_btn = QPushButton("Entrenar")
        self.train_btn.clicked.connect(self.entrenar)
        grid.addWidget(self.train_btn, 3, 0, 1, 2)

        param_group.setLayout(grid)
        layout.addWidget(param_group)

        # Gráfico
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def entrenar(self):
        try:
            eta = float(self.eta_input.text())
            epochs = int(self.epochs_input.text())
            split = float(self.split_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Verifica que todos los parámetros sean válidos.")
            return

        errores = dummy_train(eta, epochs, split)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(errores, label="Error")
        ax.set_title("Error vs Épocas")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error")
        ax.legend()
        self.canvas.draw()