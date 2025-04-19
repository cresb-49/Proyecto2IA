import sys
from PyQt5.QtWidgets import QApplication
from gui import RedNeuronalApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RedNeuronalApp()
    window.show()
    sys.exit(app.exec_())