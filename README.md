
# 🧠 Clasificación de Tumores Mamarios con Perceptrón Simple

Este proyecto implementa una aplicación interactiva para la clasificación de tumores mamarios como **malignos (0)** o **benignos (1)**, utilizando un modelo de **perceptrón simple** desarrollado desde cero en Python. La aplicación permite configurar los parámetros del modelo, entrenarlo con el conjunto de datos **Breast Cancer Wisconsin**, visualizar las gráficas del entrenamiento y probar manualmente nuevas entradas para su clasificación.

## 🚀 Características Principales

- Selección de dos características del dataset para el entrenamiento.
- Configuración de la tasa de aprendizaje, número de épocas y porcentaje de datos de entrenamiento.
- Entrenamiento del modelo con visualización de:
  - Gráfico de **Error vs Épocas**.
  - **Frontera de Decisión** generada por el modelo.
- Prueba manual con datos introducidos por el usuario.
- Opción para consultar los últimos gráficos generados sin repetir el entrenamiento.

---

## 🗂️ Estructura del Proyecto

```
Proyecto2IA/
│
├── data/                     # Dataset en formato CSV
├── docs/                     # Documentación (manual técnico y manual de usuario)
├── src/                      # Código fuente del proyecto
│   ├── carga.py              # Carga y guardado del dataset
│   ├── entreno.py            # Entrenamiento del perceptrón simple
│   ├── gui.py                # Interfaz gráfica desarrollada con PyQt5
│   ├── main.py               # Punto de entrada de la aplicación
│   └── perceptron.py         # Implementación del modelo de perceptrón simple
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
└── .gitignore                # Archivos a ignorar por Git
```

---

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone <URL-del-repositorio>
```

2. Crear el entorno virtual:
```bash
python -m venv venv
```

3. Activar el entorno virtual:
- En Windows:
  ```bash
  venv\Scripts\activate
  ```
- En Linux/Mac:
  ```bash
  source venv/bin/activate
  ```

4. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución de la Aplicación

Ubicarse en la carpeta `src` y ejecutar:

```bash
python main.py
```

La aplicación abrirá la interfaz gráfica, desde donde se podrá configurar el entrenamiento y visualizar los resultados.

---

## 📂 Dataset Utilizado

- **Breast Cancer Wisconsin Dataset**  
- Fuente: Incluido en `scikit-learn.datasets`  
- Contiene características morfológicas de núcleos celulares y la clasificación de cada muestra como maligna (0) o benigna (1).

---

## 📊 Visualización de Resultados

- **Error vs Épocas:** Permite observar la evolución del error durante el proceso de aprendizaje.
- **Frontera de Decisión:** Representa gráficamente cómo el modelo separa las dos clases en el plano definido por las características seleccionadas.

---

## 👨‍💻 Autor

- **Carlos Pac**  
- **Carné:** 201931012

---

## 📜 Licencia

Este proyecto ha sido desarrollado con fines educativos. Su uso y modificación es libre, respetando el propósito original de aprendizaje y demostración del funcionamiento del perceptrón simple.
