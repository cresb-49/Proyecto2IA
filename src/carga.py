import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def cargar_y_guardar_dataset():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    columnas = list(data.feature_names)

    df = pd.DataFrame(X, columns=columnas)
    df['target'] = y 

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "breast_cancer_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset guardado en: {output_path}")

