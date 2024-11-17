import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocesar_age(data):
    data = pd.DataFrame(data, columns=["age"])
    data["age"] = pd.cut(data["age"], bins=[-np.inf, 16, 32, 48, 64, np.inf], labels=[1, 2, 3, 4, 5])
    return data

def name_transform_age(transformer, feature_names_in):
    return ["age"]

# Función para familia
def preprocesar_familia(data):
    df = pd.DataFrame(data, columns=["sibsp", "parch"])
    df["familia"] = df["sibsp"] + df["parch"]
    return df[["familia"]]

def name_transform_familia(transformer, feature_names_in):
    return ["familia"]


# Función para fare
def preprocesar_fare(data):
    df = pd.DataFrame(data, columns=["fare"])
    df["fare"].fillna(df["fare"].mean(), inplace=True)
    df["fare"] = np.sqrt(df["fare"])
    scaler = StandardScaler()
    df["fare"] = scaler.fit_transform(df[["fare"]])
    return df[["fare"]]

def name_transform_fare(transformer, feature_names_in):
    return ["fare"]



def main():
    # Cargar modelo previamente entrenado
    modelo = joblib.load('modelo_titanic.joblib')
    
    # Solicitar al usuario el archivo CSV
    csv_path = input("Por favor, ingresa la ruta del archivo CSV para predecir: ")
    
    # Cargar datos del CSV
    try:
        data = pd.read_csv(csv_path)
        print(f"Datos cargados correctamente: {data.shape[0]} filas y {data.shape[1]} columnas.")
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return

    # Realizar predicciones
    try:
        predictions = modelo.predict(data)
        print("Predicciones realizadas con éxito. Aquí están los resultados:")
        print(predictions)
    except Exception as e:
        print(f"Error al realizar predicciones: {e}")

if __name__ == "__main__":
    main()
