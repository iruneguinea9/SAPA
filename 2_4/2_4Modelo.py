import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def familia_name(function_transformer, feature_names_in):
     return ['familia']
# juntar parch y sibsp para familia
def crearFamilia(X):
     X=pd.DataFrame(X,columns=['parch','sibsp'])
     X['familia'] = X['sibsp'] + X['parch'] 
     return X['familia'].values.reshape(-1,1)

def age_name(function_transformer, feature_names_in):
     return ['age']
def separar_age(X):
     X=pd.DataFrame(X,columns=['age'])
     X['age'] = pd.cut(X['age'], bins=[-1,16,32,48,64,np.inf], labels=[1,2,3,4,5]).to_numpy().reshape(-1,1)  
     return X

def formatearSex(function_transformer, feature_names_in):
     return ['sex']
#male 0 female 1
def sexNumeros(X):
     return np.where(X == 'female',1,0)

def main():
    # Cargar modelo previamente entrenado
    modelo = joblib.load('2_4/modelo.pkl')
    
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
