import joblib
import pandas as pd

def predecir(modelo, datos_usuario):
    try:
        col = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        df_datos = pd.DataFrame([datos_usuario], columns=col)
        prediccion = modelo.predict(df_datos)
        return prediccion
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None

def main():
    try:
        # Cargar el modelo entrenado
        modelo = joblib.load('Examen/modelo_flores.pkl')
    except FileNotFoundError:
        print("Error: No se puede encontrar el archivo del modelo. Asegúrate de que 'modelo_flores.pkl' exista.")
        return
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    try:
        # Solicitar datos de entrada al usuario
        petal_width = float(input("Introduce la anchura del pétalo (cm): "))
        petal_length = float(input("Introduce la longitud del pétalo (cm): "))
        sepal_width = input("Introduce la anchura del sépalo (cm): ")
        sepal_length = input("Introduce la longitud del sépalo (cm): ")

        datos = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width,         

            
        }
    except ValueError as ve:
        print(f"Error de entrada: {ve}. Asegúrate de ingresar los datos en el formato correcto.")
        return
    except Exception as e:
        print(f"Error al leer los datos de entrada: {e}")
        return

    resultado = predecir(modelo, datos)

    if resultado is not None:
        # Seguro que hay una manera mejor, pero es lo que sé
        if(resultado[0]==0):
            print(f"La predicción de tipo de flor es setosa ")
        if(resultado[0]==1):
            print(f"La predicción de tipo de flor es versicolor ")
        else:
            print(f"La predicción de tipo de flor es virginica ")
    else:
        print("No se pudo realizar la predicción debido a un error.")

if __name__ == '__main__':
    main()
