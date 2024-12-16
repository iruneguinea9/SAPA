import joblib
import pandas as pd

def predecir(modelo, datos_usuario):
    try:
        col = ["Volume", "Weight", "Car", "Model"]
        df_datos = pd.DataFrame([datos_usuario], columns=col)
        prediccion = modelo.predict(df_datos)
        return prediccion
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None

def main():
    try:
        # Cargar el modelo entrenado
        modelo = joblib.load('2_11/modelo_emisiones.pkl')
    except FileNotFoundError:
        print("Error: No se puede encontrar el archivo del modelo. Asegúrate de que 'modelo_emisiones.pkl' exista.")
        return
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    try:
        # Solicitar datos de entrada al usuario
        volumen = float(input("Introduce el volumen (Volume): "))
        peso = float(input("Introduce el peso (Weight): "))
        coche = input("Introduce la marca del coche (Car): ")
        modelo_coche = input("Introduce el modelo del coche (Model): ")

        datos = {
            "Volume": volumen,
            "Weight": peso,
            "Car": coche,
            "Model": modelo_coche
        }
    except ValueError as ve:
        print(f"Error de entrada: {ve}. Asegúrate de ingresar los datos en el formato correcto.")
        return
    except Exception as e:
        print(f"Error al leer los datos de entrada: {e}")
        return

    resultado = predecir(modelo, datos)

    if resultado is not None:
        print(f"La predicción de emisiones de CO2 es: {resultado[0]:.2f} gCO2/km")
    else:
        print("No se pudo realizar la predicción debido a un error.")

if __name__ == '__main__':
    main()
