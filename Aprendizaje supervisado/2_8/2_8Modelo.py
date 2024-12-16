import joblib
import pandas as pd

def predecir(modelo, datos_usuario):
    try:
        col = ['int.rate', 'installment', 'fico', 'revol.bal', 'revol.util', 'inq.last.6mths', 'pub.rec', 'purpose', 'credit.policy']
        df_datos = pd.DataFrame([datos_usuario], columns=col)
        prediccion = modelo.predict(df_datos)
        return prediccion
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None

def main():
    try:
        modelo = joblib.load('2_8/modelo.pkl')
    except FileNotFoundError:
        print("Error: No se puede encontrar el archivo del modelo. Asegúrate de que '2_8/modelo.pkl' exista.")
        return
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    try:
        datos = {
            'int.rate': float(input("Introduce la tasa de interés del préstamo (int_rate): ")),
            'installment': float(input("Introduce las cuotas mensuales (installment): ")),
            'fico': float(input("Introduce el puntaje FICO (fico): ")),
            'revol.bal': float(input("Introduce el saldo rotativo (revol_bal): ")),
            'revol.util': float(input("Introduce la tasa de utilización del crédito revolvente (revol_util): ")),
            'inq.last.6mths': float(input("Introduce el número de consultas de los acreedores en los últimos 6 meses (inq_last_6mths): ")),
            'pub.rec': int(input("Introduce el número de registros públicos (pub_rec): ")),
            'purpose': input("Introduce el propósito del préstamo ('debt_consolidation', 'credit_card', 'all_other', 'home_improvement', 'small_business', 'major_purchase', 'educational'): "),
            'credit.policy': float(input("Introduce 1 si el cliente cumple con los criterios de suscripción de crédito; 0 en caso contrario. (credit_policy): "))
        }
    except ValueError as ve:
        print(f"Error de entrada: {ve}. Asegúrate de ingresar los datos en el formato correcto.")
        return
    except Exception as e:
        print(f"Error al leer los datos de entrada: {e}")
        return

    resultado = predecir(modelo, datos)

    if resultado is not None:
        if resultado[0] == 0:
            print("La predicción es que el préstamo no será pagado.")
        else:
            print("La predicción es que el préstamo será pagado en su totalidad.")
    else:
        print("No se pudo realizar la predicción debido a un error.")

if __name__ == '__main__':
    main()
