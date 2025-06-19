# recognize.py

import sys
import os
import time

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier

def main():
    """
    Función principal para reconocer una palabra de un único archivo de audio.
    """
    # 1. Verificar los argumentos de la línea de comandos
    if len(sys.argv) != 2:
        print("Uso: python recognize.py <ruta_al_archivo_de_audio.wav>")
        return

    audio_path = sys.argv[1]

    # 2. Verificar que el archivo de audio y el modelo existan
    if not os.path.exists(audio_path):
        print(f"[ERROR] El archivo de audio '{audio_path}' no fue encontrado.")
        return

    model_path = config.PATHS['output_model']
    if not os.path.exists(model_path):
        print(f"[ERROR] El modelo entrenado '{model_path}' no fue encontrado.")
        print("Por favor, ejecuta 'train.py' primero para entrenar y guardar un modelo.")
        return

    # 3. Cargar el modelo entrenado
    try:
        classifier = VQClassifier.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    # 4. Realizar la predicción y medir el tiempo
    print(f"\nReconociendo palabra en: {os.path.basename(audio_path)}")
    start_time = time.time()
    
    predicted_label = classifier.predict(audio_path)
    
    end_time = time.time()
    prediction_time = end_time - start_time

    # 5. Mostrar el resultado
    if predicted_label:
        print(f"\n--- Resultado ---")
        print(f"Palabra reconocida: '{predicted_label}'")
        print(f"Tiempo de predicción: {prediction_time:.4f} segundos.")
    else:
        print("\nNo se pudo obtener una predicción.")


if __name__ == '__main__':
    main()