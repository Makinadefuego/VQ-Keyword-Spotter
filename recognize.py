# recognize.py

import sys
import os
import time

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier
from gmm_classifier import GMMClassifier

def main():
    """
    Función principal para reconocer una palabra de un único archivo de audio.
    """
    # 1. Verificar los argumentos de la línea de comandos
    if len(sys.argv) != 2:
        print("Uso: python recognize.py <ruta_al_archivo_de_audio.wav>")
        return

    audio_path = sys.argv[1]

    # 2. Verificar que el archivo de audio exista
    if not os.path.exists(audio_path):
        print(f"[ERROR] El archivo de audio '{audio_path}' no fue encontrado.")
        return

    # 3. Determinar qué modelo usar desde la configuración
    model_type = config.MODEL.get("model_type", "gmm") # Default a gmm si no está especificado
    
    if model_type == 'vq':
        model_path = config.PATHS['output_model_vq']
        Classifier = VQClassifier
    elif model_type == 'gmm':
        model_path = config.PATHS['output_model']
        Classifier = GMMClassifier
    else:
        print(f"[ERROR] Tipo de modelo '{model_type}' no soportado. Opciones: 'vq', 'gmm'.")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] El modelo entrenado '{model_path}' no fue encontrado para el tipo '{model_type}'.")
        print("Por favor, ejecuta 'train.py' primero para entrenar y guardar el modelo correcto.")
        return

    # 4. Cargar el modelo entrenado
    try:
        print(f"Cargando modelo {model_type.upper()} desde: {model_path}")
        classifier = Classifier.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    # 5. Realizar la predicción y medir el tiempo
    print(f"\nReconociendo palabra en: {os.path.basename(audio_path)}")
    start_time = time.time()
    
    # La forma de llamar a predict es diferente para cada modelo.
    if model_type == 'vq':
         # VQ necesita parámetros de procesamiento específicos.
        processing_params = {
            'use_subtraction': config.AUDIO.get('use_noise_reduction', False),
            'use_preemphasis': True,
            'use_trim': True,
        }
        predicted_label = classifier.predict(audio_path, processing_params=processing_params)
    else: # GMM
        # GMM también los necesita.
        processing_params = {
            'use_subtraction': False, # GMM se entrena sin reducción de ruido
            'use_preemphasis': True,
            'use_trim': True,
            'rejection_threshold': config.MODEL.get('rejection_threshold_gmm')
        }
        predicted_label = classifier.predict(audio_path, processing_params=processing_params)

    end_time = time.time()
    prediction_time = end_time - start_time

    # 6. Mostrar el resultado
    if predicted_label:
        print(f"\n--- Resultado ---")
        print(f"Palabra reconocida: '{predicted_label}'")
        print(f"Tiempo de predicción: {prediction_time:.4f} segundos.")
    else:
        print("\nNo se pudo obtener una predicción.")


if __name__ == '__main__':
    main()