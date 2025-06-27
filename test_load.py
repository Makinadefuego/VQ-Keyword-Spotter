# test_predict.py

import os
import random
import traceback

import config
from vq_classifier import VQClassifier
from gmm_classifier import GMMClassifier

def find_random_test_file():
    """Encuentra un archivo de audio aleatorio en la carpeta de prueba."""
    test_path = config.PATHS['dataset_test']
    if not os.path.exists(test_path):
        print(f"[ERROR] La carpeta de prueba '{test_path}' no existe. Ejecuta 'prepare_dataset.py'.")
        return None, None

    # Obtiene todas las carpetas de palabras en el directorio de prueba
    word_folders = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    if not word_folders:
        print(f"[ERROR] No se encontraron carpetas de palabras en '{test_path}'.")
        return None, None
        
    # Elige una palabra al azar
    random_word = random.choice(word_folders)
    word_dir = os.path.join(test_path, random_word)
    
    # Elige un archivo de audio al azar de esa carpeta
    audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(tuple(config.DATASET['audio_extensions']))]
    if not audio_files:
        print(f"[ERROR] La carpeta '{word_dir}' no contiene archivos de audio.")
        return None, None
        
    random_file = random.choice(audio_files)
    file_path = os.path.join(word_dir, random_file)
    
    return file_path, random_word


def run_test(model_type):
    """Ejecuta una prueba de predicción para un tipo de modelo específico."""
    print("\n" + "="*60)
    print(f"--- INICIANDO PRUEBA PARA EL MODELO: {model_type.upper()} ---")
    print("="*60)

    # 1. Seleccionar clasificador y ruta del modelo
    if model_type == 'vq':
        Classifier = VQClassifier
        model_path = config.PATHS.get('output_model_vq')
    elif model_type == 'gmm':
        Classifier = GMMClassifier
        model_path = config.PATHS.get('output_model_gmm')
    else:
        print(f"[ERROR] Tipo de modelo desconocido: '{model_type}'")
        return

    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Archivo de modelo no encontrado en '{model_path}'. ¿Ejecutaste 'train.py'?")
        return

    # 2. Cargar modelo
    try:
        print(f"Cargando modelo desde: {model_path}")
        classifier = Classifier.load_model(model_path)
        print("Modelo cargado exitosamente.")
        print(f"Palabras en el modelo: {classifier.labels}")
    except Exception:
        print("\n[FALLO CRÍTICO] Ocurrió un error al cargar el modelo:")
        traceback.print_exc()
        return

    # 3. Encontrar un archivo de prueba
    test_file, true_label = find_random_test_file()
    if not test_file:
        return

    print(f"\nArchivo de prueba seleccionado: {os.path.basename(test_file)}")
    print(f"Etiqueta verdadera: '{true_label}'")
    print("\n--- INICIO DE LA EXTRACCIÓN DE CARACTERÍSTICAS Y PREDICCIÓN ---")

    # 4. Realizar la predicción
    try:
        # Definimos los parámetros de procesamiento que usaría la GUI (sin sustracción de ruido)
        processing_params = {
            'use_subtraction': False,
            'noise_profile': None,
            'use_preemphasis': True,
            'use_trim': True
        }

        # Establecemos el nivel de verbosidad al máximo para ver todos los logs
        config.LOGGING['verbosity_level'] = 2
        
        predicted_label = classifier.predict(test_file, processing_params=processing_params)
        
        print("\n--- FIN DE LA EXTRACCIÓN DE CARACTERÍSTICAS Y PREDICCIÓN ---")

        # 5. Mostrar resultados
        print("\n" + "-"*60)
        print("RESULTADO DE LA PREDICCIÓN:")
        if predicted_label is not None:
            print(f"  -> Predicción: '{predicted_label}'")
            if predicted_label == true_label:
                print("  -> Resultado: ¡CORRECTO! 😄")
            else:
                print("  -> Resultado: INCORRECTO. 😞")
        else:
            print("  -> Predicción: NINGUNA (None). El proceso de características probablemente descartó el audio.")
        print("-"*60)
        
    except Exception:
        print("\n[FALLO CRÍTICO] Ocurrió un error no controlado durante la predicción:")
        traceback.print_exc()

if __name__ == '__main__':
    # Asegúrate de que los modelos existan
    if not os.path.exists(config.PATHS['output_model_vq']) or \
       not os.path.exists(config.PATHS['output_model_gmm']):
        print("Modelos no encontrados. Por favor, ejecuta 'python3 train.py' primero.")
    else:
        # Ejecuta la prueba para ambos modelos
        run_test('vq')
        run_test('gmm')