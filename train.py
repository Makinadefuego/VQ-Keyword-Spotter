# train.py

import os
import time
import numpy as np
from tqdm import tqdm
import traceback

import config
import feature_extractor
from vq_classifier import VQClassifier
try:
    from gmm_classifier import GMMClassifier
except ImportError:
    GMMClassifier = None

def train_model(ClassifierClass, model_type, training_path, feature_config):
    print("\n" + "="*50)
    print(f"--- INICIANDO ENTRENAMIENTO PARA MODELO: {model_type.upper()} ---")
    print("="*50)

    if model_type == 'gmm' and GMMClassifier is None:
        print("[ERROR] gmm_classifier.py no encontrado. Saltando entrenamiento de GMM.")
        return

    classifier = ClassifierClass()
    
    # 1. Recolectar todas las características de todos los audios
    all_features_map = {}
    labels = sorted([name for name in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, name))])
    
    train_params = {'use_subtraction': False, 'use_preemphasis': True, 'use_trim': True}

    for label in tqdm(labels, desc=f"Extrayendo Features para {model_type.upper()}"):
        word_dir = os.path.join(training_path, label)
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(tuple(config.DATASET['audio_extensions']))]
        
        label_features = []
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            try:
                signal, sr = feature_extractor.load_and_clean_audio(filepath, params=train_params)
                if len(signal) > 0:
                    features = feature_extractor.extract_features(signal, sr)
                    if features.shape[0] > 0:
                        label_features.append(features)
            except Exception as e:
                print(f"\n[WARN] Error procesando {filename}: {e}")
        
        if label_features:
            all_features_map[label] = np.vstack(label_features)

    # 2. Entrenar el modelo específico para cada palabra
    try:
        classifier.train(all_features_map)
    except Exception as e:
        print(f"\n[FATAL] Ocurrió un error crítico durante el entrenamiento de {model_type.upper()}:")
        traceback.print_exc()
        return

    # 3. Guardar el modelo
    output_path = config.PATHS.get(f"output_model_{model_type}")
    if output_path:
        classifier.save_model(output_path)
    else:
        print(f"[WARN] No se encontró ruta de salida para el modelo {model_type.upper()} en config.py")

def main():
    train_path = config.PATHS['dataset_train']
    if not os.path.exists(train_path) or not os.listdir(train_path):
        print(f"[ERROR] Directorio de entrenamiento '{train_path}' no existe o está vacío.")
        return

    start_time = time.time()
    
    # Entrenar ambos modelos
    train_model(VQClassifier, 'vq', train_path, config.FEATURES)
    train_model(GMMClassifier, 'gmm', train_path, config.FEATURES)

    total_time = time.time() - start_time
    print(f"\nProceso de entrenamiento completado en {total_time:.2f} segundos.")

if __name__ == '__main__':
    main()