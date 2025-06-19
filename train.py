# train.py

import os
import time
from tqdm import tqdm
import numpy as np

import config
import feature_extractor
from vq_classifier import VQClassifier

def main():
    """Orquesta el entrenamiento del modelo VQ."""
    train_path = config.PATHS['dataset_train']
    if not os.path.exists(train_path) or not os.listdir(train_path):
        print(f"\n[ERROR] El directorio de entrenamiento '{train_path}' no existe o está vacío.")
        print("Ejecuta 'prepare_dataset.py' y 'augment_dataset.py' primero.")
        return

    classifier = VQClassifier(n_clusters=config.MODEL['vq_clusters'])
    
    print("--- Inicio del Proceso de Entrenamiento del Clasificador VQ ---")
    start_time = time.time()
    
    try:
        classifier.train(training_path=train_path)
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error durante el entrenamiento: {e}")
        return
        
    training_time = time.time() - start_time
    print(f"\n--- Entrenamiento Finalizado en {training_time:.2f} segundos. ---")

    if not classifier.codebooks:
        print("\n[ADVERTENCIA] El entrenamiento finalizó, pero no se crearon codebooks.")
        return
        
    try:
        classifier.save_model(path=config.PATHS['output_model'])
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el modelo: {e}")

if __name__ == '__main__':
    main()