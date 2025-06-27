# prepare_dataset.py

import os
import shutil
import random
from collections import defaultdict
import math

# Importamos la configuración central
import config

def clean_and_prepare_folders():
    """Limpia y crea las carpetas de train y test para empezar de cero."""
    train_folder = config.PATHS['dataset_train']
    test_folder = config.PATHS['dataset_test']
    
    print("Limpiando directorios de dataset anteriores...")
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    print("Directorios de train y test creados.")

def split_dataset():
    """
    Escanea la carpeta de grabaciones, identifica archivos basándose en una lista
    de palabras conocidas y los divide en conjuntos de entrenamiento y prueba.
    """
    source_folder = config.PATHS['source_recordings']
    train_folder = config.PATHS['dataset_train']
    test_folder = config.PATHS['dataset_test']
    split_ratio = config.DATASET['train_split_ratio']
    audio_extensions = tuple(config.DATASET['audio_extensions'])
    known_words = config.DATASET.get('known_words', []) + config.DATASET.get('special_labels', [])

    if not os.path.exists(source_folder):
        print(f"[ERROR] La carpeta de origen '{source_folder}' no existe.")
        return

    if not known_words:
        print("[ERROR] No se han definido 'known_words' en config.py. El script no sabe qué buscar.")
        return

    # 1. Escanear y agrupar archivos por palabra clave conocida
    files_by_word = defaultdict(list)
    print(f"Escaneando '{source_folder}' en busca de palabras clave conocidas...")
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(audio_extensions):
            file_lower = filename.lower()
            word_found = None
            # Iteramos sobre nuestra lista de palabras conocidas
            for word in known_words:
                if file_lower.startswith(word):
                    word_found = word
                    break # Encontramos la palabra, pasamos al siguiente archivo
            
            if word_found:
                files_by_word[word_found].append(filename)
            else:
                print(f"[WARN] El archivo '{filename}' no comienza con ninguna palabra conocida y será omitido.")

    if not files_by_word:
        print(f"[ERROR] No se encontraron archivos que comiencen con las palabras clave definidas en '{source_folder}'.")
        return

    print(f"\nSe encontraron archivos para {len(files_by_word)} palabras: {', '.join(sorted(files_by_word.keys()))}")

    # 2. Procesar cada palabra y dividirla
    total_train_files = 0
    total_test_files = 0

    for word, files in files_by_word.items():
        print(f"\nProcesando palabra: '{word}' (encontrados {len(files)} archivos)")
        
        if len(files) < 2:
            print(f"  [WARN] La palabra '{word}' tiene menos de 2 muestras. Se necesita al menos una para train y una para test. Se omitirá.")
            continue
            
        random.shuffle(files)
        
        num_train = math.ceil(len(files) * split_ratio)
        if num_train == len(files):
            num_train -= 1
            
        train_files = files[:num_train]
        test_files = files[num_train:]
        
        print(f"  -> División ({split_ratio*100:.0f}/{100-split_ratio*100:.0f}): {len(train_files)} para train, {len(test_files)} para test.")
        total_train_files += len(train_files)
        total_test_files += len(test_files)
        
        # 3. Crear carpetas y copiar archivos
        word_train_dir = os.path.join(train_folder, word)
        os.makedirs(word_train_dir, exist_ok=True)
        for f in train_files:
            shutil.copy(os.path.join(source_folder, f), word_train_dir)
            
        word_test_dir = os.path.join(test_folder, word)
        os.makedirs(word_test_dir, exist_ok=True)
        for f in test_files:
            shutil.copy(os.path.join(source_folder, f), word_test_dir)
    
    print("\n" + "="*50)
    print("¡Proceso de división completado!")
    print(f"Total de archivos de entrenamiento: {total_train_files}")
    print(f"Total de archivos de prueba: {total_test_files}")
    print("="*50)

if __name__ == '__main__':
    clean_and_prepare_folders()
    split_dataset()