# prepare_dataset.py

import os
import shutil
import random
import re
from collections import defaultdict

# --- CONFIGURACIÓN DE LA DIVISIÓN ---
# Cambia estos valores según tus necesidades
SOURCE_FOLDER = "./VOICE" # Carpeta donde tienes todos tus audios
TRAIN_FOLDER = "./dataset/train"
TEST_FOLDER = "./dataset/test"

# Lista de palabras conocidas
KNOWN_WORDS = ["abrir", "activar", "alarma", "apagar", "ayuda", "bajar", "cancelar", "cerrar", "luz", "musica", "no", "persiana", "puerta", "si", "subir"]

# Va a se una funcion que se calcule en base a la cantidad de palabras que se encuentren pero siempre un 20% de los audios de cada palabra se irán a test
def get_train_samples_per_word(total_samples):
    return int(total_samples * 0.8)

def clean_and_prepare_folders():
    """Limpia y crea las carpetas de train y test para empezar de cero."""
    print("Limpiando directorios de dataset anteriores...")
    if os.path.exists(TRAIN_FOLDER):
        shutil.rmtree(TRAIN_FOLDER)
    if os.path.exists(TEST_FOLDER):
        shutil.rmtree(TEST_FOLDER)
    
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    print("Directorios de train y test creados.")

def split_dataset():
    """
    Divide los audios de la carpeta de origen en subcarpetas de train y test.
    Asume que los nombres de archivo son como 'palabraF1.wav', 'palabraF2.wav', etc.
    """
    if not os.path.exists(SOURCE_FOLDER):
        print(f"[ERROR] La carpeta de origen '{SOURCE_FOLDER}' no existe.")
        print("Por favor, crea esta carpeta y pon todos tus archivos .wav dentro.")
        return

   # 1. Agrupar archivos por palabra
    files_by_word = defaultdict(list)
    for filename in os.listdir(SOURCE_FOLDER):
        if filename.lower().endswith('.m4a') or filename.lower().endswith('.wav'):
            # Buscar si el archivo comienza con alguna palabra conocida
            word_found = None
            filename_lower = filename.lower()
            
            for word in KNOWN_WORDS:
                if filename_lower.startswith(word.lower()):
                    word_found = word.lower()
                    break
            
            if word_found:
                files_by_word[word_found].append(filename)
            else:
                print(f"[Advertencia] El archivo '{filename}' no comienza con ninguna palabra conocida y será omitido.")


    if not files_by_word:
        print(f"[ERROR] No se encontraron archivos .wav con el formato esperado en '{SOURCE_FOLDER}'.")
        print("Asegúrate de que los nombres sean como 'cancelarF1.wav'.")
        return

    print(f"\nSe encontraron {len(files_by_word)} palabras únicas: {', '.join(sorted(files_by_word.keys()))}")

    # 2. Procesar cada palabra
    for word, files in files_by_word.items():
        print(f"\nProcesando palabra: '{word}' (encontrados {len(files)} archivos)")
        
        # Verificar si hay suficientes archivos para dividir
        TRAIN_SAMPLES_PER_WORD = get_train_samples_per_word(len(files))


        if len(files) < TRAIN_SAMPLES_PER_WORD:
            print(f"  [Advertencia] La palabra '{word}' tiene menos archivos que los requeridos para entrenamiento. Se omitirá.")
            continue
        
        # Mezclar aleatoriamente los archivos para una división justa
        random.shuffle(files)
        
        # Dividir los archivos
        train_files = files[:TRAIN_SAMPLES_PER_WORD]
        test_files = files[TRAIN_SAMPLES_PER_WORD:]
        
        print(f"  -> División: {len(train_files)} para train, {len(test_files)} para test.")
        
        # 3. Crear carpetas y copiar archivos
        
        # Para entrenamiento
        word_train_dir = os.path.join(TRAIN_FOLDER, word)
        os.makedirs(word_train_dir, exist_ok=True)
        for f in train_files:
            shutil.copy(os.path.join(SOURCE_FOLDER, f), word_train_dir)
            
        # Para prueba (solo si hay archivos de prueba)
        if test_files:
            word_test_dir = os.path.join(TEST_FOLDER, word)
            os.makedirs(word_test_dir, exist_ok=True)
            for f in test_files:
                shutil.copy(os.path.join(SOURCE_FOLDER, f), word_test_dir)
    
    print("\n¡Proceso de división completado! El dataset está listo.")

if __name__ == '__main__':
    clean_and_prepare_folders()
    split_dataset()