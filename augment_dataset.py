# augment_dataset.py

import os
from tqdm import tqdm
import soundfile as sf
import librosa
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch

import config

def main():
    """Aplica aumento de datos al dataset de entrenamiento."""
    # Leemos las rutas y configuraciones desde el archivo central
    train_folder = config.PATHS['dataset_train']
    noise_folder = config.PATHS['background_noises']
    aug_config = config.AUGMENTATION
    sample_rate = config.AUDIO['sample_rate']
    audio_extensions = tuple(config.DATASET['audio_extensions'])

    print("--- Inicio del Proceso de Aumento de Datos ---")
    
    if not os.path.exists(train_folder):
        print(f"[ERROR] La carpeta de entrenamiento '{train_folder}' no existe. Ejecuta 'prepare_dataset.py' primero.")
        return

    # --- Configuración del Aumentador ---
    # Comprobamos si hay ruidos de fondo para añadir
    if os.path.exists(noise_folder) and os.listdir(noise_folder):
        print("Ruido de fondo encontrado. Se activará la transformación 'AddBackgroundNoise'.")
        augmenter = Compose([
            AddBackgroundNoise(
                sounds_path=noise_folder, 
                min_snr_in_db=aug_config['noise_min_snr'],
                max_snr_in_db=aug_config['noise_max_snr'],
                p=aug_config['noise_probability']
            ),
            PitchShift(min_semitones=-4, max_semitones=4, p=aug_config['pitch_probability']),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=aug_config['stretch_probability']),
        ])
    else:
        print(f"[ADVERTENCIA] La carpeta de ruidos '{noise_folder}' está vacía o no existe. El aumento se hará sin añadir ruido de fondo.")
        augmenter = Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=aug_config['pitch_probability']),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=aug_config['stretch_probability']),
        ])

    # 1. Limpiar aumentos anteriores para no acumular archivos
    print("Buscando y eliminando aumentos anteriores...")
    files_deleted = 0
    for dirpath, _, filenames in os.walk(train_folder):
        for filename in filenames:
            if "_aug_" in filename:
                os.remove(os.path.join(dirpath, filename))
                files_deleted += 1
    if files_deleted > 0:
        print(f"Se eliminaron {files_deleted} archivos aumentados previos.")

    # 2. Recopilar archivos originales del set de entrenamiento
    files_to_augment = []
    for dirpath, _, filenames in os.walk(train_folder):
        for filename in filenames:
            # Solo aumentamos los archivos originales que no contienen '_aug_'
            if filename.lower().endswith(audio_extensions) and "_aug_" not in filename:
                files_to_augment.append(os.path.join(dirpath, filename))
    
    if not files_to_augment:
        print("[ERROR] No se encontraron archivos originales para aumentar en el set de entrenamiento.")
        return

    # 3. Aplicar el aumento
    print(f"\nAumentando {len(files_to_augment)} archivos originales... (Generando {aug_config['augmentations_per_file']} versiones por archivo)")
    for filepath in tqdm(files_to_augment, desc="Aumentando archivos"):
        try:
            signal, _ = librosa.load(filepath, sr=sample_rate)
            for i in range(aug_config['augmentations_per_file']):
                augmented_signal = augmenter(samples=signal, sample_rate=sample_rate)
                
                base, _ = os.path.splitext(filepath)
                # Guardamos siempre como .wav para consistencia
                output_path = f"{base}_aug_{i+1}.wav"
                
                sf.write(output_path, augmented_signal, sample_rate)
        except Exception as e:
            print(f"\n[ADVERTENCIA] No se pudo aumentar {os.path.basename(filepath)}: {e}")
            
    print("\n--- Proceso de Aumento de Datos Finalizado ---")

if __name__ == '__main__':
    main()