# augment_dataset.py

import os
from tqdm import tqdm
import soundfile as sf
import librosa
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch

import config

def main():
    """Aplica aumento de datos al dataset de entrenamiento."""
    SOURCE_TRAIN_FOLDER = config.PATHS['dataset_train']
    NOISE_FOLDER = config.PATHS['background_noises']
    AUG_CONFIG = config.AUGMENTATION
    SAMPLE_RATE = config.AUDIO['sample_rate']

    print("--- Inicio del Proceso de Aumento de Datos ---")
    
    if not os.path.exists(NOISE_FOLDER) or not os.listdir(NOISE_FOLDER):
        print(f"[ADVERTENCIA] La carpeta de ruidos '{NOISE_FOLDER}' está vacía. El aumento se hará sin añadir ruido de fondo.")
        noise_p = 0.0
    else:
        noise_p = AUG_CONFIG['noise_probability']
    
    augmenter = Compose([
        # AddBackgroundNoise(sounds_path=NOISE_FOLDER, 
        PitchShift(min_semitones=-4, max_semitones=4, p=AUG_CONFIG['pitch_probability']),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=AUG_CONFIG['stretch_probability']),
    ])

    # Limpiar aumentos anteriores
    print("Buscando y eliminando aumentos anteriores...")
    for dirpath, _, filenames in os.walk(SOURCE_TRAIN_FOLDER):
        for filename in filenames:
            if "_aug_" in filename:
                os.remove(os.path.join(dirpath, filename))

    # Recopilar archivos originales
    files_to_augment = []
    for dirpath, _, filenames in os.walk(SOURCE_TRAIN_FOLDER):
        for filename in filenames:
            if filename.lower().endswith(('.m4a', '.wav')) and "_aug_" not in filename:
                files_to_augment.append(os.path.join(dirpath, filename))
    
    if not files_to_augment:
        print("[ERROR] No se encontraron archivos originales para aumentar.")
        return

    print(f"Aumentando {len(files_to_augment)} archivos originales...")
    for filepath in tqdm(files_to_augment, desc="Aumentando archivos"):
        try:
            signal, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            for i in range(AUG_CONFIG['augmentations_per_file']):
                augmented_signal = augmenter(samples=signal, sample_rate=SAMPLE_RATE)
                base, _ = os.path.splitext(filepath)
                output_path = f"{base}_aug_{i+1}.wav"
                sf.write(output_path, augmented_signal, SAMPLE_RATE)
        except Exception as e:
            print(f"\n[ADVERTENCIA] No se pudo aumentar {os.path.basename(filepath)}: {e}")
            
    print("\n--- Proceso de Aumento de Datos Finalizado ---")

if __name__ == '__main__':
    main()