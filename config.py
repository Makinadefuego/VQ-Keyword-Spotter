# config.py

AUDIO = {
    "sample_rate": 16000,
    "preemphasis_alpha": 0.97,
    "trim_db": 25,
    "spectral_subtraction": {"alpha": 2.0, "beta": 0.01}
}

FEATURES = {
    "frame_length_ms": 25,
    "frame_stride_ms": 10,
    "use_mfcc": True, "n_mfcc": 13,
    "use_delta": True, "use_delta2": True,
    "use_energy": True,
    "use_pitch": True,
}

MODEL = {
    "vq_clusters": 32,
    "gmm_components": 16,
    "garbage_label": "_garbage_",
    "rejection_threshold_vq": 10.0,
    "rejection_threshold_gmm": -100.0
}

PATHS = {
    "source_recordings": "./VOICE",
    "dataset_train": "./dataset/train",
    "dataset_test": "./dataset/test",
    "background_noises": "./background_noises",
    "output_model_vq": "./models/vq_model.joblib",
    "output_model_gmm": "./models/gmm_model.joblib"
}

DATASET = {
    'train_split_ratio': 0.8,
    'audio_extensions': ['.wav', '.m4a', '.mp3'],
    'known_words': ["abrir", "activar", "alarma", "apagar", "ayuda", "bajar", "cancelar", "cerrar", "luz", "musica", "no", "persiana", "puerta", "si", "subir"],
    'special_labels': ["_garbage_"]
}

GUI = {
    "title": "Laboratorio de Reconocimiento de Voz",
    "appearance": "dark",
    "silence_chunks_trigger": 15 # Número de chunks de silencio para detener grabación en GUI
}

LOGGING = {"verbosity_level": 1}

# --- SECCIÓN AÑADIDA / CORREGIDA ---
# Esta sección es necesaria para el script augment_dataset.py
AUGMENTATION = {
    'augmentations_per_file': 3,  # Crea 3 versiones aumentadas por cada archivo original
    'noise_probability': 0.9,     # 90% de las veces, añade ruido de fondo
    'noise_min_snr': 5,           # Relación señal/ruido mínima (más bajo = más ruidoso)
    'noise_max_snr': 20,          # Relación señal/ruido máxima
    'pitch_probability': 0.5,     # 50% de las veces, cambia el tono del audio
    'stretch_probability': 0.5    # 50% de las veces, cambia la velocidad del audio
}