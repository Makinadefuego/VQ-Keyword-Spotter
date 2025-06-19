# config.py

# -----------------------------------------------------------------------------
# -- CONFIGURACIÓN DE PROCESAMIENTO DE AUDIO --
# -----------------------------------------------------------------------------
AUDIO = {
    "sample_rate": 16000,
    "preemphasis_alpha": 0.97,
    # Umbral en decibelios (dB) para librosa.effects.trim.
    # Un valor más alto (ej. 30) es más estricto y corta más.
    # Un valor más bajo (ej. 20) es más permisivo.
    "trim_db": 25, 
    # Activa o desactiva la supresión de ruido con la librería noisereduce.
    "use_noise_reduction": True,

    # Parámetros para noisereduce. 'prop_decrease' controla la agresividad.
    # Un valor de 1.0 es el estándar. Más bajo es menos agresivo.
    "noise_reduce_prop_decrease": 1.0
}
# -- CONFIGURACIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS --
# -----------------------------------------------------------------------------
FEATURES = {
    "frame_length_ms": 25,
    "frame_stride_ms": 10,
    "n_mfcc": 13,
    "use_mfcc": True,
    "use_delta": True,
    "use_delta2": True,
    "use_energy": True,
    "use_pitch": True,
    "use_spectral_features": True,
    "n_gfcc": 13,
    "use_gfcc": True,
    "use_advanced_spectral_features": True,
}

# -----------------------------------------------------------------------------
# -- CONFIGURACIÓN DEL MODELO Y RUTAS --
# -----------------------------------------------------------------------------
MODEL = {
    "vq_clusters": 64,
    "garbage_label": "_garbage_"
}

PATHS = {
    "source_recordings": "./VOICE",
    "dataset_train": "./dataset/train",
    "dataset_test": "./dataset/test",
    "background_noises": "./background_noises",
    "output_model": "./models/vq_robust_model.joblib"
}

# -----------------------------------------------------------------------------
# -- CONFIGURACIÓN DE AUMENTO DE DATOS --
# -----------------------------------------------------------------------------
AUGMENTATION = {
    "augmentations_per_file": 10,
    "noise_probability": 0.8,
    "noise_min_snr": 5.0,
    "noise_max_snr": 25.0,
    "pitch_probability": 0.5,
    "stretch_probability": 0.5,
}

# -----------------------------------------------------------------------------
# -- CONFIGURACIÓN DE LA GUI Y LOGS --
# -----------------------------------------------------------------------------
GUI = {
    "title": "Asistente de Voz VQ",
    "window_size": "400x500",
    "appearance": "dark",
    # El VAD ya no se usa, pero la lógica de la GUI se basa en energía.
    # Mantenemos los triggers de silencio.
    "silence_chunks_trigger": 15,
}

LOGGING = {
    "verbosity_level": 2, # 0: Mínimo, 1: Normal, 2: Detallado
}