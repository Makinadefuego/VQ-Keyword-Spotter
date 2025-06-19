# feature_extractor.py

import numpy as np
import librosa
from scipy.signal import lfilter
import os

import config
import audio_utils

# Obtenemos el nivel de verbosidad desde la configuración
VERBOSITY = config.LOGGING.get("verbosity_level", 1)

def load_and_clean_audio(audio_path):
    """
    Carga un archivo de audio, lo limpia y pre-procesa usando librosa.effects.trim.
    """
    if VERBOSITY >= 2:
        print(f"  [FEAT] Cargando y limpiando: {os.path.basename(audio_path)}")

    try:
        signal, sr = librosa.load(audio_path, sr=config.AUDIO['sample_rate'])
        initial_duration = len(signal) / sr
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {os.path.basename(audio_path)}: {e}")
        return np.array([]), config.AUDIO['sample_rate']

    # --- LÓGICA ORIGINAL RESTAURADA ---
    # 1. Pre-énfasis
    preemphasis_coeff = config.AUDIO['preemphasis_alpha']
    emphasized_signal = lfilter([1, -preemphasis_coeff], [1], signal)
    
    # 2. Eliminar silencios con librosa.effects.trim
    trimmed_signal, _ = librosa.effects.trim(
        emphasized_signal, top_db=config.AUDIO['trim_db']
    )
    final_duration = len(trimmed_signal) / sr
    
    if VERBOSITY >= 2:
        print(f"  [FEAT] Librosa Trim: Duración original {initial_duration:.2f}s -> final {final_duration:.2f}s")

    # Si la señal es muy corta después de la limpieza, la omitimos
    if len(trimmed_signal) < 400: # ~25ms
        if VERBOSITY >= 1:
            print(f"  [WARN] Señal demasiado corta después de trim para {os.path.basename(audio_path)}. Se omite.")
        return np.array([]), sr

    # 3. Normalización de amplitud pico
    normalized_signal = trimmed_signal / (np.max(np.abs(trimmed_signal)) + 1e-10)
    
    return normalized_signal, config.AUDIO['sample_rate']


def extract_features(signal, sr):
    """Extrae el "Super-Vector" de características de una señal de audio."""
    # --- Esta función no necesita cambios, se mantiene como en la versión anterior ---
    frame_length = int(config.FEATURES['frame_length_ms'] / 1000 * sr)
    frame_stride = int(config.FEATURES['frame_stride_ms'] / 1000 * sr)
    all_features = []
    
    if VERBOSITY >= 2:
        print("  [FEAT] Extrayendo características...")

    # a) MFCCs
    if config.FEATURES.get('use_mfcc', True):
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=config.FEATURES['n_mfcc'], n_fft=frame_length, hop_length=frame_stride)
        all_features.append(mfccs)
        if config.FEATURES['use_delta']:
            all_features.append(librosa.feature.delta(mfccs))
        if config.FEATURES['use_delta2']:
            all_features.append(librosa.feature.delta(mfccs, order=2))

    # b) GFCCs
    if config.FEATURES.get('use_gfcc', False):
        gfccs = audio_utils.calculate_gfccs(signal, sr, n_gfcc=config.FEATURES['n_gfcc'], frame_len_ms=config.FEATURES['frame_length_ms'], hop_len_ms=config.FEATURES['frame_stride_ms'])
        all_features.append(gfccs)

    # c) Energía
    if config.FEATURES['use_energy']:
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=frame_stride)
        all_features.append(np.log(rms + 1e-10))
        
    # d) Pitch
    if config.FEATURES['use_pitch']:
        f0, _, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=frame_length, hop_length=frame_stride)
        f0[np.isnan(f0)] = 0
        all_features.append(f0.reshape(1, -1))
        all_features.append(voiced_probs.reshape(1, -1))
        
    # e) Características Espectrales Estándar
    if config.FEATURES['use_spectral_features']:
        all_features.append(librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=frame_length, hop_length=frame_stride))
        all_features.append(librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=frame_length, hop_length=frame_stride))
        
    # f) Características Espectrales Avanzadas
    if config.FEATURES['use_advanced_spectral_features']:
        frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_stride)
        k_vals, r_vals, sl_vals, sk_vals, f_vals = [], [], [], [], [0.0]
        for i in range(1, frames.shape[1]):
            curr, prev = frames[:, i], frames[:, i - 1]
            k_vals.append(audio_utils.calculate_kurtosis(curr))
            r_vals.append(audio_utils.calculate_spectral_rolloff(curr, sr))
            sl_vals.append(audio_utils.calculate_spectral_slope(curr, sr))
            sk_vals.append(audio_utils.calculate_spectral_skewness(curr, sr))
            f_vals.append(audio_utils.calculate_spectral_flux(curr, prev))
        k_vals.insert(0, k_vals[0] if k_vals else 0.0)
        r_vals.insert(0, r_vals[0] if r_vals else 0.0)
        sl_vals.insert(0, sl_vals[0] if sl_vals else 0.0)
        sk_vals.insert(0, sk_vals[0] if sk_vals else 0.0)
        all_features.append(np.array([k_vals, r_vals, sl_vals, sk_vals, f_vals]))

    if not all_features:
        raise ValueError("No se seleccionó ninguna característica para extraer.")

    min_len = min(feat.shape[1] for feat in all_features)
    features_matrix = np.vstack([feat[:, :min_len] for feat in all_features]).T
    
    if VERBOSITY >= 2:
        print(f"  [FEAT] Matriz de características ensamblada. Forma: {features_matrix.shape}")

    mean = np.mean(features_matrix, axis=0)
    std = np.std(features_matrix, axis=0)
    return (features_matrix - mean) / (std + 1e-10)