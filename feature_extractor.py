# feature_extractor.py

import numpy as np
import librosa
from scipy.signal import lfilter
import os

import config
import audio_utils

VERBOSITY = config.LOGGING.get("verbosity_level", 1)
# Umbral mínimo de muestras para que una señal sea considerada válida para procesar.
# 1600 muestras a 16000 Hz son 100 ms, un buen mínimo para una palabra.
MIN_SIGNAL_LENGTH = 1600

def load_and_clean_audio(audio_path_or_array, params: dict):
    """
    Carga y limpia el audio de forma robusta, con múltiples puntos de control
    para evitar que la señal se destruya durante el proceso.
    """
    sr = config.AUDIO['sample_rate']
    
    audio_name_for_logs = "grabación en tiempo real"
    if isinstance(audio_path_or_array, str):
        audio_name_for_logs = os.path.basename(audio_path_or_array)

    if VERBOSITY >= 2: print(f"  [FEAT] Cargando: {audio_name_for_logs}")

    try:
        signal = audio_path_or_array if isinstance(audio_path_or_array, np.ndarray) else librosa.load(audio_path_or_array, sr=sr)[0]
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {audio_name_for_logs}: {e}"); return np.array([]), sr

    # 1. Sanitizar la señal de entrada para evitar errores de overflow/NaN en librosa
    if not np.all(np.isfinite(signal)):
        if VERBOSITY >= 1: print(f"  [WARN] Se detectaron valores no finitos en la señal de entrada. Sanitizando.");
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if VERBOSITY >= 2: print(f"  [FEAT-DEBUG] Longitud inicial de la señal: {len(signal)} muestras.")

    # 2. Comprobación de longitud mínima inicial
    if len(signal) < MIN_SIGNAL_LENGTH:
        if VERBOSITY >= 1: print(f"  [WARN] Señal de entrada demasiado corta ({len(signal)} muestras). Se omite."); return np.array([]), sr
    
    # 3. Sustracción espectral (si se solicita)
    noise_profile = params.get('noise_profile')
    if params.get('use_subtraction', False) and noise_profile is not None and np.any(noise_profile):
        if VERBOSITY >= 2: print("  [FEAT] Aplicando Sustracción Espectral...")
        signal = audio_utils.spectral_subtraction(signal, noise_profile, sr, alpha=params.get('noise_alpha', 2.0))
        if VERBOSITY >= 2: print(f"  [FEAT-DEBUG] Longitud tras sustracción: {len(signal)} muestras.")
    
    # 4. Comprobación de longitud tras filtrado de ruido
    if len(signal) < MIN_SIGNAL_LENGTH:
        if VERBOSITY >= 1: print(f"  [WARN] Señal demasiado corta tras filtrado de ruido ({len(signal)}). Se omite."); return np.array([]), sr

    # 5. Pre-énfasis
    if params.get('use_preemphasis', True):
        if VERBOSITY >= 2: print("  [FEAT] Aplicando Pre-énfasis...")
        signal = lfilter([1, -config.AUDIO['preemphasis_alpha']], [1], signal)
    
    # 6. Recorte de silencios
    if params.get('use_trim', True):
        if VERBOSITY >= 2: print(f"  [FEAT] Aplicando Recorte de Silencios (Trim) con top_db={config.AUDIO['trim_db']}...")
        signal, _ = librosa.effects.trim(signal, top_db=config.AUDIO['trim_db'])
        if VERBOSITY >= 2: print(f"  [FEAT-DEBUG] Longitud tras recorte: {len(signal)} muestras.")
    
    # 7. Comprobación final de longitud
    if len(signal) < MIN_SIGNAL_LENGTH:
        if VERBOSITY >= 1: print(f"  [WARN] Señal demasiado corta después de la limpieza final ({len(signal)}). Se omite."); return np.array([]), sr
    
    # 8. Normalización de amplitud
    max_abs = np.max(np.abs(signal))
    if max_abs > 1e-9: # Evitar división por cero en señales silenciosas
        signal = signal / max_abs
    
    return signal, sr


def extract_features(signal, sr):
    """
    Lógica de extracción de características final y robusta. Si la señal es demasiado
    corta para calcular deltas (si están activados), se descarta por completo.
    """
    frame_length = int(config.FEATURES['frame_length_ms'] / 1000 * sr)
    frame_stride = int(config.FEATURES['frame_stride_ms'] / 1000 * sr)
    
    if len(signal) < frame_length: 
        return np.array([])
        
    # --- 1. Calcular MFCCs y realizar la comprobación crítica de longitud ---
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=config.FEATURES['n_mfcc'], n_fft=frame_length, hop_length=frame_stride)
    num_frames = mfccs.shape[1]

    if num_frames == 0:
        return np.array([])

    # --- CORRECCIÓN CLAVE: ESTA ES LA PROTECCIÓN QUE EVITA EL ERROR ---
    delta_width = 9 # Ancho por defecto en librosa
    if (config.FEATURES.get('use_delta') or config.FEATURES.get('use_delta2')) and (num_frames < delta_width):
        if VERBOSITY >= 1: 
            print(f"  [WARN] Descartando audio: tramas insuficientes ({num_frames}) para deltas (se necesitan {delta_width}).")
        return np.array([])
    
    # --- 2. Si la comprobación pasa, procedemos a construir la lista de características ---
    all_features = []
    
    if config.FEATURES.get('use_mfcc'):
        all_features.append(mfccs)
    if config.FEATURES.get('use_delta'):
        all_features.append(librosa.feature.delta(mfccs, width=delta_width))
    if config.FEATURES.get('use_delta2'):
        all_features.append(librosa.feature.delta(mfccs, order=2, width=delta_width))
        
    # --- 3. Calcular el resto de características ---
    # Todas las siguientes características deben ser recortadas a num_frames para garantizar consistencia.
    
    if config.FEATURES.get('use_energy'):
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=frame_stride)
        all_features.append(np.log(rms[:, :num_frames] + 1e-10))
        
    if config.FEATURES.get('use_pitch'):
        f0, _, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=frame_length, hop_length=frame_stride)
        f0[np.isnan(f0)] = 0
        all_features.append(f0.reshape(1, -1)[:, :num_frames])
        all_features.append(voiced_probs.reshape(1, -1)[:, :num_frames])
        
    # --- 4. Ensamblaje y Normalización Final ---
    try:
        features_matrix = np.vstack(all_features).T
    except ValueError as e:
        print(f"  [WARN] Error al apilar características, probablemente por inconsistencia de formas: {e}. Descartando audio.")
        return np.array([])

    mean = np.mean(features_matrix, axis=0)
    std = np.std(features_matrix, axis=0)
    std[std < 1e-9] = 1.0 # Evitar división por cero
    
    return (features_matrix - mean) / std