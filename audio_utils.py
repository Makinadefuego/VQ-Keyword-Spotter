# audio_utils.py

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import stft
import librosa
from gammatone.gtgram import gtgram
import webrtcvad
from scipy.fft import dct  # <-- ¡ESTA ES LA IMPORTACIÓN CLAVE!

# --- FUNCIONES DE CÁLCULO DE CARACTERÍSTICAS ---

def calculate_kurtosis(frame):
    if np.all(frame == 0): return 0.0
    return kurtosis(frame, fisher=True)

def calculate_spectral_rolloff(frame, sr, roll_percent=0.85):
    fft_vals = np.abs(np.fft.rfft(frame)) ** 2
    total_energy = np.sum(fft_vals)
    if total_energy == 0: return 0.0
    cumulative_energy = np.cumsum(fft_vals)
    rolloff_indices = np.where(cumulative_energy >= roll_percent * total_energy)[0]
    if len(rolloff_indices) == 0: return 0.0
    freqs = np.fft.rfftfreq(len(frame), d=1/sr)
    return freqs[rolloff_indices[0]]

def calculate_spectral_slope(frame, sr):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1/sr)
    if np.sum(spectrum) < 1e-10: return 0.0
    log_spectrum = np.log10(spectrum + 1e-10)
    try:
        slope, _ = np.polyfit(freqs, log_spectrum, 1)
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return slope

def calculate_spectral_skewness(frame, sr):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1/sr)
    sum_spectrum = np.sum(spectrum)
    if sum_spectrum < 1e-10: return 0.0
    mean_freq = np.sum(spectrum * freqs) / sum_spectrum
    std_freq = np.sqrt(np.sum(spectrum * (freqs - mean_freq)**2) / sum_spectrum)
    if std_freq < 1e-10: return 0.0
    return np.sum(spectrum * (freqs - mean_freq)**3) / (sum_spectrum * std_freq**3)

def calculate_spectral_flux(frame_current, frame_previous):
    spec_current = np.abs(np.fft.rfft(frame_current))
    spec_previous = np.abs(np.fft.rfft(frame_previous))
    spec_current /= (np.sum(spec_current) + 1e-10)
    spec_previous /= (np.sum(spec_previous) + 1e-10)
    return np.sqrt(np.sum((spec_current - spec_previous)**2))

def calculate_gfccs(signal, sr, n_gfcc=13, frame_len_ms=25, hop_len_ms=10):
    """Calcula los Coeficientes Cepstrales en la Escala de Gammatone (GFCC)."""
    window_time = frame_len_ms / 1000
    hop_time = hop_len_ms / 1000
    gt_spec = gtgram(signal, sr, window_time, hop_time, channels=64, f_min=50)
    log_gt_spec = np.log(gt_spec + 1e-10)
    
    # --- CORRECCIÓN APLICADA ---
    # Usamos dct de scipy.fft, no de librosa.feature
    gfccs = dct(log_gt_spec, type=2, axis=0, norm='ortho')
    
    return gfccs[:n_gfcc, :]

# --- FUNCIONES DE PROCESAMIENTO DE AUDIO ---

def vad_trim(signal, sr, aggressiveness=3):
    """Usa VAD de WebRTC para recortar los silencios de una señal."""
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError(f"VAD soporta 8k, 16k, 32k, 48k Hz. Se recibió {sr} Hz.")
    
    vad = webrtcvad.Vad(aggressiveness)
    signal_int16 = np.int16(signal * 32767)
    
    frame_duration_ms = 30
    frame_samples = int(sr * frame_duration_ms / 1000)
    
    voiced_frames = []
    for i in range(0, len(signal_int16), frame_samples):
        frame = signal_int16[i:i+frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)), 'constant')
        
        try:
            if vad.is_speech(frame.tobytes(), sample_rate=sr):
                voiced_frames.append(signal[i:i+frame_samples])
        except Exception:
            pass
    
    if not voiced_frames:
        return np.array([])
    
    return np.concatenate(voiced_frames)