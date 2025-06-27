# audio_utils.py

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import stft
import librosa
from gammatone.gtgram import gtgram
import webrtcvad
from scipy.fft import dct

import config

# --- FUNCIONES DE CÁLCULO DE CARACTERÍSTICAS (sin cambios) ---
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
    window_time = frame_len_ms / 1000
    hop_time = hop_len_ms / 1000
    gt_spec = gtgram(signal, sr, window_time, hop_time, channels=64, f_min=50)
    log_gt_spec = np.log(gt_spec + 1e-10)
    gfccs = dct(log_gt_spec, type=2, axis=0, norm='ortho')
    return gfccs[:n_gfcc, :]

# --- FUNCIONES DE PROCESAMIENTO DE AUDIO (sin cambios) ---
def vad_trim(signal, sr, aggressiveness=3):
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
    
    return np.concatenate(voiced_frames) if voiced_frames else np.array([])

# --- FUNCIONES DE SUPRESIÓN DE RUIDO ADAPTATIVA (CORREGIDAS) ---

def create_noise_profile(noise_clip, sr):
    frame_len_ms = config.FEATURES.get('frame_length_ms', 25)
    hop_len_ms = config.FEATURES.get('frame_stride_ms', 10)
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)
    
    # 1. Sanitizar el clip de entrada para eliminar valores no finitos.
    if not np.all(np.isfinite(noise_clip)):
        noise_clip = np.nan_to_num(noise_clip, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Si después de sanitizar, el clip es demasiado corto, retornamos un perfil de silencio.
    if len(noise_clip) < frame_length:
        print("[AUDIO-UTILS-WARN] El clip de ruido es demasiado corto para crear un perfil. Se usará un perfil de silencio.")
        return np.zeros(frame_length // 2 + 1)

    # 3. Calcular el perfil de ruido.
    noise_stft = librosa.stft(noise_clip, n_fft=frame_length, hop_length=hop_length)
    noise_magnitude = np.abs(noise_stft)
    mean_noise_magnitude = np.mean(noise_magnitude, axis=1)
    
    # 4. Asegurarnos de que el perfil de ruido resultante no contenga valores no finitos.
    mean_noise_magnitude = np.nan_to_num(mean_noise_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    
    return mean_noise_magnitude


def spectral_subtraction(signal, noise_profile, sr, alpha=2.0):
    beta = config.AUDIO['spectral_subtraction'].get('beta', 0.01)
    frame_len_ms = config.FEATURES.get('frame_length_ms', 25)
    hop_len_ms = config.FEATURES.get('frame_stride_ms', 10)
    
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)

    # Si la señal ya es muy corta, no hacemos nada.
    if len(signal) < frame_length: return np.array([]) 

    signal_stft = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    signal_magnitude, signal_phase = librosa.magphase(signal_stft)
    
    # Asegurar que el perfil de ruido tenga la forma correcta para la resta
    if noise_profile.shape[0] != signal_magnitude.shape[0]:
        print("[AUDIO-UTILS-WARN] Inconsistencia de formas entre señal y perfil de ruido. Omitiendo sustracción.")
        return signal

    noise_profile_2d = np.tile(noise_profile, (signal_magnitude.shape[1], 1)).T
    
    signal_power = signal_magnitude ** 2
    noise_power = (noise_profile_2d ** 2) * alpha
    
    subtracted_power = signal_power - noise_power
    noise_floor = (noise_profile_2d ** 2) * beta
    cleaned_power = np.maximum(subtracted_power, noise_floor)
    cleaned_magnitude = np.sqrt(cleaned_power)

    cleaned_stft = cleaned_magnitude * signal_phase
    cleaned_signal = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(signal))

    return cleaned_signal