# vq_classifier.py

import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import joblib
from tqdm import tqdm

import config
import feature_extractor

VERBOSITY = config.LOGGING.get("verbosity_level", 1)

class VQClassifier:
    def __init__(self, n_clusters=config.MODEL['vq_clusters']):
        self.n_clusters = n_clusters
        self.codebooks = {}
        self.labels = []
        if VERBOSITY >= 1:
            print(f"[VQ] Clasificador inicializado con k={self.n_clusters} clusters.")

    def train(self, training_path):
        self.labels = sorted([name for name in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, name))])
        
        if not self.labels:
            raise ValueError(f"No se encontraron directorios de palabras en '{training_path}'.")

        if VERBOSITY >= 1:
            print(f"[VQ-TRAIN] Palabras a entrenar: {', '.join(self.labels)}")

        for label in self.labels:
            if VERBOSITY >= 1:
                print(f"\n--- Entrenando palabra: '{label}' ---")
            
            word_features = []
            word_dir = os.path.join(training_path, label)
            audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(('.m4a', '.wav'))]
            
            for filename in tqdm(audio_files, desc=f"  Procesando '{label}'", disable=(VERBOSITY < 1)):
                filepath = os.path.join(word_dir, filename)
                try:
                    signal, sr = feature_extractor.load_and_clean_audio(filepath)
                    if len(signal) > 0:
                        features = feature_extractor.extract_features(signal, sr)
                        if features.shape[0] > 0:
                            word_features.append(features)
                except Exception as e:
                    print(f"\n  [ERROR] Durante extracción en {filename}: {e}")

            if not word_features:
                if VERBOSITY >= 1:
                    print(f"  [WARN] No se pudieron extraer características para '{label}'. Se omitirá.")
                continue

            all_word_features = np.vstack(word_features)
            
            if VERBOSITY >= 1:
                print(f"  [VQ-TRAIN] Entrenando codebook para '{label}' con {all_word_features.shape[0]} vectores de características.")
            
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10, verbose=(1 if VERBOSITY >= 2 else 0))
            kmeans.fit(all_word_features)
            self.codebooks[label] = kmeans.cluster_centers_
            if VERBOSITY >= 1:
                print(f"  [VQ-TRAIN] Codebook para '{label}' creado exitosamente.")

    def predict(self, audio_path):
        if not self.codebooks:
            raise RuntimeError("El modelo no ha sido entrenado.")
            
        if VERBOSITY >= 1:
            print(f"\n--- Iniciando predicción para: {os.path.basename(audio_path)} ---")
        
        try:
            signal, sr = feature_extractor.load_and_clean_audio(audio_path)
            if len(signal) == 0:
                if VERBOSITY >= 1: print("  [VQ-PREDICT] El audio está vacío o no contiene voz. No se puede predecir.")
                return None
            
            features = feature_extractor.extract_features(signal, sr)
            if features.shape[0] == 0:
                if VERBOSITY >= 1: print("  [VQ-PREDICT] No se pudieron extraer características. No se puede predecir.")
                return None
        except Exception as e:
            print(f"  [ERROR] Durante la predicción en {os.path.basename(audio_path)}: {e}")
            return None

        errors = {}
        if VERBOSITY >= 2:
            print("  [VQ-PREDICT] Calculando error de cuantización contra cada codebook:")
            
        for label, codebook in self.codebooks.items():
            error = self._calculate_quantization_error(features, codebook)
            errors[label] = error
            if VERBOSITY >= 2:
                print(f"    - Error para '{label}': {error:.4f}")
            
        prediction = min(errors, key=errors.get)
        if VERBOSITY >= 1:
            print(f"  [VQ-PREDICT] Predicción final: '{prediction}' (con el menor error: {errors[prediction]:.4f})")
        
        return prediction

    def _calculate_quantization_error(self, features, codebook):
        distances = cdist(features, codebook, 'euclidean')
        return np.mean(np.min(distances, axis=1))

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if VERBOSITY >= 1:
            print(f"\n[VQ] Guardando modelo en: {path}")
        joblib.dump(self, path)
        if VERBOSITY >= 1:
            print("[VQ] Modelo guardado exitosamente.")

    @staticmethod
    def load_model(path):
        if config.LOGGING.get("verbosity_level", 1) >= 1:
            print(f"[VQ] Cargando modelo desde: {path}")
        model = joblib.load(path)
        if config.LOGGING.get("verbosity_level", 1) >= 1:
            print("[VQ] Modelo cargado exitosamente.")
        return model