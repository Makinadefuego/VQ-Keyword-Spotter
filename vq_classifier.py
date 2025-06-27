# vq_classifier.py

import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import joblib
from tqdm import tqdm
import traceback

import config
import feature_extractor

VERBOSITY = config.LOGGING.get("verbosity_level", 1)

class VQClassifier:
    def __init__(self, n_clusters=config.MODEL.get('vq_clusters', 32)):
        self.n_clusters = n_clusters
        self.codebooks = {}
        self.labels = []
        if VERBOSITY >= 1:
            print(f"[VQ] Clasificador inicializado con k={self.n_clusters} clusters.")

    def train(self, all_features_map: dict):
        self.labels = sorted(all_features_map.keys())
        if not self.labels:
            raise ValueError("El mapa de características para el entrenamiento VQ está vacío.")
        
        for label, features in all_features_map.items():
            if VERBOSITY >= 1:
                print(f"\n[VQ-TRAIN] Entrenando codebook para '{label}' con {features.shape[0]} vectores.")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init='auto', verbose=0)
            kmeans.fit(features)
            self.codebooks[label] = kmeans.cluster_centers_

    def predict_scores(self, audio_path_or_array, processing_params: dict):
        if not self.codebooks:
            raise RuntimeError("El modelo VQ no ha sido entrenado.")
        try:
            signal, sr = feature_extractor.load_and_clean_audio(audio_path_or_array, params=processing_params)
            if signal.size == 0: return None
            features = feature_extractor.extract_features(signal, sr)
            if features.size == 0: return None
        except Exception:
            print(f"[ERROR] Excepción durante la extracción de características para VQ:"); traceback.print_exc()
            return None
        errors = {label: np.mean(np.min(cdist(features, codebook), axis=1)) for label, codebook in self.codebooks.items()}
        return errors

    def predict(self, audio_path_or_array, processing_params: dict, rejection_threshold=None):
        scores = self.predict_scores(audio_path_or_array, processing_params)
        if scores is None: return None
        if rejection_threshold is None: rejection_threshold = config.MODEL.get("rejection_threshold_vq", 1.5)
        best_prediction = min(scores, key=scores.get)
        min_error = scores[best_prediction]
        if VERBOSITY >= 1:
            print(f"  [VQ-PREDICT] Mejor candidato: '{best_prediction}' (Error: {min_error:.4f}), Umbral: {rejection_threshold}")
        return best_prediction if min_error <= rejection_threshold else config.MODEL.get("garbage_label")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[VQ] Modelo guardado en: {path}")

    @staticmethod
    def load_model(path):
        if not os.path.exists(path): raise FileNotFoundError(f"Modelo VQ no encontrado en: {path}")
        return joblib.load(path)