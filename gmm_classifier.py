# gmm_classifier.py

import os
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
from tqdm import tqdm
import traceback

import config
import feature_extractor

VERBOSITY = config.LOGGING.get("verbosity_level", 1)

class GMMClassifier:
    def __init__(self, n_components=config.MODEL.get('gmm_components', 16), covariance_type='diag'):
        """
        Inicializa el clasificador GMM.
        - n_components: El número de distribuciones gaussianas para modelar cada palabra.
        - covariance_type: 'diag' es rápido y generalmente funciona bien para audio.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmms = {}
        self.labels = []
        if VERBOSITY >= 1:
            print(f"[GMM] Clasificador inicializado con {self.n_components} componentes y covarianza '{self.covariance_type}'.")

    def train(self, all_features_map: dict):
        """
        Entrena un modelo GMM separado para cada palabra en el mapa de características.
        """
        self.labels = sorted(all_features_map.keys())
        if not self.labels:
            raise ValueError("El mapa de características para el entrenamiento GMM está vacío.")
        
        for label, features in all_features_map.items():
            if features.shape[0] < self.n_components:
                print(f"\n[GMM-WARN] No hay suficientes datos para la palabra '{label}' "
                      f"({features.shape[0]} muestras < {self.n_components} componentes). "
                      "Saltando entrenamiento para esta palabra.")
                continue

            if VERBOSITY >= 1:
                print(f"\n[GMM-TRAIN] Entrenando GMM para '{label}' con {features.shape[0]} vectores.")
            
            # Inicializamos y entrenamos el GMM para la palabra actual
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=0,
                verbose=0, # Cambia a 1 para ver el progreso de la convergencia
                max_iter=100,
                n_init=3 # Entrenar varias veces y quedarse con el mejor
            )
            gmm.fit(features)
            self.gmms[label] = gmm
        
        # Actualizamos las etiquetas para que solo contengan los modelos que sí se entrenaron
        self.labels = sorted(self.gmms.keys())

    def predict_scores(self, audio_path_or_array, processing_params: dict):
        """
        Calcula los log-likelihood scores para un audio dado contra todos los GMMs entrenados.
        Un score más alto (menos negativo) es mejor.
        """
        if not self.gmms:
            raise RuntimeError("El modelo GMM no ha sido entrenado.")
        try:
            signal, sr = feature_extractor.load_and_clean_audio(audio_path_or_array, params=processing_params)
            if signal.size == 0: 
                return None
            
            features = feature_extractor.extract_features(signal, sr)
            if features.size == 0: 
                return None
        except Exception:
            print(f"[ERROR] Excepción durante la extracción de características para GMM:")
            traceback.print_exc()
            return None
        
        # Calculamos el log-likelihood promedio para cada modelo GMM
        log_likelihoods = {}
        for label, gmm in self.gmms.items():
            # gmm.score_samples devuelve el log-likelihood por muestra. Lo promediamos.
            score = np.mean(gmm.score_samples(features))
            log_likelihoods[label] = score
            
        return log_likelihoods

    def predict(self, audio_path_or_array, processing_params: dict, rejection_threshold=None):
        """
        Predice la etiqueta de un audio.
        """
        scores = self.predict_scores(audio_path_or_array, processing_params)
        if scores is None: 
            return None
        
        # Si no se pasa un umbral, usamos el de la configuración
        if rejection_threshold is None:
            rejection_threshold = config.MODEL.get("rejection_threshold_gmm", -80.0)
            
        # La mejor predicción es la que tiene el score MÁS ALTO (máxima probabilidad)
        best_prediction = max(scores, key=scores.get)
        max_score = scores[best_prediction]
        
        if VERBOSITY >= 1:
            print(f"  [GMM-PREDICT] Mejor candidato: '{best_prediction}' (Score: {max_score:.4f}), Umbral: {rejection_threshold}")
            
        # La predicción es válida si el score es MAYOR que el umbral de rechazo
        return best_prediction if max_score >= rejection_threshold else config.MODEL.get("garbage_label")

    def save_model(self, path):
        """Guarda el objeto clasificador completo en un archivo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[GMM] Modelo guardado en: {path}")

    @staticmethod
    def load_model(path):
        """Carga un clasificador desde un archivo."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo GMM no encontrado en: {path}")
        return joblib.load(path)