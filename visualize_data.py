# explain_model.py
#
# Una caja de herramientas unificada para analizar y explicar el comportamiento
# del clasificador VQ. Ofrece tres modos de análisis a través de un menú interactivo.
#
# MODOS:
# 1. Huella Acústica: Visualiza el codebook de una palabra como un heatmap.
# 2. Matriz de Distorsión: Muestra qué tan "cerca" está cada palabra de ser confundida con otra.
# 3. Importancia de Características: Determina qué features son más útiles para la clasificación.
#
# USO: python explain_model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

# Importaciones de nuestro propio proyecto
import config
from vq_classifier import VQClassifier
import feature_extractor

# --- HERRAMIENTA 1: VISUALIZACIÓN DE HUELLA ACÚSTICA (CODEBOOK) ---

def plot_codebook_fingerprint(classifier, word):
    """
    Visualiza el codebook de una palabra como un heatmap (huella acústica).
    """
    if word not in classifier.codebooks:
        print(f"\n[ERROR] La palabra '{word}' no se encuentra en el modelo.")
        print(f"Palabras disponibles: {', '.join(classifier.labels)}")
        return

    print(f"\nGenerando huella acústica para '{word}'...")
    codebook = classifier.codebooks[word]
    normalized_codebook = minmax_scale(codebook, axis=0)

    plt.figure(figsize=(20, 8))
    sns.heatmap(
        normalized_codebook.T,
        cmap='viridis',
        yticklabels=False,
    )
    
    plt.title(f'Huella Acústica (Codebook) para la Palabra: "{word.upper()}"', fontsize=16)
    plt.xlabel(f'Clusters del Codebook (0 a {classifier.n_clusters - 1})', fontsize=12)
    plt.ylabel('Dimensiones de Características (Features)', fontsize=12)
    
    output_path = f'fingerprint_{word}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualización guardada en '{output_path}'")
    plt.show()

# --- HERRAMIENTA 2: ANÁLISIS DE MATRIZ DE DISTORSIÓN ---

def analyze_distortion_matrix(classifier, test_path):
    """
    Calcula y visualiza una matriz de distorsión promedio entre todas las palabras del set de prueba.
    """
    labels = classifier.labels
    distortion_matrix = pd.DataFrame(index=labels, columns=labels, dtype=np.float64)

    print("\nCalculando matriz de distorsión en el set de prueba (esto puede tardar)...")
    for true_label in tqdm(labels, desc="Palabra Verdadera"):
        word_dir = os.path.join(test_path, true_label)
        if not os.path.exists(word_dir): continue
            
        errors_for_all_codebooks = {label: [] for label in labels}
        
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(('.m4a', '.wav'))]
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            try:
                signal, sr = feature_extractor.load_and_clean_audio(filepath)
                if len(signal) == 0: continue
                features = feature_extractor.extract_features(signal, sr)
                if features.shape[0] == 0: continue
                
                for codebook_label, codebook in classifier.codebooks.items():
                    error = classifier._calculate_quantization_error(features, codebook)
                    errors_for_all_codebooks[codebook_label].append(error)
            except Exception:
                continue
        
        for codebook_label in labels:
            if errors_for_all_codebooks[codebook_label]:
                distortion_matrix.loc[true_label, codebook_label] = np.mean(errors_for_all_codebooks[codebook_label])

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        distortion_matrix.astype(float), annot=True, fmt=".3f", cmap="coolwarm_r", linewidths=.5
    )
    plt.title('Matriz de Distorsión Promedio', fontsize=16)
    plt.xlabel('Codebook del Clasificador (Modelo)', fontsize=12)
    plt.ylabel('Audio de Prueba Real (Verdad)', fontsize=12)
    
    output_path = 'distortion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nMatriz de distorsión guardada en '{output_path}'")
    plt.show()

# --- HERRAMIENTA 3: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS ---

def analyze_feature_importance(classifier, train_path):
    """
    Calcula y visualiza la importancia de cada característica para la clasificación.
    """
    labels = classifier.labels
    mean_feature_vectors = {}
    print("\nCalculando vector promedio por palabra (usando datos de entrenamiento)...")
    for label in tqdm(labels, desc="Procesando palabras"):
        word_dir = os.path.join(train_path, label)
        if not os.path.exists(word_dir): continue

        all_features = []
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(('.m4a', '.wav')) and "_aug_" not in f]
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            try:
                signal, sr = feature_extractor.load_and_clean_audio(filepath)
                if len(signal) > 0:
                    features = feature_extractor.extract_features(signal, sr)
                    all_features.append(features)
            except Exception:
                continue
        
        if all_features:
            mean_feature_vectors[label] = np.mean(np.vstack(all_features), axis=0)

    mean_matrix = np.array(list(mean_feature_vectors.values()))
    feature_importance = np.std(mean_matrix, axis=0)
    
    # Crear nombres para las características (aproximación basada en config.py)
    feat_config = config.FEATURES
    feature_names = []
    if feat_config.get('use_mfcc'):
        for i in range(feat_config['n_mfcc']): feature_names.append(f'MFCC_{i}')
        if feat_config.get('use_delta'):
            for i in range(feat_config['n_mfcc']): feature_names.append(f'Delta_MFCC_{i}')
        if feat_config.get('use_delta2'):
            for i in range(feat_config['n_mfcc']): feature_names.append(f'Delta2_MFCC_{i}')
    if feat_config.get('use_gfcc'):
        for i in range(feat_config['n_gfcc']): feature_names.append(f'GFCC_{i}')
    if feat_config.get('use_energy'): feature_names.append('Energy')
    if feat_config.get('use_pitch'):
        feature_names.extend(['Pitch(f0)', 'Voiced_Prob'])
    if feat_config.get('use_spectral_features'):
        feature_names.extend(['Spec_Centroid', 'Spec_Bandwidth'])
    if feat_config.get('use_advanced_spectral_features'):
        feature_names.extend(['Kurtosis', 'Rolloff', 'Slope', 'Skewness', 'Flux'])

    if len(feature_names) != len(feature_importance):
        print(f"\n[ADVERTENCIA] El número de nombres de features ({len(feature_names)}) no coincide con el de importancias ({len(feature_importance)}). Se usarán índices numéricos.")
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
    n_top_features = 20
    indices = np.argsort(feature_importance)[-n_top_features:]
    top_scores = feature_importance[indices]
    top_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 8))
    plt.barh(np.arange(n_top_features), top_scores, align='center')
    plt.yticks(np.arange(n_top_features), top_names)
    plt.xlabel('Importancia (Desviación Estándar entre Clases)')
    plt.title(f'Top {n_top_features} Características Más Importantes')
    plt.gca().invert_yaxis()
    
    output_path = 'feature_importance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGráfico de importancia de características guardado en '{output_path}'")
    plt.show()


# --- MENÚ PRINCIPAL ---

def main():
    """
    Muestra un menú para que el usuario elija qué herramienta de análisis ejecutar.
    """
    model_path = config.PATHS['output_model']
    train_path = config.PATHS['dataset_train']
    test_path = config.PATHS['dataset_test']

    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo '{model_path}' no encontrado. Ejecuta train.py primero.")
        return
        
    print("Cargando modelo...")
    try:
        classifier = VQClassifier.load_model(model_path)
        print("¡Modelo cargado!")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    while True:
        print("\n" + "="*50)
        print("--- CAJA DE HERRAMIENTAS PARA EXPLICAR EL MODELO ---")
        print("="*50)
        print("1. Visualizar Huella Acústica (Codebook) de una palabra")
        print("2. Generar Matriz de Distorsión (Análisis de Confusión)")
        print("3. Analizar Importancia de Características (Feature Importance)")
        print("4. Salir")
        
        choice = input("\nElige una opción (1-4): ")

        if choice == '1':
            print(f"\nPalabras disponibles: {', '.join(classifier.labels)}")
            word_to_visualize = input("Introduce la palabra que quieres visualizar: ").strip().lower()
            if word_to_visualize:
                plot_codebook_fingerprint(classifier, word_to_visualize)
            else:
                print("[ERROR] No introdujiste ninguna palabra.")
        
        elif choice == '2':
            if not os.path.exists(test_path):
                print(f"[ERROR] Directorio de prueba '{test_path}' no encontrado. Ejecuta prepare_dataset.py primero.")
            else:
                analyze_distortion_matrix(classifier, test_path)
        
        elif choice == '3':
            if not os.path.exists(train_path):
                print(f"[ERROR] Directorio de entrenamiento '{train_path}' no encontrado. Ejecuta prepare_dataset.py primero.")
            else:
                analyze_feature_importance(classifier, train_path)

        elif choice == '4':
            print("Saliendo de la herramienta de análisis.")
            break
        
        else:
            print("[ERROR] Opción no válida. Por favor, elige un número del 1 al 4.")

if __name__ == '__main__':
    main()