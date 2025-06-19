# analyze_and_report.py

import os
import time
from collections import Counter
import numpy as np
import librosa
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier

def plot_confusion_matrix(cm, labels, path='confusion_matrix.png'):
    """Función para visualizar y guardar la matriz de confusión."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión del Clasificador VQ')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(path)
    plt.show()
    print(f"Matriz de confusión guardada en '{path}'")

def get_audio_duration(file_path):
    """Obtiene la duración de un archivo de audio en segundos."""
    try:
        return librosa.get_duration(path=file_path)
    except Exception:
        return 0

def main():
    """
    Función principal para realizar un análisis estadístico completo del modelo.
    """
    print("--- INICIO DEL ANÁLISIS ESTADÍSTICO DEL MODELO ---")

    # 1. Cargar modelo y verificar dataset
    model_path = config.PATHS['output_model']
    test_path = config.PATHS['dataset_test']
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo '{model_path}' no encontrado. Ejecuta train.py primero.")
        return
    if not os.path.exists(test_path) or not os.listdir(test_path):
        print(f"[ERROR] Directorio de prueba '{test_path}' no encontrado o vacío.")
        return
        
    print("Cargando modelo...")
    classifier = VQClassifier.load_model(model_path)

    # 2. Recolectar predicciones y metadatos
    results = []
    print("\nRealizando predicciones en el conjunto de prueba...")
    start_time = time.time()

    for label in classifier.labels:
        word_dir = os.path.join(test_path, label)
        if not os.path.exists(word_dir):
            continue
        
        # Adaptado para leer .m4a o .wav, como en tu código
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(('.m4a', '.wav'))]
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            prediction = classifier.predict(filepath)
            duration = get_audio_duration(filepath)
            
            if prediction:
                results.append({
                    "file": filename,
                    "true_label": label,
                    "predicted_label": prediction,
                    "is_correct": label == prediction,
                    "duration": duration
                })
    
    total_time = time.time() - start_time
    print(f"Análisis completado en {total_time:.2f} segundos.")

    if not results:
        print("\nNo se pudo procesar ningún archivo. El análisis no puede continuar.")
        return

    true_labels = [r['true_label'] for r in results]
    predicted_labels = [r['predicted_label'] for r in results]
    
    # --- 3. REPORTE DE MÉTRICAS DE CLASIFICACIÓN ---
    print("\n\n--- 1. REPORTE DE RENDIMIENTO GENERAL ---")
    report = classification_report(true_labels, predicted_labels, labels=classifier.labels, zero_division=0)
    print(report)
    
    # --- 4. MATRIZ DE CONFUSIÓN Y ANÁLISIS DE ERRORES ---
    print("\n\n--- 2. ANÁLISIS DETALLADO DE ERRORES ---")
    cm = confusion_matrix(true_labels, predicted_labels, labels=classifier.labels)
    plot_confusion_matrix(cm, classifier.labels)

    # Top confusions
    np.fill_diagonal(cm, 0) # Ignoramos los aciertos en la diagonal
    top_confusions = []
    for i in range(len(classifier.labels)):
        for j in range(len(classifier.labels)):
            if cm[i, j] > 0:
                top_confusions.append((cm[i, j], classifier.labels[i], classifier.labels[j]))
    
    print("\nTop Confusiones (Nº de veces, Palabra Real -> Predicción Errónea):")
    if not top_confusions:
        print("¡Felicidades! No hubo ninguna confusión en el set de prueba.")
    else:
        for count, true, pred in sorted(top_confusions, reverse=True):
            print(f"  - {count} vez/veces: '{true}' fue confundido con '{pred}'")
            
    # Lista de fallos específicos
    failed_predictions = [r for r in results if not r['is_correct']]
    print("\nArchivos específicos con errores:")
    if not failed_predictions:
        print("¡No se encontraron errores específicos!")
    else:
        for fail in failed_predictions:
            print(f"  - Archivo: {fail['file']:<25} | Real: {fail['true_label']:<10} | Predicho: {fail['predicted_label']:<10}")
            
    # --- 5. ESTADÍSTICAS DEL DATASET DE PRUEBA ---
    print("\n\n--- 3. ESTADÍSTICAS DEL DATASET DE PRUEBA ---")
    durations = [r['duration'] for r in results]
    print(f"Número total de muestras de prueba: {len(durations)}")
    print(f"Duración total de los audios: {sum(durations):.2f} segundos")
    print(f"Duración promedio por audio: {np.mean(durations):.2f} segundos")
    print(f"Duración audio más corto: {np.min(durations):.2f} segundos")
    print(f"Duración audio más largo: {np.max(durations):.2f} segundos")

    print("\n\n--- ANÁLISIS FINALIZADO ---")

if __name__ == '__main__':
    # Librosa puede necesitar ffmpeg para leer archivos .m4a
    # Este es un recordatorio para el usuario
    print("Recordatorio: Para procesar archivos .m4a, asegúrate de tener ffmpeg instalado y accesible en el PATH del sistema.")
    main()