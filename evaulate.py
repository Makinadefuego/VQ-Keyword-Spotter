# evaulate.py

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# --- CORRECCIÓN DE LA IMPORTACIÓN ---
from tqdm import tqdm

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier

def plot_confusion_matrix(cm, labels, output_path='confusion_matrix.png'):
    """Función mejorada para visualizar y guardar la matriz de confusión."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión del Clasificador VQ', fontsize=16)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show(block=False) # Usar block=False para que no detenga el script
    print(f"\nMatriz de confusión guardada en '{output_path}'")

def main():
    """
    Función principal para realizar una evaluación completa y rigurosa del modelo.
    """
    print("--- Inicio de la Evaluación Profesional del Modelo ---")

    # 1. Cargar configuración y verificar rutas
    test_path = config.PATHS['dataset_test']
    model_path = config.PATHS['output_model']
    audio_extensions = tuple(config.DATASET['audio_extensions'])
    garbage_label = config.MODEL.get('garbage_label')

    if not os.path.exists(test_path) or not os.listdir(test_path):
        print(f"[ERROR] El directorio de prueba '{test_path}' no existe o está vacío. Ejecuta 'prepare_dataset.py' primero.")
        return
    if not os.path.exists(model_path):
        print(f"[ERROR] El modelo entrenado '{model_path}' no fue encontrado. Ejecuta 'train.py' primero.")
        return

    # 2. Cargar el modelo
    try:
        classifier = VQClassifier.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    # 3. Recolectar predicciones del conjunto de prueba
    true_labels = []
    predicted_labels = []
    
    test_labels = sorted([name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))])
    
    print("\nEvaluando archivos del dataset de prueba...")
    start_time = time.time()

    for label in tqdm(test_labels, desc="Evaluando palabras"):
        word_dir = os.path.join(test_path, label)
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(audio_extensions)]
        
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            
            prediction = classifier.predict(filepath, noise_profile=None)
            
            if prediction is None:
                prediction = garbage_label
            
            true_labels.append(label)
            predicted_labels.append(prediction)

    total_time = time.time() - start_time
    print(f"\nEvaluación completada en {total_time:.2f} segundos.")

    if not true_labels:
        print("\nNo se pudo evaluar ningún archivo. Finalizando.")
        return
    
    all_eval_labels = sorted(list(set(true_labels + predicted_labels)))
        
    # --- 4. REPORTE DE RENDIMIENTO GENERAL ---
    print("\n" + "="*50)
    print("1. REPORTE DE RENDIMIENTO GENERAL")
    print("="*50)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nPrecisión General (Accuracy): {accuracy:.2%}")
    print(f"Total de archivos evaluados: {len(true_labels)}")
    
    report = classification_report(
        true_labels, 
        predicted_labels, 
        labels=all_eval_labels, 
        zero_division=0,
        target_names=[l if l is not None else 'None' for l in all_eval_labels] # Nombres para el reporte
    )
    print("\nReporte de Clasificación por Palabra:")
    print(report)

    # --- 5. ANÁLISIS DE RECHAZO DE BASURA ---
    if garbage_label in true_labels:
        print("\n" + "="*50)
        print(f"2. ANÁLISIS DE RECHAZO DE '{garbage_label}'")
        print("="*50)
        
        garbage_true_indices = [i for i, true in enumerate(true_labels) if true == garbage_label]
        num_garbage_samples = len(garbage_true_indices)
        
        correctly_rejected = 0
        false_alarms = 0
        for i in garbage_true_indices:
            if predicted_labels[i] == garbage_label:
                correctly_rejected += 1
            else:
                false_alarms += 1
        
        if num_garbage_samples > 0:
            rejection_accuracy = correctly_rejected / num_garbage_samples
            print(f"Total de muestras de basura ('{garbage_label}'): {num_garbage_samples}")
            print(f"Rechazadas correctamente: {correctly_rejected} ({rejection_accuracy:.2%})")
            print(f"Alarmas falsas (basura reconocida como palabra): {false_alarms}")
        else:
            print(f"No se encontraron muestras de '{garbage_label}' en el set de prueba para evaluar el rechazo.")

    # --- 6. MATRIZ DE CONFUSIÓN VISUAL ---
    print("\n" + "="*50)
    print("3. MATRIZ DE CONFUSIÓN")
    print("="*50)
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_eval_labels)
    plot_confusion_matrix(cm, all_eval_labels)
    

if __name__ == '__main__':
    try:
        import pandas
        import seaborn
        import matplotlib
        import sklearn
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("\nEste script requiere algunas librerías adicionales. Por favor, instálalas con:")
        print("pip install pandas seaborn matplotlib scikit-learn tqdm")
        exit() # Salir si no están las dependencias
    
    main()