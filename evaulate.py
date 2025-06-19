# evaluate.py

import os
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier

def plot_confusion_matrix(cm, labels, title='Matriz de Confusión'):
    """Función para visualizar la matriz de confusión."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def main():
    """
    Función principal para evaluar el rendimiento del modelo en el dataset de prueba.
    """
    print("--- Inicio de la Evaluación del Modelo ---")

    # 1. Verificar que el dataset de prueba y el modelo existan
    test_path = config.PATHS['dataset_test']
    model_path = config.PATHS['output_model']

    if not os.path.exists(test_path) or not os.listdir(test_path):
        print(f"[ERROR] El directorio de prueba '{test_path}' no existe o está vacío.")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] El modelo entrenado '{model_path}' no fue encontrado.")
        print("Por favor, ejecuta 'train.py' primero.")
        return

    # 2. Cargar el modelo
    try:
        classifier = VQClassifier.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    # 3. Preparar listas para almacenar las etiquetas verdaderas y las predicciones
    true_labels = []
    predicted_labels = []
    
    print("\nEvaluando archivos del dataset de prueba...")
    start_time = time.time()

    # 4. Iterar sobre el dataset de prueba
    for label in classifier.labels:
        word_dir = os.path.join(test_path, label)
        if not os.path.exists(word_dir):
            print(f"[Advertencia] No se encontró la carpeta de prueba para la palabra '{label}'. Se omitirá.")
            continue
            
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith('.m4a')]
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            
            # Realizar la predicción
            prediction = classifier.predict(filepath)
            
            if prediction:
                true_labels.append(label)
                predicted_labels.append(prediction)
                print(f"  Archivo: {filename:<25} | Verdadero: {label:<10} | Predicho: {prediction:<10} {'✓' if label == prediction else '✗'}")
            else:
                print(f"  No se pudo predecir para el archivo: {filename}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nEvaluación completada en {total_time:.2f} segundos.")
    
    # 5. Calcular y mostrar métricas de rendimiento
    if not true_labels:
        print("\nNo se pudo evaluar ningún archivo. Finalizando.")
        return
        
    # Precisión (Accuracy)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n--- Métricas de Rendimiento ---")
    print(f"Precisión (Accuracy): {accuracy:.2%}")
    print(f"Total de archivos evaluados: {len(true_labels)}")
    print(f"Aciertos: {int(accuracy * len(true_labels))}")
    print(f"Fallos: {len(true_labels) - int(accuracy * len(true_labels))}")

    # Matriz de Confusión
    cm = confusion_matrix(true_labels, predicted_labels, labels=classifier.labels)
    print("\nMatriz de Confusión:")
    print(classifier.labels)
    print(cm)
    
    # 6. Visualizar la matriz de confusión
    # Necesitarás 'matplotlib' y 'seaborn' instalados: pip install matplotlib seaborn
    try:
        plot_confusion_matrix(cm, classifier.labels)
    except ImportError:
        print("\nPara visualizar la matriz de confusión, instala matplotlib y seaborn:")
        print("pip install matplotlib seaborn")

if __name__ == '__main__':
    main()