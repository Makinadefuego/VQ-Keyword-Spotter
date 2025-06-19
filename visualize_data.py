# visualize_data.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import config
from vq_classifier import VQClassifier
import feature_extractor



def plot_feature_space(words_to_plot, train_path=config.PATHS['dataset_train'], model_path=config.PATHS['output_model']):
    """
    Extrae características y clusters (codebooks) de las palabras especificadas, 
    los proyecta a 2D usando PCA y los visualiza en un scatter plot.
    """
    print(f"Visualizando el espacio de características y clusters para: {', '.join(words_to_plot)}")
    
    # --- 1. Cargar el modelo para obtener los codebooks ---
    ### NUEVO ###
    classifier = None
    if os.path.exists(model_path):
        classifier = VQClassifier.load_model(model_path)
    else:
        print("[Advertencia] No se encontró un modelo entrenado. No se visualizarán los clusters.")

    # --- 2. Extraer características de los datos crudos ---
    features_list = []
    labels_list = []

    for word in  os.listdir(train_path):
        # ... (esta parte es idéntica a la anterior, la dejo por completitud)
        word_dir = os.path.join(train_path, word)
        if not os.path.exists(word_dir):
            print(f"[Advertencia] No se encontró la carpeta de entrenamiento para '{word}'. Se omitirá.")
            continue
        print(f"  Procesando datos crudos de '{word}'...")
        audio_files = [f for f in os.listdir(word_dir) if f.lower().endswith(('.m4a', '.wav'))]
        for filename in audio_files:
            filepath = os.path.join(word_dir, filename)
            try:
                signal, sr = feature_extractor.load_and_clean_audio(filepath)
                if len(signal) > 200:
                    features = feature_extractor.extract_features(signal, sr)
                    features_list.append(features)
                    labels_list.extend([word] * features.shape[0])
            except Exception as e:
                print(f"    Error procesando {filename}: {e}")

    if not features_list:
        print("No se pudieron extraer características para visualizar.")
        return

    all_features = np.vstack(features_list)
    
    # --- 3. Recopilar los centroides (clusters) de los codebooks ---
    ### NUEVO ###
    codebook_features_list = []
    codebook_labels_list = []
    if classifier:
        print("\nRecopilando clusters (centroides) de los codebooks...")
        for word in words_to_plot:
            if word in classifier.codebooks:
                codebook = classifier.codebooks[word]
                codebook_features_list.append(codebook)
                codebook_labels_list.extend([word] * codebook.shape[0])
            else:
                print(f"  [Advertencia] No se encontró codebook para '{word}' en el modelo.")

    # --- 4. Aplicar PCA a TODO junto (datos + clusters) ---
    # Es crucial aplicar PCA a todo junto para que estén en el mismo espacio proyectado.
    print("\nAplicando PCA para reducir a 2 dimensiones...")
    
    # Combinamos datos crudos y centroides para el ajuste de PCA
    combined_features_for_pca = all_features
    if codebook_features_list:
        all_codebook_features = np.vstack(codebook_features_list)
        combined_features_for_pca = np.vstack([all_features, all_codebook_features])

    pca = PCA(n_components=2)
    pca.fit(combined_features_for_pca) # Ajustamos PCA con todo
    
    # Transformamos los datos crudos y los centroides por separado
    features_2d = pca.transform(all_features)
    if codebook_features_list:
        codebook_2d = pca.transform(all_codebook_features)

    # --- 5. Creación del Gráfico ---
    plt.figure(figsize=(14, 12))
    
    # Graficar los datos crudos (puntos pequeños y semitransparentes)
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1], 
        hue=labels_list,
        palette='viridis',
        alpha=0.3, # Hacemos los puntos de datos muy transparentes
        s=10,      # Puntos pequeños
        legend='full'
    )
    
    # ### NUEVO ### - Superponer los clusters (puntos grandes y opacos)
    if codebook_features_list:
        # Usamos un marcador diferente (X) y un borde negro para los centroides
        sns.scatterplot(
            x=codebook_2d[:, 0],
            y=codebook_2d[:, 1],
            hue=codebook_labels_list,
            palette='viridis',
            marker='X',           # Marcador de "X" grande
            s=200,                # Tamaño grande
            edgecolor='black',    # Borde negro para que resalten
            linewidth=1.5,
            legend=False          # No necesitamos una segunda leyenda
        )

    plt.title('Espacio de Características (puntos) y Clusters del Codebook (X)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Palabra')
    plt.grid(True)
    
    output_path = 'feature_space_with_clusters.png'
    plt.savefig(output_path)
    print(f"\nVisualización del espacio con clusters guardada en '{output_path}'")
    plt.show()

# La función main() no necesita cambios, ya que ahora llama a la nueva versión de plot_feature_space.
# Si quieres, puedes limpiar la lógica de la opción 1 para que sea más simple, ya que
# ahora plot_feature_space maneja todo.

def main():
    """
    Menú principal para elegir qué visualización realizar.
    """
    print("--- Herramienta de Visualización de Datos y Modelos ---")
    print("1. Visualizar Espacio de Características y Clusters (PCA)")
    print("2. Visualizar Huella Acústica (Codebook) de una palabra")
    
    choice = input("Elige una opción (1 o 2): ")
    
    if choice == '1':
        print("\nElige 2 o 3 palabras para comparar (separadas por comas).")
        print("Ejemplo: abrir,cerrar,luz")
        words_str = input("Palabras: ")
        words = [w.strip().lower() for w in words_str.split(',')]
        if len(words) >= 2:
            plot_feature_space(words) # No se necesita pasar el modelo, lo carga internamente
        else:
            print("[ERROR] Debes introducir al menos 2 palabras.")
            
    elif choice == '2':
        model_path = config.PATHS['output_model']
        if not os.path.exists(model_path):
            print(f"[ERROR] No se encuentra el modelo en '{model_path}'. Entrena un modelo primero.")
            return
            
        print("\nElige una palabra del modelo para visualizar su codebook.")
        word_str = input("Palabra: ").strip().lower()
        if word_str:
            plot_codebook_fingerprint(model_path, word_str)
        else:
            print("[ERROR] Debes introducir una palabra.")
    else:
        print("Opción no válida.")

if __name__ == '__main__':
    main()