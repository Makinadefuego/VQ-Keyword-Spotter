# 🗣️ Clasificador de Palabras Clave por Cuantización Vectorial (VQ)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-%23FF4800.svg?style=for-the-badge&logo=librosa&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-%233A7EBF.svg?style=for-the-badge)

Este proyecto es un sistema de **reconocimiento de palabras clave aisladas** (Keyword Spotting) construido desde cero en Python. Utiliza técnicas clásicas de procesamiento de señales y un clasificador basado en **Cuantización Vectorial (VQ)** para identificar un vocabulario predefinido de palabras a partir de grabaciones de audio.

El sistema está diseñado para ser **robusto al ruido** y a la variabilidad del hablante, gracias al uso de un "Super-Vector" de características avanzadas y técnicas de aumento de datos. Incluye una interfaz gráfica moderna para el reconocimiento en tiempo real.

## ✨ Características Principales

-   **Clasificador VQ:** Utiliza K-Means para crear un "codebook" (huella acústica) único para cada palabra.
-   **Extracción de Características Avanzada:** Construye un "Super-Vector" por trama de audio, combinando:
    -   MFCCs (con Deltas y Delta-Deltas)
    -   GFCCs (alternativa robusta a los MFCCs)
    -   Energía, Pitch y Probabilidad de Voz
    -   Características Espectrales (Estándar y Avanzadas como curtosis, pendiente, rolloff, etc.)
-   **Robustez al Ruido:**
    -   **Aumento de Datos:** Genera automáticamente nuevas muestras de entrenamiento con ruido de fondo, cambios de tono y variaciones de velocidad.
    -   **Modelo de Basura:** Capacidad de entrenar un modelo `_garbage_` para rechazar activamente palabras desconocidas y ruidos.
-   **Interfaz Gráfica Moderna:** Una GUI intuitiva construida con `CustomTkinter` para el reconocimiento en tiempo real.
-   **Altamente Configurable:** Todos los parámetros del sistema (características, modelo, rutas, etc.) se gestionan centralmente en el archivo `config.py`.

## 🏛️ Estructura del Proyecto

```
.
├── VOICE/                # 1. Tus grabaciones de voz originales van aquí.
├── background_noises/    # 2. Archivos de ruido para el aumento de datos.
├── dataset/              # 3. Creado automáticamente (train/test sets).
│   ├── train/
│   └── test/
├── models/               # 4. El modelo entrenado (.joblib) se guarda aquí.
│
├── gui_recognizer.py     # ✅ La aplicación principal con interfaz gráfica.
├── train.py              # Script para entrenar el modelo.
├── augment_dataset.py    # Script para generar datos aumentados.
├── prepare_dataset.py    # Script para dividir el dataset.
├── evaulate.py           # Script para medir la precisión del modelo.
│
├── config.py             # Archivo de configuración central.
├── vq_classifier.py      # Lógica del clasificador VQ.
├── feature_extractor.py  # Lógica de extracción de características.
└── audio_utils.py        # Funciones auxiliares para cálculos de audio.
```

---

## 🚀 Guía de Inicio Rápido

### 1. Configuración del Entorno

**a) Clona el repositorio (si aplica):**
```bash
git clone https://github.com/Makinadefuego/VQ-Keyword-Spotter.git
cd tu_repositorio
```

**b) Instala todas las dependencias:**
Abre una terminal en la carpeta del proyecto y ejecuta:
```bash
pip install numpy librosa scipy scikit-learn joblib tqdm sounddevice soundfile audiomentations gammatone Pillow customtkinter
```

**c) Prepara las carpetas y recursos:**
-   Crea una carpeta llamada `VOICE`.
-   Crea una carpeta llamada `background_noises` y llénala con algunos archivos de ruido de fondo en formato `.wav` (ej. ruido de oficina, calle, cafetería).
-   Descarga un ícono de micrófono (`.png` con transparencia) y guárdalo como `mic_icon.png` en la raíz del proyecto.

### 2. Flujo de Trabajo

Sigue estos pasos en orden para entrenar y ejecutar tu propio reconocedor.

#### **Paso 1: Graba tus Palabras**

-   Graba cada palabra de tu vocabulario varias veces (se recomiendan más de 10 por palabra).
-   Guarda los archivos en la carpeta `./VOICE`.
-   **Nombra los archivos** de forma que la palabra clave esté al principio. Por ejemplo: `abrir_1.wav`, `abrir_user2.m4a`, `cerrar_intento3.wav`.

**Consejo Pro:** Para implementar el rechazo de palabras, graba sonidos que no pertenezcan a tu vocabulario (otras palabras, ruidos, toses) y nómbralos como `_garbage_1.wav`, `_garbage_2.wav`, etc.

#### **Paso 2: Prepara el Dataset**

Este script divide tus grabaciones en un conjunto de entrenamiento (80%) y uno de prueba (20%).
```bash
python prepare_dataset.py
```
Se creará la carpeta `./dataset` con los sets de `train` y `test`.

#### **Paso 3: Aumenta los Datos**

Este paso es crucial para la robustez. Genera nuevas muestras de audio con variaciones.
```bash
python augment_dataset.py
```
Tu carpeta `dataset/train` ahora contendrá muchas más muestras, etiquetadas con `_aug_`.

#### **Paso 4: Entrena el Modelo**

Ahora, el sistema aprenderá las "huellas acústicas" de cada palabra a partir del dataset de entrenamiento enriquecido.
```bash
python train.py
```
El modelo entrenado se guardará como `vq_robust_model.joblib` en la carpeta `./models`.

#### **Paso 5: ¡Ejecuta la Aplicación!**

Lanza la interfaz gráfica para el reconocimiento en tiempo real.
```bash
python gui.py
```
Haz clic en el ícono del micrófono, di una de tus palabras clave y observa el resultado. La consola te mostrará logs detallados de todo el proceso.

---

## 🛠️ Personalización y Uso Avanzado

### Evaluar el Rendimiento

Para medir qué tan bueno es tu modelo, ejecútalo contra el set de datos de prueba. Esto te dará la precisión y una matriz de confusión para ver dónde falla.
```bash
python evaulate.py
```

### Ajustar el Sistema (`config.py`)

El archivo `config.py` es el centro de control. Puedes experimentar cambiando:
-   **`FEATURES`**: Activa o desactiva diferentes tipos de características (`use_mfcc`, `use_gfcc`) para ver cómo impactan en la precisión.
-   **`MODEL['vq_clusters']`**: Aumenta este valor (ej. a 128) para un modelo más detallado (requiere más datos y tiempo), o redúcelo (ej. a 32) para un modelo más simple.
-   **`AUGMENTATION`**: Modifica la cantidad y la intensidad de las transformaciones de aumento de datos.
-   **`GUI['vad_aggressiveness']`**: En futuras versiones con VAD, este parámetro sería clave.

**Importante:** Después de cualquier cambio en `config.py` que afecte a las características o al modelo, **debes borrar el modelo antiguo y volver a entrenar** (`python train.py`).

### Reconocer un Único Archivo

Si solo quieres clasificar un archivo de audio específico desde la terminal, puedes usar el script `recognize.py` (si lo mantienes en el proyecto).
```bash
python recognize.py /ruta/a/tu/audio.wav
```

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

MDF, raya_cuadernos