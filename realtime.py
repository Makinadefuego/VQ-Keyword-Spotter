# realtime_recognizer.py

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time

# Importamos nuestros módulos
import config
from vq_classifier import VQClassifier


# --- Parámetros para la escucha en tiempo real ---
SAMPLE_RATE = config.AUDIO['sample_rate']
CHUNK_DURATION = 0.05  # Duración de cada bloque de audio a procesar (en segundos)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
ENERGY_THRESHOLD = 0.03 # Umbral de energía (RMS) para detectar el inicio del habla. ¡AJUSTA ESTE VALOR!
SILENCE_CHUNKS_TRIGGER = 10 # Número de bloques de silencio consecutivos para detener la grabación

class RealTimeRecognizer:
    def __init__(self, model_path=config.PATHS['output_model']):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo entrenado '{model_path}' no fue encontrado. "
                                    "Por favor, ejecuta 'train.py' primero.")
        print("Cargando modelo...")
        self.classifier = VQClassifier.load_model(model_path)
        print("¡Modelo cargado! El sistema está listo.")
        
    def listen_and_predict(self):
        """
        Bucle principal que escucha continuamente el micrófono y realiza predicciones.
        """
        print("\n--- Sistema de Reconocimiento en Tiempo Real ---")
        print(f"Vocabulario: {', '.join(self.classifier.labels)}")
        print(f"Ajusta el UMBRAL DE ENERGÍA si no detecta bien tu voz (valor actual: {ENERGY_THRESHOLD})")
        print("\nDi una palabra del vocabulario cuando veas 'Escuchando...'.")

        recording = False
        recorded_frames = []
        silence_counter = 0

        # El stream escucha indefinidamente
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES) as stream:
            while True:
                try:
                    # Lee un bloque de audio del micrófono
                    audio_chunk, overflowed = stream.read(CHUNK_SAMPLES)
                    # if overflowed:
                        # print("¡Advertencia! Se perdieron muestras de audio (overflow).")

                    # Calcula la energía del bloque (Root Mean Square)
                    rms = np.sqrt(np.mean(audio_chunk**2))
                    
                    if not recording:
                        print(f"Escuchando... (Energía: {rms:.4f})", end='\r')
                        # Si se supera el umbral, empezamos a grabar
                        if rms > ENERGY_THRESHOLD:
                            print("\n¡Habla detectada! Grabando...")
                            recording = True
                            recorded_frames.append(audio_chunk)
                            silence_counter = 0
                    else: # Ya estamos grabando
                        recorded_frames.append(audio_chunk)
                        
                        # Si la energía baja del umbral, contamos como un bloque de silencio
                        if rms < ENERGY_THRESHOLD:
                            silence_counter += 1
                        else: # Si vuelve a haber sonido, reseteamos el contador
                            silence_counter = 0
                        
                        # Si hay suficientes bloques de silencio seguidos, terminamos la grabación
                        if silence_counter > SILENCE_CHUNKS_TRIGGER:
                            print("Fin del habla detectado. Procesando...")
                            
                            # Concatenar todos los frames grabados
                            full_recording = np.concatenate(recorded_frames)
                            
                            # --- Guardar temporalmente y predecir ---
                            # Guardamos el audio en un archivo temporal para que nuestro pipeline
                            # existente (que espera un path) pueda procesarlo.
                            temp_wav_path = "temp_recording.wav"
                            wav.write(temp_wav_path, SAMPLE_RATE, full_recording)
                            
                            # Realizar la predicción
                            start_time = time.time()
                            predicted_label = self.classifier.predict(temp_wav_path)
                            prediction_time = time.time() - start_time
                            
                            print("-" * 20)
                            if predicted_label:
                                print(f">>> Predicción: '{predicted_label.upper()}' ({prediction_time:.2f}s)")
                            else:
                                print(">>> No se pudo obtener una predicción.")
                            print("-" * 20)
                            
                            # Limpiar para la siguiente ronda
                            os.remove(temp_wav_path)
                            recording = False
                            recorded_frames = []
                            silence_counter = 0
                            
                            print("\nDi una palabra del vocabulario cuando veas 'Escuchando...'.")
                            
                except KeyboardInterrupt:
                    print("\nSaliendo del programa.")
                    break
                except Exception as e:
                    print(f"Ocurrió un error: {e}")
                    break

def main():
    try:
        recognizer = RealTimeRecognizer()
        recognizer.listen_and_predict()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al iniciar: {e}")

if __name__ == '__main__':
    main()