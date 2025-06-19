# gui_recognizer.py

import customtkinter as ctk
from PIL import Image
import sounddevice as sd
import numpy as np
import threading
import os
import time
import soundfile as sf

import config
from vq_classifier import VQClassifier

# --- PARÁMETROS DE LA GUI Y ESCUCHA EN TIEMPO REAL ---
ENERGY_THRESHOLD = 0.03       # <<<< ¡IMPORTANTE! Umbral de energía para detectar voz. AJUSTA ESTE VALOR.
SAMPLE_RATE = config.AUDIO['sample_rate']
CHUNK_DURATION_S = 0.1
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
SILENCE_CHUNKS_TRIGGER = config.GUI.get("silence_chunks_trigger", 10)
VERBOSITY = config.LOGGING.get("verbosity_level", 1)

class App(ctk.CTk):
    # --- El resto de la clase no cambia ---
    # (init, create_widgets, toggle, start/stop_listening, on_closing)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(config.GUI["title"])
        self.geometry(config.GUI["window_size"])
        ctk.set_appearance_mode(config.GUI["appearance"])
        self.is_listening = False
        self.is_recording = False # Añadimos este estado
        self.recorded_frames = []
        self.silence_counter = 0
        self.load_model()
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.listening_thread = None

    def load_model(self):
        try:
            self.classifier = VQClassifier.load_model(config.PATHS['output_model'])
            self.model_loaded = True
        except Exception as e:
            if VERBOSITY >= 0: print(f"[GUI-ERROR] No se pudo cargar el modelo: {e}")
            self.classifier = None
            self.model_loaded = False

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.status_label = ctk.CTkLabel(main_frame, text="Presiona para hablar", font=ctk.CTkFont(size=18))
        self.status_label.pack(pady=20)
        mic_image = ctk.CTkImage(Image.open("mic_icon.png"), size=(80, 80))
        self.mic_button = ctk.CTkButton(main_frame, image=mic_image, text="", width=120, height=120,
                                        fg_color="transparent", hover_color="#333333", command=self.toggle_listening)
        self.mic_button.pack(pady=20)
        result_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        result_frame.pack(pady=20, padx=20, fill="x")
        self.result_label = ctk.CTkLabel(result_frame, text="---", font=ctk.CTkFont(size=28, weight="bold"), text_color="#3399FF")
        self.result_label.pack(pady=15, padx=10)
        if not self.model_loaded:
            self.status_label.configure(text="ERROR: Modelo no encontrado.\nEjecuta train.py", text_color="red")
            self.mic_button.configure(state="disabled")

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        if VERBOSITY >= 1: print("\n[GUI-INFO] Hilo de escucha iniciado.")
        self.mic_button.configure(fg_color="#005f7e")
        self.status_label.configure(text="Escuchando...", text_color="cyan")
        self.result_label.configure(text="---")
        self.listening_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.listening_thread.start()

    def stop_listening(self):
        if VERBOSITY >= 1: print("[GUI-INFO] Hilo de escucha detenido.")
        self.is_listening = False
        self.is_recording = False
        self.mic_button.configure(fg_color="transparent")
        self.status_label.configure(text="Presiona para hablar")

    def on_closing(self):
        if VERBOSITY >= 1: print("[GUI-INFO] Cerrando aplicación...")
        self.is_listening = False
        if self.listening_thread and self.listening_thread.is_alive(): 
            self.listening_thread.join(timeout=0.2)
        self.destroy()

    def audio_loop(self):
        """Bucle de audio que usa detección por energía (RMS)."""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES) as stream:
            while self.is_listening:
                audio_chunk, _ = stream.read(CHUNK_SAMPLES)
                rms = np.sqrt(np.mean(audio_chunk**2))

                if VERBOSITY >= 2:
                    print(f"Energía: {rms:.4f} | Grabando: {self.is_recording}", end='\r')

                if not self.is_recording:
                    if rms > ENERGY_THRESHOLD:
                        if VERBOSITY >= 1: print("\n[GUI-INFO] ¡Habla detectada! Iniciando grabación.")
                        self.is_recording = True
                        self.recorded_frames = [audio_chunk]
                        self.silence_counter = 0
                        self.status_label.configure(text="Grabando...", text_color="yellow")
                else: # Ya estamos grabando
                    self.recorded_frames.append(audio_chunk)
                    
                    if rms < ENERGY_THRESHOLD:
                        self.silence_counter += 1
                    else:
                        self.silence_counter = 0
                    
                    if self.silence_counter > SILENCE_CHUNKS_TRIGGER:
                        if VERBOSITY >= 1: print("\n[GUI-INFO] Fin del habla detectado. Procesando...")
                        
                        frames_to_process = list(self.recorded_frames)
                        threading.Thread(target=self.process_recording, args=(frames_to_process,)).start()
                        
                        self.is_recording = False
                        self.recorded_frames = []

    def process_recording(self, frames_to_process):
        """Procesa el audio grabado y realiza la predicción."""
        if not frames_to_process:
            if VERBOSITY >= 1: print("[GUI-PROC-WARN] Se intentó procesar una grabación vacía.")
            return

        self.status_label.configure(text="Procesando...", text_color="orange")
        
        full_recording = np.concatenate(frames_to_process)
        temp_wav_path = "temp_recording.wav"
        
        sf.write(temp_wav_path, full_recording, SAMPLE_RATE)
        
        if VERBOSITY >= 1:
            print(f"[GUI-PROC] Audio temporal guardado en '{temp_wav_path}' ({len(full_recording)/SAMPLE_RATE:.2f}s).")
            
        prediction = self.classifier.predict(temp_wav_path)
        
        if prediction:
            if prediction == config.MODEL.get("garbage_label"):
                if VERBOSITY >= 1: print(f"[GUI-PROC] Resultado: Basura/Ruido detectado.")
                self.result_label.configure(text="[Ruido/Desconocido]", text_color="gray")
            else:
                if VERBOSITY >= 1: print(f"[GUI-PROC] Resultado: '{prediction.upper()}'")
                self.result_label.configure(text=prediction.upper(), text_color="#33FF99")
        else:
            if VERBOSITY >= 1: print("[GUI-PROC] Resultado: No reconocido.")
            self.result_label.configure(text="No reconocido", text_color="red")
            
        os.remove(temp_wav_path)
        self.status_label.configure(text="Escuchando...", text_color="cyan")


if __name__ == "__main__":
    app = App()
    app.mainloop()