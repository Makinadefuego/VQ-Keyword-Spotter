# gui.py (Completo, Final y Corregido)

import customtkinter as ctk
from PIL import Image
import sounddevice as sd
import numpy as np
import threading
import os
import time
import operator
import traceback

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa.display

import config
from vq_classifier import VQClassifier
try:
    from gmm_classifier import GMMClassifier
except ImportError:
    GMMClassifier = VQClassifier # Fallback por si no existe gmm_classifier

import audio_utils

# --- CONSTANTES DE CONFIGURACIÓN DE LA GUI ---
CHUNK_DURATION_S = 0.1
SAMPLE_RATE = config.AUDIO['sample_rate']
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
VISUALIZATION_BUFFER_S = 2.0
VISUALIZATION_BUFFER_SAMPLES = int(SAMPLE_RATE * VISUALIZATION_BUFFER_S)
NOISE_CALIBRATION_S = 2.0
NUM_PREDICTIONS_TO_SHOW = 4

class AudioLabApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(config.GUI["title"])
        self.geometry("800x750")
        self.minsize(700, 700)
        ctk.set_appearance_mode(config.GUI.get("appearance", "dark"))

        self.is_listening = False
        self.is_recording = False
        self.is_processing = False

        self.active_model_type = ctk.StringVar(value="gmm")
        self.classifier_vq = self.load_model('vq', VQClassifier)
        self.classifier_gmm = self.load_model('gmm', GMMClassifier)

        self.param_energy_threshold = ctk.DoubleVar(value=0.03)
        self.param_rejection_threshold = ctk.DoubleVar()
        self.param_noise_alpha = ctk.DoubleVar(value=config.AUDIO['spectral_subtraction']['alpha'])
        
        self.use_spectral_subtraction = ctk.BooleanVar(value=True)
        self.use_preemphasis = ctk.BooleanVar(value=True)
        self.use_trim = ctk.BooleanVar(value=True)
        
        self.recorded_frames = []
        self.silence_counter = 0
        self.vis_buffer = np.zeros(VISUALIZATION_BUFFER_SAMPLES, dtype=np.float32)
        self.noise_profile = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.listening_thread = None
        self.on_model_change()

    def load_model(self, model_type, ClassifierClass):
        path = config.PATHS.get(f"output_model_{model_type}")
        if not path:
            return None
        try:
            model = ClassifierClass.load_model(path)
            print(f"[GUI-INFO] Modelo {model_type.upper()} cargado exitosamente.")
            return model
        except Exception:
            print(f"[GUI-WARN] Archivo de modelo para {model_type.upper()} no encontrado en '{path}'.")
            return None

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_panel = ctk.CTkFrame(self, fg_color="transparent")
        main_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_panel.grid_rowconfigure(0, weight=3)
        main_panel.grid_rowconfigure(1, weight=1)

        side_panel = ctk.CTkFrame(self)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.create_visualizer_panel(main_panel)
        self.create_predictions_panel(main_panel)
        self.create_control_panel(side_panel)
        self.create_options_panel(side_panel)
        self.create_hyperparameters_panel(side_panel)

    def create_visualizer_panel(self, parent):
        spec_frame = ctk.CTkFrame(parent)
        spec_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.fig = Figure(facecolor="#2B2B2B")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=spec_frame)
        self.update_spectrogram_style()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.update_spectrogram()

    def create_predictions_panel(self, parent):
        results_panel = ctk.CTkFrame(parent)
        results_panel.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        results_panel.grid_columnconfigure(1, weight=1)
        self.prediction_widgets = []
        for i in range(NUM_PREDICTIONS_TO_SHOW):
            label = ctk.CTkLabel(results_panel, text="-", font=ctk.CTkFont(size=18, weight="bold"))
            label.grid(row=i, column=0, padx=10, pady=8, sticky="w")
            progress_bar = ctk.CTkProgressBar(results_panel, orientation="horizontal", height=18)
            progress_bar.set(0)
            progress_bar.grid(row=i, column=1, padx=10, pady=8, sticky="ew")
            self.prediction_widgets.append({'label': label, 'bar': progress_bar})

    def create_control_panel(self, parent):
        panel = ctk.CTkFrame(parent)
        panel.pack(fill="x", padx=10, pady=10)
        panel.grid_columnconfigure(0, weight=1)
        self.model_selector = ctk.CTkSegmentedButton(panel, values=["GMM", "VQ"], variable=self.active_model_type, command=self.on_model_change)
        self.model_selector.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        mic_icon = Image.open("mic_icon.png")
        self.mic_image = ctk.CTkImage(mic_icon, size=(48, 48))
        self.mic_button = ctk.CTkButton(panel, image=self.mic_image, text=" Escuchar", height=60, font=ctk.CTkFont(size=16, weight="bold"), command=self.toggle_listening)
        self.mic_button.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        self.status_label = ctk.CTkLabel(panel, text="Listo", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=2, column=0, pady=(5, 10), padx=10, sticky="ew")

    def create_options_panel(self, parent):
        panel = ctk.CTkFrame(parent)
        panel.pack(fill="x", padx=10, pady=5)
        panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(panel, text="Procesamiento", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        self.calibrate_button = ctk.CTkButton(panel, text="Calibrar Ruido", command=self.calibrate_noise)
        self.calibrate_button.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
        ctk.CTkSwitch(panel, text="Sustracción Espectral", variable=self.use_spectral_subtraction).grid(row=2, column=0, pady=5, padx=10, sticky="w")
        ctk.CTkSwitch(panel, text="Pre-énfasis", variable=self.use_preemphasis).grid(row=3, column=0, pady=5, padx=10, sticky="w")
        ctk.CTkSwitch(panel, text="Recortar Silencios", variable=self.use_trim).grid(row=4, column=0, pady=5, padx=10, sticky="w")

    def create_hyperparameters_panel(self, parent):
        panel = ctk.CTkFrame(parent)
        panel.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(panel, text="Hiperparámetros", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=(5,10))
        ctk.CTkLabel(panel, text="Umbral Energía (VAD)").grid(row=1, column=0, sticky="w", padx=10)
        ctk.CTkSlider(panel, from_=0.001, to=0.2, variable=self.param_energy_threshold).grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.rejection_label = ctk.CTkLabel(panel, text="Umbral Rechazo")
        self.rejection_label.grid(row=3, column=0, sticky="w", padx=10)
        self.rejection_slider = ctk.CTkSlider(panel, variable=self.param_rejection_threshold)
        self.rejection_slider.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkLabel(panel, text="Agresividad Ruido (Alpha)").grid(row=5, column=0, sticky="w", padx=10)
        ctk.CTkSlider(panel, from_=0.5, to=5.0, variable=self.param_noise_alpha).grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))

    def on_model_change(self, *args):
        model_type = self.active_model_type.get().lower()
        classifier = self.classifier_gmm if model_type == 'gmm' else self.classifier_vq
        if not classifier:
            self.mic_button.configure(state="disabled")
            self.status_label.configure(text=f"Modelo {model_type.upper()} no cargado", text_color="#FF5555")
        else:
            self.mic_button.configure(state="normal")
            self.status_label.configure(text="Listo", text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        
        if model_type == 'gmm':
            self.rejection_slider.configure(from_=-150, to=-30)
            self.param_rejection_threshold.set(config.MODEL.get("rejection_threshold_gmm", -80.0))
        else: # VQ
            self.rejection_slider.configure(from_=0.1, to=5.0) # Ampliado el rango por si acaso
            self.param_rejection_threshold.set(config.MODEL.get("rejection_threshold_vq", 1.5))

    def update_spectrogram_style(self):
        self.ax.set_facecolor("#1D1D1D")
        [s.set_edgecolor("#565B5E") for s in self.ax.spines.values()]
        self.ax.tick_params(colors='#DCE4EE', which='both')
        self.ax.set_xlabel("Tiempo (s)", color='#DCE4EE', fontsize=9)
        self.ax.set_ylabel("Frecuencia (Hz)", color='#DCE4EE', fontsize=9)
        self.fig.tight_layout(pad=2.0)

    def toggle_listening(self):
        if self.is_processing:
            return
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        self.mic_button.configure(text=" Detener")
        self.status_label.configure(text="Escuchando...", text_color="#33FFFF")
        [w['label'].configure(text="-", text_color="white") or w['bar'].set(0) for w in self.prediction_widgets]
        self.update_spectrogram()
        self.listening_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.listening_thread.start()

    def stop_listening(self):
        self.is_listening = False
        self.mic_button.configure(text=" Escuchar")
        self.status_label.configure(text="Listo", text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        self.after(10, self.update_spectrogram)

    def on_closing(self):
        self.is_listening = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=0.2)
        self.destroy()

    def calibrate_noise(self):
        if self.is_listening:
            return
        self.mic_button.configure(state="disabled")
        self.calibrate_button.configure(state="disabled", text="Calibrando...")
        threading.Thread(target=self._noise_calibration_thread, daemon=True).start()

    # --- FUNCIÓN CORREGIDA Y BLINDADA ---
    def _noise_calibration_thread(self):
        self.status_label.configure(text=f"Grabando ruido...")
        try:
            noise_clip = sd.rec(int(NOISE_CALIBRATION_S * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1).flatten()
            sd.wait()

            # Sanitización del audio grabado. Es CRÍTICO.
            if not np.all(np.isfinite(noise_clip)):
                print("[GUI-WARN] Se detectaron valores no finitos en el clip de ruido. Sanitizando.")
                noise_clip = np.nan_to_num(noise_clip, nan=0.0, posinf=0.0, neginf=0.0)

            # Comprobación de que el clip no sea silencio puro, lo que crearía un perfil inútil.
            if np.max(np.abs(noise_clip)) < 1e-5:
                print("[GUI-WARN] El clip de ruido grabado es prácticamente silencioso. Se usará un perfil de silencio.")
                self.noise_profile = None # Forzamos a no usar sustracción si el ruido es silencio
                self.after(0, lambda: self.use_spectral_subtraction.set(False)) # Desactivamos el switch en la GUI
                self.status_label.configure(text="Ruido es silencio. Sustracción desactivada.", text_color="#FFFF00")
            else:
                self.noise_profile = audio_utils.create_noise_profile(noise_clip, SAMPLE_RATE)
                self.status_label.configure(text="¡Perfil de ruido creado!", text_color="#33FFFF")
            
            self.after(10, self.update_spectrogram) # Actualizamos la visualización para mostrar el perfil
        except Exception as e:
            self.status_label.configure(text="Error en calibración", text_color="#FF5555")
            self.noise_profile = None
            print(f"[GUI-ERROR] Fallo en la calibración de ruido: {e}")
            traceback.print_exc()
        finally:
            self.mic_button.configure(state="normal")
            self.calibrate_button.configure(state="normal", text="Calibrar Ruido")
            self.after(2000, lambda: self.status_label.configure(text="Listo", text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]))

    def audio_loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES) as stream:
            while self.is_listening:
                if self.is_processing:
                    time.sleep(0.1)
                    continue
                audio_chunk, _ = stream.read(CHUNK_SAMPLES)
                self.vis_buffer = np.roll(self.vis_buffer, -CHUNK_SAMPLES)
                self.vis_buffer[-CHUNK_SAMPLES:] = audio_chunk[:, 0]
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                if not self.is_recording and rms > self.param_energy_threshold.get():
                    self.is_recording = True
                    self.recorded_frames = [audio_chunk]
                    self.silence_counter = 0
                    self.status_label.configure(text="Grabando...", text_color="#FFFF00")
                elif self.is_recording:
                    self.recorded_frames.append(audio_chunk)
                    self.silence_counter = self.silence_counter + 1 if rms < self.param_energy_threshold.get() else 0
                    
                    if self.silence_counter > config.GUI.get("silence_chunks_trigger", 15):
                        self.is_processing = True
                        self.status_label.configure(text="Procesando...", text_color="#FFA500")
                        threading.Thread(target=self.process_recording, args=(list(self.recorded_frames),)).start()
                        self.is_recording = False
                        self.recorded_frames = []

    def update_spectrogram(self):
        self.ax.clear()
        if self.is_listening:
            S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=self.vis_buffer, sr=SAMPLE_RATE, n_mels=64), ref=np.max)
            librosa.display.specshow(S_dB, sr=SAMPLE_RATE, ax=self.ax, x_axis='time', y_axis='mel', cmap='magma')
        elif self.noise_profile is not None and np.any(self.noise_profile):
            if np.all(np.isfinite(self.noise_profile)):
                self.ax.plot(self.noise_profile, color='#FF8C00')
                self.ax.set_title("Perfil de Ruido Ambiental", color='#DCE4EE', fontsize=12)
                max_val = np.max(self.noise_profile)
                if max_val > 0: self.ax.set_ylim([0, max_val * 1.2])
                self.ax.set_xlim(0, len(self.noise_profile))
            else:
                self.ax.set_title("Error: Perfil de Ruido Inválido", color='#FF5555', fontsize=12)
                self.ax.set_xticks([]); self.ax.set_yticks([])
        else:
            self.ax.set_title("Selecciona un modelo y pulsa 'Escuchar'", color='gray', fontsize=12)
            self.ax.set_xticks([]); self.ax.set_yticks([])
        
        self.update_spectrogram_style()
        self.canvas.draw()
        if self.is_listening:
            self.after(50, self.update_spectrogram)

    def process_recording(self, frames_to_process):
        try:
            model_type = self.active_model_type.get().lower()
            active_classifier = self.classifier_gmm if model_type == 'gmm' else self.classifier_vq
            if not active_classifier:
                self.after(0, self.show_recognition_error, "Modelo no cargado")
                return
            
            full_recording = np.concatenate(frames_to_process).flatten()
            
            processing_params = {
                'use_subtraction': self.use_spectral_subtraction.get(),
                'noise_profile': self.noise_profile,
                'noise_alpha': self.param_noise_alpha.get(),
                'use_preemphasis': self.use_preemphasis.get(),
                'use_trim': self.use_trim.get()
            }
            
            rejection_threshold = self.param_rejection_threshold.get()
            
            scores = active_classifier.predict_scores(full_recording, processing_params)

            if scores:
                is_gmm = (model_type == 'gmm')
                if is_gmm:
                    best_prediction = max(scores, key=scores.get)
                    best_score = scores[best_prediction]
                    final_prediction = best_prediction if best_score >= rejection_threshold else config.MODEL.get("garbage_label")
                else: # VQ
                    best_prediction = min(scores, key=scores.get)
                    best_score = scores[best_prediction]
                    final_prediction = best_prediction if best_score <= rejection_threshold else config.MODEL.get("garbage_label")
                
                if final_prediction == config.MODEL.get("garbage_label"):
                    self.after(0, self.show_recognition_error, "[RECHAZADO]")
                else:
                    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=is_gmm)
                    self.after(0, self.update_prediction_widgets, sorted_scores, is_gmm)
            else:
                self.after(0, self.show_recognition_error, "Predicción fallida")

        except Exception:
            print("\n" + "="*50)
            print("[GUI-PROC-FATAL] Ocurrió un error no manejado en el hilo de procesamiento:")
            traceback.print_exc()
            print("="*50 + "\n")
            self.after(0, self.show_recognition_error, "Error Interno")
        finally:
            self.is_processing = False
            if self.is_listening:
                self.status_label.configure(text="Escuchando...", text_color="#33FFFF")

    def update_prediction_widgets(self, sorted_scores, is_gmm):
        if not sorted_scores:
            return
        scores_vals = [s[1] for s in sorted_scores]
        min_score, max_score = min(scores_vals), max(scores_vals)
        for i in range(NUM_PREDICTIONS_TO_SHOW):
            if i < len(sorted_scores):
                label, score = sorted_scores[i]
                if (max_score - min_score) < 1e-9:
                    confidence = 1.0
                elif is_gmm:
                    confidence = (score - min_score) / (max_score - min_score)
                else: # VQ (lower is better)
                    confidence = 1.0 - ((score - min_score) / (max_score - min_score))
                
                color = "#33FF99" if i == 0 else "white"
                self.prediction_widgets[i]['label'].configure(text=f"{label.upper()}", text_color=color)
                self.prediction_widgets[i]['bar'].set(max(0, confidence))
            else:
                self.prediction_widgets[i]['label'].configure(text="-")
                self.prediction_widgets[i]['bar'].set(0)

    def show_recognition_error(self, message="Error"):
        self.prediction_widgets[0]['label'].configure(text=message, text_color="#FF5555")
        self.prediction_widgets[0]['bar'].set(0)
        for i in range(1, NUM_PREDICTIONS_TO_SHOW):
            self.prediction_widgets[i]['label'].configure(text="-")
            self.prediction_widgets[i]['bar'].set(0)

if __name__ == "__main__":
    app = AudioLabApp()
    app.mainloop()