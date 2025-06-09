"""
Vehicle Assistant UI Module for Multimodal Vehicle Assistant

This module handles the graphical user interface and coordinates interactions
between the voice and retrieval management modules.
"""

import os
import io
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk

from config import (
    logger, TEMP_AUDIO_FILE, UI_TITLE, UI_FONT_FAMILY, UI_FONT_SIZE,
    UI_BUTTON_FONT_SIZE, UI_TEXT_HEIGHT, UI_TEXT_WIDTH
)


class VehicleAssistantUI:
    """
    Clase encargada de gestionar la interfaz gráfica (Tkinter) y coordinar la interacción
    entre los módulos de voz y recuperación de información.
    """
    def __init__(self, retrieval_manager, voice_manager):
        """
        Inicializa la UI y define las variables y estados necesarios.
        
        Args:
            retrieval_manager (RetrievalManager): Instancia para gestionar la búsqueda.
            voice_manager (VoiceManager): Instancia para gestionar la grabación y síntesis de voz.
        """
        self.retrieval_manager = retrieval_manager
        self.voice_manager = voice_manager
        self.conversation_history = []
        self.state = "waiting"
        self.temp_audio_file = TEMP_AUDIO_FILE
        self.create_gui()

    def append_history(self, text):
        """
        Agrega un mensaje al historial de conversación mostrado en la interfaz.
        
        Args:
            text (str): Mensaje a agregar.
        """
        self.conversation_history.append(text)
        self.text_box.config(state='normal')
        self.text_box.insert(tk.END, text + "\n")
        self.text_box.see(tk.END)
        self.text_box.config(state='disabled')

    def start_listening(self):
        """
        Inicia el proceso de grabación y actualiza la interfaz para reflejar el estado de escucha.
        """
        self.state = "listening"
        self.btn_listening.configure(text="Stop\nListening")
        self.voice_manager.start_recording(self.temp_audio_file)
        self.append_history("Sistema: Escuchando...")

    def stop_listening_and_process(self):
        """
        Detiene la grabación y lanza en un hilo separado el procesamiento de la consulta de voz.
        """
        self.state = "processing"
        self.btn_listening.configure(text="Processing...")
        self.voice_manager.stop_recording()
        threading.Thread(target=self.process_voice_query, args=(self.temp_audio_file,), daemon=True).start()

    def process_voice_query(self, audio_filepath):
        """
        Procesa la consulta de voz:
          - Transcribe el audio.
          - Verifica si la consulta está relacionada con vehículos.
          - Realiza la búsqueda y genera una respuesta.
          - Actualiza la interfaz con el historial, imagen y reproduce la respuesta en audio.
        
        Args:
            audio_filepath (str): Ruta del archivo de audio grabado.
        """
        try:
            query_text = self.voice_manager.transcribe_audio(audio_filepath)
            self.append_history(f"Tú: {query_text}")
            is_vehicle_query = self.retrieval_manager.is_vehicle_related(query_text)
            best_result, response_text = self.retrieval_manager.search_by_text(query_text)
            self.append_history(f"Asistente: {response_text}")
            if is_vehicle_query and best_result and best_result.get("imagen"):
                try:
                    image_bytes = best_result["imagen"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image.thumbnail((400, 400))
                    tk_image = ImageTk.PhotoImage(image)
                    self.image_label.config(image=tk_image)
                    self.image_label.image = tk_image
                except Exception as img_err:
                    logger.error("Error al procesar imagen: %s", img_err)
                    self.image_label.config(image="")
                    self.image_label.image = None
            else:
                self.image_label.config(image="")
                self.image_label.image = None
            self.voice_manager.play_response_audio(response_text)
        except Exception as e:
            self.append_history(f"Error: {e}")
        finally:
            self.btn_listening.configure(text="Start\nListening")
            self.state = "waiting"
            # Elimina el archivo de audio temporal
            try:
                if os.path.exists(audio_filepath):
                    os.remove(audio_filepath)
                    logger.info("Archivo de audio temporal eliminado.")
            except Exception as cleanup_err:
                logger.error("Error al eliminar archivo temporal: %s", cleanup_err)

    def toggle_listening(self):
        """
        Alterna entre iniciar y detener la grabación según el estado actual.
        """
        if self.state == "waiting":
            self.start_listening()
        elif self.state == "listening":
            self.stop_listening_and_process()

    def new_session(self):
        """
        Reinicia la sesión de conversación, limpiando el historial y la imagen.
        """
        self.conversation_history = []
        self.text_box.delete("1.0", tk.END)
        self.image_label.config(image="")
        self.image_label.image = None

    def create_gui(self):
        """
        Crea y configura la interfaz gráfica usando Tkinter.
        """
        self.root = tk.Tk()
        self.root.title(UI_TITLE)
        # Configura la ventana para ocupar la pantalla completa
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Frame para botones de acción
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        # Configurar estilo para los botones
        style = ttk.Style()
        style.configure('Custom.TButton', 
                       font=(UI_FONT_FAMILY, UI_BUTTON_FONT_SIZE),
                       padding=10)

        # Botón New Chat con texto ajustado
        btn_new = ttk.Button(btn_frame, 
                            text="New\nChat", 
                            style='Custom.TButton',
                            command=self.new_session)
        btn_new.pack(side=tk.LEFT, padx=20)

        # Botón Start Listening con texto ajustado
        self.btn_listening = ttk.Button(btn_frame, 
                                      text="Start\nListening", 
                                      style='Custom.TButton',
                                      command=self.toggle_listening)
        self.btn_listening.pack(side=tk.LEFT, padx=20)

        # Label para mostrar imágenes (resultado)
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Área de texto para mostrar el historial de conversación
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=20)
        self.text_box = tk.Text(text_frame, 
                               font=(UI_FONT_FAMILY, UI_FONT_SIZE), 
                               height=UI_TEXT_HEIGHT, 
                               width=UI_TEXT_WIDTH, 
                               wrap="word")
        self.text_box.pack(side=tk.LEFT)
        scrollbar = tk.Scrollbar(text_frame, command=self.text_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.config(yscrollcommand=scrollbar.set, state='disabled')

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """
        Maneja el cierre de la aplicación cerrando streams, archivos y terminando PyAudio.
        """
        try:
            if self.voice_manager.rec_stream is not None and self.voice_manager.rec_stream.is_active():
                self.voice_manager.rec_stream.stop_stream()
                self.voice_manager.rec_stream.close()
        except Exception as e:
            logger.error("Error al cerrar stream de audio: %s", e)
        try:
            if self.voice_manager.wav_file is not None:
                self.voice_manager.wav_file.close()
        except Exception as e:
            logger.error("Error al cerrar archivo WAV: %s", e)
        try:
            self.voice_manager.audio.terminate()
        except Exception as e:
            logger.error("Error al terminar PyAudio: %s", e)
        self.root.destroy() 