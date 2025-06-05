import os
import io
import re
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import lancedb
import faiss
import threading
import wave
import pyaudio
import asyncio
import requests
import tempfile
import pydub
from pydub import playback
import logging
from functools import lru_cache

# Modelos y cadenas de LLM y embeddings
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Clientes para voz: Whisper y ElevenLabs
import openai  # Se usará para Whisper (speech-to-text)
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

# Configuración de logging para registrar información y errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parámetros de configuración y API keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Debes establecer la variable de entorno OPENAI_API_KEY con tu clave de OpenAI.")

ELEVEN_API_KEY = os.environ.get('ELEVEN_LABS_KEY')
if not ELEVEN_API_KEY:
    raise ValueError("Debes establecer la variable de entorno ELEVEN_LABS_KEY con tu clave de ElevenLabs.")

# Rutas y configuraciones para bases de datos y modelos
LANCEDB_PATH = r'LanceDB_OPENAI'
VECTOR_DB_DIR = 'vector_database_chspark_1536'
LLM_MODEL = "gpt-4o-mini"  # Modelo del LLM para re-ranking y QA

# Parámetros de audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
TEMP_AUDIO_FILE = "prompt_recording.wav"

# Configurar dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Usando dispositivo: %s", device)

###############################################################################
# MÓDULO: RetrievalManager
###############################################################################
class RetrievalManager:
    """
    Clase encargada de gestionar la recuperación de información utilizando:
      - Conexión a la base de datos y construcción del índice FAISS.
      - Inicialización de RetrievalQA para extraer contexto del manual.
      - Generación y re-ranking de embeddings multimodales usando BridgeTower.
    """
    def __init__(self, device, db_path, vector_db_dir, llm_model, openai_api_key):
        """
        Inicializa los componentes necesarios para la búsqueda y recuperación de información.
        """
        self.device = device
        self.db_path = db_path
        self.vector_db_dir = vector_db_dir
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self.embedding_cache = {}
        self._connect_db()
        self._build_ann_index()
        self._initialize_retrieval_qa()
        self._initialize_bridgetower()
        self._preload_models()

    def _connect_db(self):
        """Conecta a la base de datos LanceDB y abre la tabla de embeddings."""
        try:
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table("embeddings")
        except Exception as e:
            logger.error("Error al conectar con la base de datos: %s", e)
            raise

    def _build_ann_index(self):
        """Construye el índice FAISS optimizado para búsqueda rápida."""
        data = self.table.to_pandas()
        embeddings = np.array(data['vector'].tolist(), dtype='float32')
        d = embeddings.shape[1]
        
        # Usar IndexHNSWFlat con parámetros optimizados
        self.index_ann = faiss.IndexHNSWFlat(d, 64)  # Aumentar M a 64 para mejor precisión
        self.index_ann.hnsw.efConstruction = 200  # Aumentar para mejor construcción
        self.index_ann.hnsw.efSearch = 100  # Aumentar para mejor búsqueda
        
        # Normalizar vectores antes de añadirlos
        faiss.normalize_L2(embeddings)  # ACTIVADO: Normalización solo en FAISS
        self.index_ann.add(embeddings)
        logger.info("Índice FAISS construido con %d vectores.", self.index_ann.ntotal)
        self.rows_data = data.to_dict('records')

    def _initialize_retrieval_qa(self):
        """Inicializa el módulo RetrievalQA utilizando Chroma y un LLM para obtener contexto del manual."""
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_api_key)
        vectordb = Chroma(embedding_function=embedding_model, persist_directory=self.vector_db_dir)
        template = (
            "You are IDAS, a vehicle expert assistant. Follow these rules:\n"
            "1. MAX 3 bullet points\n"
            "2. MAX 15 words per point\n"
            "3. Technical terms only\n"
            "4. No explanations\n"
            "5. Actionable steps\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:\n"
        )
        qa_chain_prompt = PromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0, model_name=self.llm_model),
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )

    def _initialize_bridgetower(self):
        """Carga el modelo BridgeTower y su procesador para generar embeddings multimodales."""
        self.processor = BridgeTowerProcessor.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir="./model_cache"
        )
        self.model_bridgetower = BridgeTowerForContrastiveLearning.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir="./model_cache"
        ).to(self.device)
        self.model_bridgetower.eval()

    @lru_cache(maxsize=1000)
    def generate_query_embedding(self, query_text):
        """
        Genera un embedding combinado (texto e imagen) para una consulta.
        
        Args:
            query_text (str): El texto de la consulta.
        
        Returns:
            list: Embedding combinado normalizado.
        """
        dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        encoding = self.processor(
            images=dummy_image,
            text=query_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model_bridgetower(**encoding)
            # image_embeds = torch.nn.functional.normalize(outputs.image_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
            # text_embeds = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            combined_embeds = 0.5 * text_embeds + 0.5 * image_embeds
            # combined_embeds = torch.nn.functional.normalize(combined_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
        return combined_embeds.cpu().numpy().flatten().astype('float32').tolist()

    def re_rank_candidates(self, query, candidates):
        """
        Utiliza un LLM para reordenar las candidaturas y determinar la que mejor responde a la consulta.
        
        Args:
            query (str): La consulta del usuario.
            candidates (list): Lista de candidatos con 'caption' y otros datos.
        
        Returns:
            int: El índice del candidato seleccionado.
        """
        prompt = f'Given the following query:\n"{query}"\n\nAnd considering these results:\n'
        for i, cand in enumerate(candidates, start=1):
            prompt += f"{i}. {cand['caption']}\n"
        prompt += "\nChoose the number of the result that best answers the query. Respond with just the number."
        response = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0, model_name=self.llm_model)(prompt)
        response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        match = re.search(r'\d+', response_text)
        if match:
            try:
                best_number = int(match.group())
                if 1 <= best_number <= len(candidates):
                    return best_number - 1
            except Exception as e:
                logger.error("Error al convertir la respuesta: %s", e)
        logger.warning("Respuesta inesperada en re-ranking: %s", response_text)
        return 0

    def is_vehicle_related(self, query_text):
        """
        Determina si una consulta está relacionada con vehículos o manuales técnicos.
        
        Args:
            query_text (str): Consulta del usuario.
        
        Returns:
            bool: True si la consulta es sobre vehículos, de lo contrario False.
        """
        prompt = f"""
Determine si la siguiente consulta está relacionada con vehículos, manuales de vehículos, 
o información técnica sobre automóviles.

Consulta: "{query_text}"

Responde únicamente con "SI" si está relacionada con vehículos o "NO" si no lo está.
"""
        response = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0, model_name=self.llm_model)(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response).strip()
        return "SI" in response_text.upper()

    def search_by_text(self, query_text, top_k=5):
        """
        Realiza una búsqueda utilizando la consulta del usuario. Combina el embedding de la consulta,
        re-ranking y el contexto del manual para generar una respuesta final.
        
        Args:
            query_text (str): Consulta del usuario.
            top_k (int): Número de candidatos a recuperar.
        
        Returns:
            tuple: (mejor resultado candidato, respuesta generada en texto)
        """
        query_embedding = self.generate_query_embedding(query_text)
        query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index_ann.search(query_vector, top_k)
        candidates = []
        for i in range(top_k):
            if indices[0, i] == -1:
                continue
            row = self.rows_data[indices[0, i]]
            candidates.append({
                "imagen": row["imagen"],
                "caption": row["texto"],
                "score": distances[0, i]
            })
        if not candidates:
            best_result = None
            image_caption = "No se encontró imagen relevante."
        else:
            best_idx = self.re_rank_candidates(query_text, candidates)
            best_result = candidates[best_idx]
            image_caption = best_result["caption"]
        model_response = self.qa_chain({"query": query_text})
        manual_context = model_response.get("result", "No se encontró contexto relevante.")
        final_prompt = f"""
You are a vehicle expert virtual assistant. Your responses must be extremely concise and structured.

IMPORTANT GUIDELINES:
1. Use bullet points or numbered lists
2. Keep each point to 1-2 sentences maximum
3. Focus only on essential information
4. Avoid unnecessary explanations
5. Be direct and actionable

Image Description:
{image_caption}

Manual Extract:
{manual_context}

Query:
{query_text}

Provide a structured, concise response following the guidelines above.
"""
        response_generated = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0, model_name=self.llm_model)(final_prompt)
        response_text = response_generated.content if hasattr(response_generated, 'content') else str(response_generated)
        return best_result, response_text

    def _preload_models(self):
        """Precarga modelos para reducir latencia."""
        self.processor = BridgeTowerProcessor.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir="./model_cache"
        )
        self.model_bridgetower = BridgeTowerForContrastiveLearning.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir="./model_cache"
        ).to(self.device)
        self.model_bridgetower.eval()

###############################################################################
# MÓDULO: VoiceManager
###############################################################################
class VoiceManager:
    """
    Clase encargada de gestionar la captura, transcripción y síntesis de voz.
    Utiliza PyAudio para grabar, Whisper de OpenAI para transcribir y ElevenLabs para generar audio.
    """
    def __init__(self, eleven_api_key, openai_api_key):
        """
        Inicializa el gestor de voz, configurando PyAudio, la API de ElevenLabs y obteniendo la lista de voces.
        """
        self.eleven_api_key = eleven_api_key
        self.openai_api_key = openai_api_key
        self.audio = pyaudio.PyAudio()
        self.rec_stream = None
        self.wav_file = None
        self.state = "waiting"
        self.voice_id = None
        self.eleven_client = ElevenLabs(api_key=self.eleven_api_key)
        self.voices = self.list_available_voices()
        if self.voices and len(self.voices) > 23:
            self.voice_id = self.voices[23]["voice_id"]
            logger.info("Voz seleccionada: %s (ID: %s)", self.voices[23]["name"], self.voice_id)
        else:
            logger.warning("No se encontraron voces disponibles. Verifique su API key de ElevenLabs.")

    def list_available_voices(self):
        """
        Recupera y devuelve la lista de voces disponibles desde la API de ElevenLabs.
        
        Returns:
            list: Lista de voces disponibles.
        """
        try:
            url = "https://api.elevenlabs.io/v1/voices"
            headers = {"xi-api-key": self.eleven_api_key}
            response = requests.get(url, headers=headers)
            voices = response.json().get("voices", [])
            logger.info("Voces disponibles en ElevenLabs: %d", len(voices))
            return voices
        except Exception as e:
            logger.error("Error al obtener voces: %s", e)
            return []

    def transcribe_audio(self, audio_filepath):
        """
        Transcribe el audio guardado en un archivo utilizando Whisper de OpenAI.
        
        Args:
            audio_filepath (str): Ruta del archivo de audio.
        
        Returns:
            str: Texto transcrito.
        """
        try:
            with open(audio_filepath, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text.strip()
        except Exception as e:
            logger.error("Error en la transcripción: %s", e)
            return ""

    def text_to_voice(self, text):
        """
        Convierte texto a voz utilizando la API de ElevenLabs y guarda el audio generado en un archivo temporal.
        
        Args:
            text (str): Texto a convertir en audio.
        
        Returns:
            str or None: Ruta del archivo temporal generado o None en caso de error.
        """
        if not self.voice_id:
            logger.warning("No hay una voz seleccionada. No se puede generar audio.")
            return None
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.eleven_api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code != 200:
                logger.error("Error en la API de ElevenLabs: %d - %s", response.status_code, response.text)
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                f.flush()
                temp_filename = f.name
            logger.info("Audio generado y guardado en: %s", temp_filename)
            return temp_filename
        except Exception as e:
            logger.error("Error al generar audio: %s", e)
            return None

    def play_audio(self, file_path):
        """
        Reproduce un archivo de audio utilizando pydub.
        
        Args:
            file_path (str): Ruta del archivo de audio a reproducir.
        """
        if not file_path:
            logger.warning("No hay archivo de audio para reproducir")
            return
        try:
            sound = pydub.AudioSegment.from_file(file_path, format="mp3")
            playback.play(sound)
            logger.info("Audio reproducido correctamente")
        except Exception as e:
            logger.error("Error al reproducir audio: %s", e)

    def play_response_audio(self, response_text):
        """
        Genera y reproduce la respuesta de audio a partir de un texto.
        
        Args:
            response_text (str): Texto a sintetizar en audio.
        """
        try:
            logger.info("Generando audio para respuesta...")
            audio_file = self.text_to_voice(response_text)
            if audio_file:
                logger.info("Reproduciendo respuesta de audio...")
                self.play_audio(audio_file)
            else:
                logger.warning("No se pudo generar el audio para la respuesta.")
        except Exception as e:
            logger.error("Error en la generación o reproducción de audio: %s", e)

    def initialize_wave_file(self, filename):
        """
        Inicializa y configura un archivo WAV para grabación.
        
        Args:
            filename (str): Nombre del archivo WAV a crear.
        
        Returns:
            wave.Wave_write or None: Objeto para escribir el archivo WAV o None en caso de error.
        """
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(SAMPLE_RATE)
            return wf
        except Exception as e:
            logger.error("Error al inicializar archivo WAV: %s", e)
            return None

    def audio_recording_callback(self, in_data, frame_count, time_info, status, wav_file):
        """
        Callback para la grabación de audio. Escribe los datos recibidos en el archivo WAV.
        
        Args:
            in_data: Datos de audio entrantes.
            frame_count: Número de frames.
            time_info: Información temporal.
            status: Estado de la grabación.
            wav_file: Archivo WAV abierto para escribir.
        
        Returns:
            tuple: Datos de audio y flag de continuación.
        """
        if wav_file is not None:
            wav_file.writeframes(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self, filename):
        """
        Inicia la grabación de audio y guarda los datos en un archivo WAV temporal.
        
        Args:
            filename (str): Ruta del archivo WAV temporal.
        
        Returns:
            tuple: (stream de grabación, objeto de archivo WAV) o (None, None) en caso de error.
        """
        self.state = "listening"
        wf = self.initialize_wave_file(filename)
        if wf is None:
            return None, None
        self.wav_file = wf
        try:
            self.rec_stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=lambda in_data, frame_count, time_info, status: self.audio_recording_callback(in_data, frame_count, time_info, status, self.wav_file)
            )
            self.rec_stream.start_stream()
            return self.rec_stream, self.wav_file
        except Exception as e:
            logger.error("Error al iniciar grabación: %s", e)
            return None, None

    def stop_recording(self):
        """
        Detiene la grabación de audio y cierra el stream y el archivo WAV.
        """
        self.state = "processing"
        try:
            if self.rec_stream is not None:
                self.rec_stream.stop_stream()
                self.rec_stream.close()
            if self.wav_file is not None:
                self.wav_file.close()
                self.wav_file = None
        except Exception as e:
            logger.error("Error al detener grabación: %s", e)

###############################################################################
# MÓDULO: VehicleAssistantUI
###############################################################################
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
        self.root.title("Asistente Multimodal y de Voz para Vehículos")
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
                       font=('Verdana', 32),
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
                               font=("Verdana", 16), 
                               height=15, 
                               width=80, 
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

###############################################################################
# MAIN: Inicialización de la aplicación
###############################################################################
if __name__ == "__main__":
    try:
        retrieval_manager = RetrievalManager(
            device=device,
            db_path=LANCEDB_PATH,
            vector_db_dir=VECTOR_DB_DIR,
            llm_model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        voice_manager = VoiceManager(
            eleven_api_key=ELEVEN_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        app_ui = VehicleAssistantUI(retrieval_manager, voice_manager)
    except Exception as e:
        logger.exception("Error al iniciar la aplicación: %s", e)
