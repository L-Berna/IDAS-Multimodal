"""
Configuration module for the Multimodal Vehicle Assistant

This module contains all configuration parameters, API keys, paths, and constants
used throughout the application.
"""

import os
import torch
import logging

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
LANCEDB_PATH = r'C:/Users/krato/Documents/Documentos Uni/3° Semestre/Proyecto/Version reciente/LanceDB_OPENAI'
VECTOR_DB_DIR = 'vector_database_chspark_1536'
LLM_MODEL = "gpt-4o-mini"  # Modelo del LLM para re-ranking y QA

# Parámetros de audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = None  # Se definirá en voice_manager.py como pyaudio.paInt16
CHANNELS = 1
TEMP_AUDIO_FILE = "prompt_recording.wav"

# Configurar dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Usando dispositivo: %s", device)

# Configuración del modelo BridgeTower
BRIDGETOWER_MODEL_NAME = "BridgeTower/bridgetower-large-itm-mlm-itc"
MODEL_CACHE_DIR = "./model_cache"

# Configuración de ElevenLabs
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"
VOICE_STABILITY = 0.5
VOICE_SIMILARITY_BOOST = 0.75

# Configuración de embeddings
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
LRU_CACHE_SIZE = 1000

# Configuración de FAISS
FAISS_M_PARAMETER = 64
FAISS_EF_CONSTRUCTION = 200
FAISS_EF_SEARCH = 100

# Configuración de UI
UI_TITLE = "Asistente Multimodal y de Voz para Vehículos"
UI_FONT_FAMILY = "Verdana"
UI_FONT_SIZE = 16
UI_BUTTON_FONT_SIZE = 32
UI_TEXT_HEIGHT = 15
UI_TEXT_WIDTH = 80 