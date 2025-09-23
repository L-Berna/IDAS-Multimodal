import os
import io
import re
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import lancedb
# Usaremos la búsqueda nativa de LanceDB en lugar de FAISS
import threading
import wave
import pyaudio
import requests
import tempfile
import pydub
from pydub import playback
import logging
from functools import lru_cache
try:
    # tqdm es opcional; se usa para mostrar barras de progreso y ETA
    from tqdm import tqdm
except Exception:
    tqdm = None

# New imports for improvements
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import difflib
from textblob import TextBlob
import librosa
from scipy.signal import wiener
import webrtcvad
from collections import deque
import json

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Modelos y cadenas de LLM y embeddings
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Clientes para voz: Whisper y ElevenLabs
import openai  # Se usará para Whisper (speech-to-text)
from elevenlabs.client import ElevenLabs

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
# Ruta a la base generada por el nuevo SimpleEmbeddings (image_vector/text_vector)
LANCEDB_PATH = r'C:\Users\krato\Documents\Documentos Uni\4° Semestre\Proyecto\Fase2-MMRAG\LanceDB_KIA-Simple_Mejorado'
VECTOR_DB_DIR = r'C:\Users\krato\Documents\Documentos Uni\4° Semestre\Proyecto\Fase1-RAG\vector_database_kia_sorento_3072'
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
# MÓDULO: QueryProcessor
###############################################################################
class QueryProcessor:
    """
    Clase encargada de procesar, mejorar y validar las consultas del usuario
    antes de enviarlas al sistema de recuperación.
    """
    def __init__(self):
        """
        Inicializa el procesador de consultas con diccionarios y herramientas de NLP.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
        self.conversation_context = deque(maxlen=5)  # Últimas 5 interacciones
        
        # Diccionario de corrección para términos automotrices
        self.automotive_corrections = {
            "frenos": ["freno", "break", "breik"],
            "motor": ["motos", "moto", "engin"],
            "aceite": ["aseit", "oil", "oyl"],
            "batería": ["bateria", "battery", "bateri"],
            "transmisión": ["transmision", "transmission", "transmishon"],
            "embrague": ["embrage", "clutch", "cloch"],
            "radiador": ["radiadors", "radiator", "rediator"],
            "filtro": ["filter", "filtros"],
            "bujías": ["bujias", "spark plugs", "spark plug"],
            "alternador": ["alternadors", "alternator"],
            "arranque": ["starter", "estarter"],
            "dirección": ["direccion", "steering", "stiring"],
            "suspensión": ["suspension", "shocks", "shock"],
            "neumáticos": ["neumaticos", "llantas", "tires", "tire"],
            "escape": ["exhaust", "exost"],
            "turbo": ["turbo", "turbos"],
            "diésel": ["diesel", "disel"],
            "gasolina": ["gasolin", "gas", "fuel"],
            "refrigerante": ["coolant", "colant"],
            "válvulas": ["valvulas", "valves", "valve"]
        }
        
        # Sinónimos para expansión de consultas
        self.automotive_synonyms = {
            "problema": ["falla", "error", "avería", "defecto", "mal funcionamiento"],
            "ruido": ["sonido", "sonidos", "ruidos", "noise"],
            "temperatura": ["calor", "calentamiento", "sobrecalentamiento"],
            "vibración": ["vibraciones", "temblor", "shake"],
            "pérdida": ["fuga", "goteo", "derrame"],
            "cambio": ["reemplazo", "sustitución", "reemplazar"],
            "revisar": ["verificar", "chequear", "inspeccionar", "examinar"],
            "reparar": ["arreglar", "componer", "fix"]
        }
        
        # Tipos de consulta para clasificación
        self.query_types = {
            "diagnostic": ["problema", "falla", "error", "ruido", "vibración", "síntoma"],
            "maintenance": ["cambio", "reemplazo", "mantenimiento", "revisar", "inspección"],
            "procedure": ["cómo", "procedimiento", "pasos", "instrucciones", "manual"],
            "specification": ["especificación", "medida", "capacidad", "torque", "presión"]
        }

        # Términos del dominio automotriz (para evitar falsos positivos de "consulta incompleta")
        domain_terms = set()
        for key, variants in self.automotive_corrections.items():
            domain_terms.add(key)
            domain_terms.update(variants)
        for key, variants in self.automotive_synonyms.items():
            domain_terms.add(key)
            domain_terms.update(variants)
        # Ampliación manual frecuente
        domain_terms.update({
            "llanta", "llantas", "neumático", "neumáticos", "rueda", "ruedas",
            "repuesto", "refacción", "refacciones", "gato", "maletero", "herramienta",
            "herramientas", "cruceta", "tuerca", "tuercas", "tornillo", "tornillos",
            "faro", "focos", "fusible", "fusibles", "manual", "kit", "compresor"
        })
        self.domain_terms = {t.lower() for t in domain_terms}

    def detect_voice_activity(self, audio_data, sample_rate=16000):
        """
        Detecta si hay actividad de voz en el audio usando WebRTC VAD.
        
        Args:
            audio_data (bytes): Datos de audio.
            sample_rate (int): Frecuencia de muestreo.
        
        Returns:
            bool: True si se detecta voz, False en caso contrario.
        """
        try:
            vad = webrtcvad.Vad(2)  # Agresividad media
            # Convertir a formato requerido por VAD
            frame_duration = 30  # ms
            frame_size = int(sample_rate * frame_duration / 1000)
            
            # Procesar en chunks
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_data), frame_size * 2):  # *2 para 16-bit
                frame = audio_data[i:i + frame_size * 2]
                if len(frame) == frame_size * 2:
                    total_frames += 1
                    if vad.is_speech(frame, sample_rate):
                        voice_frames += 1
            
            if total_frames == 0:
                return False
                
            voice_ratio = voice_frames / total_frames
            return voice_ratio > 0.3  # Al menos 30% debe ser voz
            
        except Exception as e:
            logger.warning("Error en detección de voz: %s", e)
            return True  # Asumimos que hay voz si hay error

    def improve_audio_quality(self, audio_file_path):
        """
        Mejora la calidad del audio aplicando filtros de ruido.
        
        Args:
            audio_file_path (str): Ruta del archivo de audio.
        
        Returns:
            str: Ruta del archivo mejorado.
        """
        try:
            # Cargar audio
            audio, sr = librosa.load(audio_file_path, sr=16000)
            
            # Aplicar filtro Wiener para reducir ruido
            filtered_audio = wiener(audio, noise=0.01)
            
            # Normalizar audio
            filtered_audio = librosa.util.normalize(filtered_audio)
            
            # Guardar audio mejorado
            improved_path = audio_file_path.replace('.wav', '_improved.wav')
            import soundfile as sf
            sf.write(improved_path, filtered_audio, sr)
            
            return improved_path
            
        except Exception as e:
            logger.warning("Error mejorando audio: %s", e)
            return audio_file_path

    def clean_transcribed_text(self, text):
        """
        Limpia el texto transcrito eliminando muletillas y normalizando.
        
        Args:
            text (str): Texto transcrito.
        
        Returns:
            str: Texto limpio.
        """
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Eliminar muletillas comunes en español
        muletillas = ["eh", "este", "esto", "bueno", "entonces", "o sea", "digamos", 
                     "como que", "verdad", "¿no?", "mmm", "ahh", "uhh"]
        
        for muletilla in muletillas:
            text = re.sub(r'\b' + re.escape(muletilla) + r'\b', '', text, flags=re.IGNORECASE)
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Eliminar signos de puntuación innecesarios al final
        text = text.rstrip('.,!?;:')
        
        return text

    def correct_automotive_terms(self, text):
        """
        Corrige errores comunes en términos automotrices.
        
        Args:
            text (str): Texto a corregir.
        
        Returns:
            str: Texto con términos corregidos.
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            best_match = word
            best_ratio = 0.8  # Umbral mínimo de similitud
            
            # Buscar correcciones en el diccionario automotriz
            for correct_term, variations in self.automotive_corrections.items():
                for variation in variations:
                    ratio = difflib.SequenceMatcher(None, word.lower(), variation.lower()).ratio()
                    if ratio > best_ratio:
                        best_match = correct_term
                        best_ratio = ratio
            
            corrected_words.append(best_match)
        
        return ' '.join(corrected_words)

    def expand_query_with_synonyms(self, text):
        """
        Expande la consulta añadiendo sinónimos relevantes.
        
        Args:
            text (str): Consulta original.
        
        Returns:
            str: Consulta expandida con sinónimos.
        """
        words = word_tokenize(text, language='spanish')
        expanded_terms = []
        
        for word in words:
            word_lower = word.lower()
            expanded_terms.append(word)
            
            # Buscar sinónimos
            for key, synonyms in self.automotive_synonyms.items():
                if word_lower == key:
                    expanded_terms.extend(synonyms[:2])  # Máximo 2 sinónimos
                    break
        
        return ' '.join(expanded_terms)

    def add_conversation_context(self, current_query):
        """
        Añade contexto de la conversación anterior a la consulta actual.
        
        Args:
            current_query (str): Consulta actual.
        
        Returns:
            str: Consulta con contexto añadido.
        """
        # Detectar referencias a respuestas anteriores
        references = ["eso", "esto", "lo anterior", "la anterior", "ese", "esa", "el mismo", "la misma"]
        
        has_reference = any(ref in current_query.lower() for ref in references)
        
        if has_reference and len(self.conversation_context) > 0:
            # Obtener el último tema mencionado
            last_context = self.conversation_context[-1]
            contextual_query = f"En relación a {last_context}, {current_query}"
            return contextual_query
        
        return current_query

    def classify_query_type(self, text):
        """
        Clasifica el tipo de consulta para optimizar la búsqueda.
        
        Args:
            text (str): Consulta del usuario.
        
        Returns:
            str: Tipo de consulta clasificada.
        """
        text_lower = text.lower()
        scores = {}
        
        for query_type, keywords in self.query_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[query_type] = score
        
        # Retornar el tipo con mayor puntuación
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "general"

    def detect_ambiguous_query(self, text):
        """
        Detecta si una consulta es demasiado ambigua o incompleta.
        
        Args:
            text (str): Consulta del usuario.
        
        Returns:
            tuple: (es_ambigua, sugerencia_clarificación)
        """
        # Criterios para detectar ambigüedad
        text_lower = text.lower()
        tokens = text_lower.split()
        word_count = len(tokens)

        # Si contiene términos automotrices claros, no la marcamos como ambigua
        domain_hits = any(term in text_lower for term in self.domain_terms)

        # Consulta muy corta: permitimos consultas de 2-3 palabras si hay términos del dominio
        if word_count < 3 and not domain_hits:
            return True, "Tu consulta es muy breve. ¿Podrías proporcionar más detalles sobre el problema o procedimiento?"

        # Consultas muy generales
        general_terms = ["carro", "auto", "vehículo", "vehiculo", "problema", "ayuda"]
        if any(term in text_lower for term in general_terms) and word_count < 5 and not domain_hits:
            return True, "Tu consulta es muy general. ¿Podrías especificar qué parte del vehículo o qué tipo de problema específico?"

        # Falta de verbos (indicativo de consulta incompleta): aplicar solo si no hay términos del dominio
        if not domain_hits:
            try:
                blob = TextBlob(text)
                verbs = [word for word, pos in blob.tags if pos.startswith('VB')]
            except Exception:
                verbs = []
            # Más permisivo: exigir > 3 palabras para marcar como incompleta por falta de verbo
            if len(verbs) == 0 and word_count > 3:
                return True, "Tu consulta parece incompleta. ¿Qué necesitas hacer o qué está ocurriendo con tu vehículo?"

        return False, ""

    def process_query(self, raw_query, conversation_history=None):
        """
        Procesa completamente una consulta aplicando todas las mejoras.
        
        Args:
            raw_query (str): Consulta sin procesar.
            conversation_history (list): Historial de conversación.
        
        Returns:
            dict: Diccionario con la consulta procesada y metadatos.
        """
        # Paso 1: Limpiar texto transcrito
        cleaned_query = self.clean_transcribed_text(raw_query)
        
        # Paso 2: Corregir términos automotrices
        corrected_query = self.correct_automotive_terms(cleaned_query)
        
        # Paso 3: Detectar ambigüedad
        is_ambiguous, clarification = self.detect_ambiguous_query(corrected_query)
        
        if is_ambiguous:
            return {
                "processed_query": corrected_query,
                "original_query": raw_query,
                "is_ambiguous": True,
                "clarification_needed": clarification,
                "query_type": "ambiguous"
            }
        
        # Paso 4: Añadir contexto de conversación
        contextual_query = self.add_conversation_context(corrected_query)
        
        # Paso 5: Expandir con sinónimos
        expanded_query = self.expand_query_with_synonyms(contextual_query)
        
        # Paso 6: Clasificar tipo de consulta
        query_type = self.classify_query_type(expanded_query)
        
        # Actualizar contexto de conversación
        self.conversation_context.append(corrected_query)
        
        return {
            "processed_query": expanded_query,
            "original_query": raw_query,
            "cleaned_query": cleaned_query,
            "corrected_query": corrected_query,
            "contextual_query": contextual_query,
            "is_ambiguous": False,
            "query_type": query_type,
            "processing_steps": {
                "cleaned": cleaned_query != raw_query,
                "corrected": corrected_query != cleaned_query,
                "contextualized": contextual_query != corrected_query,
                "expanded": expanded_query != contextual_query
            }
        }

###############################################################################
# MÓDULO: RetrievalManager
###############################################################################
class RetrievalManager:
    """
    Clase encargada de gestionar la recuperación de información utilizando:
      - Conexión a la base de datos LanceDB y creación/verificación de su índice nativo.
      - Inicialización de RetrievalQA (Chroma + LLM) para extraer contexto del manual.
      - Generación de embeddings de texto con BridgeTower y re-ranking mejorado.
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
        
        # Inicializar el procesador de consultas
        self.query_processor = QueryProcessor()
        
        self._connect_db()
        self._initialize_bridgetower()
        self._initialize_retrieval_qa()
        # Crear índice nativo en LanceDB para image_vector (búsqueda texto→imagen)
        self._ensure_lancedb_index()

    def _connect_db(self):
        """Conecta a la base de datos LanceDB y abre la tabla de embeddings."""
        try:
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table("embeddings")
        except Exception as e:
            logger.error("Error al conectar con la base de datos: %s", e)
            raise

    def _ensure_lancedb_index(self):
        """
        Asegura un índice nativo en LanceDB sobre la columna image_vector con métrica coseno.
        Esto permite búsquedas texto→imagen eficientes sin construir FAISS manualmente.
        """
        try:
            logger.info("Creando/verificando índice nativo en LanceDB (image_vector)...")
            # type puede ser ivf_pq o hnsw; ivf_pq es un buen balance para escala media
            self.table.create_index(
                column="image_vector",
                metric="cosine",
                type="ivf_pq",
                num_partitions=256,
                num_sub_vectors=64
            )
            logger.info("Índice LanceDB listo.")
        except Exception as e:
            logger.warning("No se pudo crear/verificar índice LanceDB: %s", e)

    def _initialize_retrieval_qa(self):
        """Inicializa el módulo RetrievalQA utilizando Chroma y un LLM para obtener contexto del manual."""
        # Reutilizamos un cliente LLM único para todo el flujo
        self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0, model_name=self.llm_model)
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_api_key)
        vectordb = Chroma(embedding_function=embedding_model, persist_directory=self.vector_db_dir)
        template = (
            "Eres IDAS, un asistente vehicular atento y amigable. Responde con precisión y sin rodeos. Sigue estas reglas:\n"
            "1. Máximo 3 puntos o 3 oraciones.\n"
            "2. Usa términos técnicos del manual cuando corresponda.\n"
            "3. Prioriza solo lo relevante a la pregunta.\n"
            "4. Incluye advertencias de seguridad solo si aplican.\n"
            "5. Sé claro y conciso.\n\n"
            "Contexto: {context}\n"
            "Pregunta: {question}\n"
            "Respuesta:\n"
        )
        qa_chain_prompt = PromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )

    def _initialize_bridgetower(self):
        """Carga el modelo BridgeTower y su procesador para generar embeddings multimodales."""
        # Cargar BridgeTower una sola vez para generar embeddings de texto de consulta
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
        Genera embedding SOLO de TEXTO para la consulta, normalizado (L2).
        La comparación se realiza contra image_vector almacenado en LanceDB.
        
        Args:
            query_text (str): El texto de la consulta.
        
        Returns:
            list: Embedding de texto normalizado.
        """
        # BridgeTower requiere imagen y texto; usamos una imagen mínima de relleno
        dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        encoding = self.processor(
            images=dummy_image,
            text=query_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model_bridgetower(**encoding)
            text_embeds = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=1)
        return text_embeds.cpu().numpy().flatten().astype('float32').tolist()

    # Nota: El re-ranking básico se conserva solo como respaldo en la copia del script.
    # En el flujo principal se utiliza el re-ranking mejorado (avanzado).

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
        response = self.llm(prompt)
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
            tuple: (mejor resultado candidato, respuesta generada en texto, información de procesamiento)
        """
        # Procesar la consulta con todas las mejoras
        processed_result = self.query_processor.process_query(query_text)
        
        # Verificar si la consulta es ambigua
        if processed_result["is_ambiguous"]:
            return None, processed_result["clarification_needed"], processed_result
        
        # Usar la consulta procesada para la búsqueda
        final_query = processed_result["processed_query"]
        query_type = processed_result["query_type"]
        
        logger.info("Consulta original: %s", query_text)
        logger.info("Consulta procesada: %s", final_query)
        logger.info("Tipo de consulta: %s", query_type)
        
        # Generar embedding de la consulta procesada (texto)
        query_embedding = self.generate_query_embedding(final_query)
        query_vector = np.array(query_embedding, dtype='float32')  # (d,)
        
        # Ajustar top_k según el tipo de consulta
        adjusted_top_k = top_k
        if query_type == "diagnostic":
            adjusted_top_k = min(top_k + 3, 10)  # Más candidatos para diagnósticos
        elif query_type == "procedure":
            adjusted_top_k = min(top_k + 2, 8)   # Más candidatos para procedimientos
        
        # Búsqueda nativa LanceDB sobre image_vector con métrica coseno
        logger.info("Buscando top-%d candidatos en LanceDB...", adjusted_top_k)
        results_df = self.table.search(query_vector.tolist(), vector_column_name="image_vector") \
            .metric("cosine").limit(adjusted_top_k).to_pandas()
        
        candidates = []
        if results_df is not None and len(results_df) > 0:
            # Re-ranking híbrido: combina similitud con image_vector y text_vector
            logger.info("Aplicando re-ranking híbrido (imagen/texto)...")
            # Extraer matrices de vectores
            img_vecs = np.vstack(results_df["image_vector"].to_list()).astype("float32")
            txt_vecs = np.vstack(results_df["text_vector"].to_list()).astype("float32")
            # Normalizar por seguridad (deberían venir normalizados)
            def l2_normalize(x):
                norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                return x / norms
            img_vecs = l2_normalize(img_vecs)
            txt_vecs = l2_normalize(txt_vecs)
            q = query_vector / (np.linalg.norm(query_vector) + 1e-12)
            # Similaridades coseno
            s1 = img_vecs @ q  # texto→imagen
            s2 = txt_vecs @ q  # texto→caption
            # Ponderación (ajustable)
            score = 0.8 * s1 + 0.2 * s2
            order = np.argsort(-score)
            # Construir lista de candidatos ordenados
            iterator = order
            if tqdm is not None:
                iterator = tqdm(order, desc="Re-rankeando", unit="item")
            for idx in iterator:
                row = results_df.iloc[int(idx)]
                candidates.append({
                    "imagen": row.get("imagen"),
                    "caption": row.get("texto", ""),
                    "score": float(score[int(idx)])
                })
        
        if not candidates:
            best_result = None
            image_caption = "No se encontró imagen relevante."
        else:
            # Re-ranking mejorado considerando el tipo de consulta
            best_idx = self.enhanced_re_ranking(final_query, candidates, query_type)
            best_result = candidates[best_idx]
            image_caption = best_result["caption"]
        
        # Obtener contexto del manual usando la consulta original y procesada
        model_response = self.qa_chain({"query": final_query})
        manual_context = model_response.get("result", "No se encontró contexto relevante.")
        # Extraer posibles páginas de referencia desde las fuentes
        reference_pages_list = []
        try:
            source_docs = model_response.get("source_documents", []) or []
            pages_set = set()
            for d in source_docs:
                meta = getattr(d, "metadata", {}) or {}
                page_val = meta.get("page") or meta.get("page_number") or meta.get("page_index")
                if isinstance(page_val, int):
                    pages_set.add(page_val)
            if pages_set:
                reference_pages_list = sorted(pages_set)
        except Exception:
            reference_pages_list = []
        
        # Generar respuesta final considerando el tipo de consulta
        final_prompt = self.create_enhanced_prompt(
            query_text, final_query, image_caption, manual_context, query_type, processed_result, reference_pages_list
        )
        
        response_generated = self.llm(final_prompt)
        
        response_text = response_generated.content if hasattr(response_generated, 'content') else str(response_generated)
        
        return best_result, response_text, processed_result

    # Re-ranking avanzado (principal)
    def enhanced_re_ranking(self, query, candidates, query_type):
        """
        Re-ranking mejorado que considera el tipo de consulta para mejor selección.
        
        Args:
            query (str): Consulta procesada.
            candidates (list): Lista de candidatos.
            query_type (str): Tipo de consulta clasificada.
        
        Returns:
            int: Índice del mejor candidato.
        """
        # Instrucciones específicas según el tipo de consulta
        type_instructions = {
            "diagnostic": "Prioriza resultados que muestren síntomas, problemas o fallas específicas.",
            "maintenance": "Prioriza resultados sobre mantenimiento, cambios de partes o inspecciones.",
            "procedure": "Prioriza resultados que muestren pasos, procedimientos o instrucciones.",
            "specification": "Prioriza resultados con datos técnicos, especificaciones o medidas.",
            "general": "Selecciona el resultado más relevante para la consulta general."
        }
        
        instruction = type_instructions.get(query_type, type_instructions["general"])
        
        prompt = f'''Como experto automotriz, evalúa estos resultados para la consulta:
"{query}"

Tipo de consulta: {query_type}
Instrucción específica: {instruction}

Resultados disponibles:
'''
        
        for i, cand in enumerate(candidates, start=1):
            prompt += f"{i}. {cand['caption']}\n"
        
        prompt += f"\nConsiderando el tipo de consulta ({query_type}) y la instrucción específica, elige el número del resultado que mejor responde a la consulta. Responde solo con el número."
        
        response = ChatOpenAI(
            openai_api_key=self.openai_api_key, 
            temperature=0, 
            model_name=self.llm_model
        )(prompt)
        
        response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        match = re.search(r'\d+', response_text)
        
        if match:
            try:
                best_number = int(match.group())
                if 1 <= best_number <= len(candidates):
                    return best_number - 1
            except Exception as e:
                logger.error("Error al convertir la respuesta de re-ranking: %s", e)
        
        logger.warning("Respuesta inesperada en re-ranking mejorado: %s", response_text)
        return 0

    def create_enhanced_prompt(self, original_query, processed_query, image_caption, manual_context, query_type, processing_info, reference_pages_list):
        """
        Crea un prompt mejorado considerando el tipo de consulta y el procesamiento realizado.
        
        Args:
            original_query (str): Consulta original del usuario.
            processed_query (str): Consulta procesada.
            image_caption (str): Descripción de la imagen.
            manual_context (str): Contexto del manual.
            query_type (str): Tipo de consulta.
            processing_info (dict): Información del procesamiento.
        
        Returns:
            str: Prompt optimizado para el LLM.
        """
        # Templates específicos según el tipo de consulta (respuestas naturales, sin viñetas ni encabezados)
        type_templates = {
            "diagnostic": """
Información Visual: {image_caption}
Contexto del Manual: {manual_context}
Consulta: {original_query}

Redacta una explicación breve (1–3 oraciones), directa y sin viñetas, indicando el diagnóstico probable y la(s) acción(es) inmediatas recomendadas. Menciona advertencias solo si aplican.
""",
            "maintenance": """
Información Visual: {image_caption}
Contexto del Manual: {manual_context}
Consulta: {original_query}

Responde en 1–3 oraciones, sin viñetas, indicando el componente, el procedimiento esencial y la frecuencia recomendada. Incluye precauciones solo si son críticas.
""",
            "procedure": """
Información Visual: {image_caption}
Contexto del Manual: {manual_context}
Consulta: {original_query}

Ofrece instrucciones claras en 1–3 oraciones, sin viñetas ni encabezados, priorizando los pasos clave y cómo verificar que quedó correcto.
""",
            "specification": """
Información Visual: {image_caption}
Contexto del Manual: {manual_context}
Consulta: {original_query}

Da el dato solicitado de forma precisa en 1–2 oraciones (valor exacto y tolerancias si aplican), sin viñetas.
""",
            "general": """
Información Visual: {image_caption}
Contexto del Manual: {manual_context}
Consulta: {original_query}

Responde de forma concisa en 1–3 oraciones, sin viñetas ni encabezados, enfocándote solo en lo más relevante a la pregunta.
"""
        }
        
        template = type_templates.get(query_type, type_templates["general"])
        
        rendered = template.format(
            image_caption=image_caption,
            manual_context=manual_context,
            original_query=original_query
        )

        # Anexar sugerencia de consulta al manual con páginas si están disponibles
        try:
            pages_note = ""
            if reference_pages_list:
                pages_joined = ", ".join(str(p) for p in reference_pages_list)
                pages_note = f" Para mayor detalle, consulte su manual vehicular (páginas {pages_joined})."
            else:
                pages_note = " Para mayor detalle, consulte su manual vehicular."
            rendered = rendered.strip() + pages_note
        except Exception:
            rendered = rendered.strip() + " Para mayor detalle, consulte su manual vehicular."

        return rendered

    # Nota: Se eliminó la precarga duplicada; BridgeTower se inicializa una sola vez en _initialize_bridgetower

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
        if self.voices and len(self.voices) > 25:
            self.voice_id = self.voices[25]["voice_id"]
            logger.info("Voz seleccionada: %s (ID: %s)", self.voices[25]["name"], self.voice_id)
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
            # Validar actividad de voz antes de transcribir
            with open(audio_filepath, "rb") as f:
                audio_data = f.read()
            
            # Crear instancia del procesador de consultas para verificar voz
            from pathlib import Path
            temp_processor = QueryProcessor()
            
            # Verificar si hay voz en el audio
            if not temp_processor.detect_voice_activity(audio_data):
                logger.warning("No se detectó actividad de voz en el audio")
                return ""
            
            # Mejorar calidad del audio antes de transcribir
            improved_audio_path = temp_processor.improve_audio_quality(audio_filepath)
            
            # Transcribir usando el audio mejorado
            with open(improved_audio_path, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="es"  # Especificar idioma español para mejor precisión
                )
            
            # Limpiar archivo temporal mejorado si es diferente del original
            if improved_audio_path != audio_filepath and os.path.exists(improved_audio_path):
                try:
                    os.remove(improved_audio_path)
                except:
                    pass
                    
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
            
            # Verificar si se obtuvo texto válido
            if not query_text.strip():
                self.append_history("Sistema: No se pudo transcribir el audio o no se detectó voz.")
                return
            
            self.append_history(f"Tú: {query_text}")
            
            # Verificar si está relacionado con vehículos
            is_vehicle_query = self.retrieval_manager.is_vehicle_related(query_text)
            
            if not is_vehicle_query:
                response_text = "Lo siento, solo puedo ayudarte con consultas relacionadas con vehículos y manuales automotrices. ¿Tienes alguna pregunta sobre tu automóvil?"
                self.append_history(f"Asistente: {response_text}")
                self.voice_manager.play_response_audio(response_text)
                return
            
            # Procesar la consulta con todas las mejoras
            best_result, response_text, processing_info = self.retrieval_manager.search_by_text(query_text)
            
            # Verificar si la consulta es ambigua y necesita clarificación
            if processing_info.get("is_ambiguous", False):
                clarification = processing_info.get("clarification_needed", "")
                self.append_history(f"Asistente: {clarification}")
                self.voice_manager.play_response_audio(clarification)
                return
            
            # Mostrar información de procesamiento si es relevante
            # Ocultar mensajes de procesamiento para una experiencia más limpia
            
            # Mostrar tipo de consulta detectado
            query_type = processing_info.get("query_type", "general")
            # Ocultar anuncio de clasificación de consulta en la UI
            
            # Mostrar respuesta
            self.append_history(f"Asistente: {response_text}")
            
            # Procesar y mostrar imagen si está disponible
            if best_result and best_result.get("imagen"):
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
            
            # Reproducir respuesta en audio
            self.voice_manager.play_response_audio(response_text)
            
        except Exception as e:
            error_message = f"Error procesando consulta: {str(e)}"
            self.append_history(f"Error: {error_message}")
            logger.error("Error en process_voice_query: %s", e)
            
        finally:
            # Restaurar estado del botón
            self.btn_listening.configure(text="Start\nListening")
            self.state = "waiting"
            
            # Eliminar archivos temporales
            temp_files = [audio_filepath]
            # Agregar posibles archivos mejorados
            improved_path = audio_filepath.replace('.wav', '_improved.wav')
            if os.path.exists(improved_path):
                temp_files.append(improved_path)
                
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"Archivo temporal eliminado: {temp_file}")
                except Exception as cleanup_err:
                    logger.error(f"Error al eliminar archivo temporal {temp_file}: {cleanup_err}")

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
