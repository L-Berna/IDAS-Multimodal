"""
Retrieval Manager Module for Multimodal Vehicle Assistant

This module handles all information retrieval operations including:
- LanceDB connection and FAISS index construction
- RetrievalQA initialization with Chroma
- BridgeTower multimodal embeddings generation
- Search and re-ranking operations
- Vehicle-related query filtering
"""

import re
import torch
import numpy as np
import lancedb
import faiss
from functools import lru_cache
from PIL import Image
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from config import (
    logger, device, LANCEDB_PATH, VECTOR_DB_DIR, LLM_MODEL, OPENAI_API_KEY,
    BRIDGETOWER_MODEL_NAME, MODEL_CACHE_DIR, EMBEDDING_MODEL_NAME,
    LRU_CACHE_SIZE, FAISS_M_PARAMETER, FAISS_EF_CONSTRUCTION, FAISS_EF_SEARCH
)


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
        self.index_ann = faiss.IndexHNSWFlat(d, FAISS_M_PARAMETER)  # Aumentar M a 64 para mejor precisión
        self.index_ann.hnsw.efConstruction = FAISS_EF_CONSTRUCTION  # Aumentar para mejor construcción
        self.index_ann.hnsw.efSearch = FAISS_EF_SEARCH  # Aumentar para mejor búsqueda
        
        # Normalizar vectores antes de añadirlos
        faiss.normalize_L2(embeddings)  # ACTIVADO: Normalización solo en FAISS
        self.index_ann.add(embeddings)
        logger.info("Índice FAISS construido con %d vectores.", self.index_ann.ntotal)
        self.rows_data = data.to_dict('records')

    def _initialize_retrieval_qa(self):
        """Inicializa el módulo RetrievalQA utilizando Chroma y un LLM para obtener contexto del manual."""
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=self.openai_api_key)
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
        """
        Carga el modelo BridgeTower y su procesador para generar embeddings multimodales.
        Los modelos se cargan con cache para reducir latencia en inicializaciones futuras.
        """
        self.processor = BridgeTowerProcessor.from_pretrained(
            BRIDGETOWER_MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR
        )
        self.model_bridgetower = BridgeTowerForContrastiveLearning.from_pretrained(
            BRIDGETOWER_MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR
        ).to(self.device)
        self.model_bridgetower.eval()

    @lru_cache(maxsize=LRU_CACHE_SIZE)
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
            combined_embeds = 0.75 * text_embeds + 0.3 * image_embeds
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