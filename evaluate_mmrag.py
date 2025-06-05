import os
import numpy as np
import torch
import logging
import unicodedata

# Importa tu clase RetrievalManager.
from RetrieverMMRAGVOZORI import RetrievalManager  # Asegúrate de que el nombre del módulo sea el correcto


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parámetros (usa los mismos que en tu código)
LANCEDB_PATH = r'LanceDB_Simple'
VECTOR_DB_DIR = 'vector_database_chspark_1536'
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Debes establecer la variable de entorno OPENAI_API_KEY con tu clave de OpenAI.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializa el RetrievalManager
retrieval_manager = RetrievalManager(
    device=device,
    db_path=LANCEDB_PATH,
    vector_db_dir=VECTOR_DB_DIR,
    llm_model=LLM_MODEL,
    openai_api_key=OPENAI_API_KEY
)

def get_candidates(rm, query_text, top_k=5):
    """
    Obtiene la lista de candidatos (documentos) a partir de la consulta,
    usando el ranking inicial proporcionado por FAISS.
    """
    query_embedding = rm.generate_query_embedding(query_text)
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = rm.index_ann.search(query_vector, top_k)
    candidate_list = []
    for i in range(top_k):
        if indices[0, i] == -1:
            continue
        row = rm.rows_data[indices[0, i]]
        candidate_list.append({
            "imagen": row["imagen"],
            "caption": row["texto"],
            "score": distances[0, i],
            "rank": i + 1
        })
    return candidate_list

def remove_accents(text):
    """
    Elimina acentos y diacríticos de un texto para facilitar comparaciones.
    """
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

# Conjunto de datos de evaluación:
# Cada entrada contiene una consulta y la palabra clave que se espera encontrar en el caption correcto.
evaluation_data = [
    {"query": "¿Como debo de ajustar de manera correcta mi cinturón de seguridad?", "expected_keyword": "Cinturón de seguridad"},
    {"query": "¿Como activo los limpiaparabrisas?", "expected_keyword": "limpiaparabrisas"},
    {"query": "¿Cómo sé si tengo el suficiente aceite en mi carro usando la varilla de nivel?", "expected_keyword": "varilla"},
    {"query": "¿Como puedo ajustar el asiento delantero para que este mas adelante o atras?", "expected_keyword": "asiento"},
    {"query": "¿Como es la distribucion del aire acondicionado dentro del carro?", "expected_keyword": "aire"}, 
    {"query": "¿Cómo funciona el Sistema de Seguridad Infantil?", "expected_keyword": "seguridad"},
    {"query": "¿Dónde se encuentra ubicado el claxon?", "expected_keyword": "Claxon"},
    {"query": "¿Como debería de rotar mis llantas para garantizar un desgaste uniforme?", "expected_keyword": "llantas"},               
    {"query": "¿Como se ve un neumático excesivamente inflado?", "expected_keyword": "sobreinflado"},                                   
    {"query": "¿Con que botón puedo activar las luces de emergencia de mi vehículo?", "expected_keyword": "luces"},
    #{"query": "¿?", "expected_keyword": ""},

    #{"query": "¿Qué se debe consultar para conocer la distancia de separación correcta?", "expected_keyword": "distancia de separacion"},
    #{"query": "¿Qué tipo de llave se recomienda para aflojar las tuercas de las ruedas?", "expected_keyword": "llave"},
    #{"query": "¿Qué materiales se suelen utilizar para fabricar la tapa del depósito de combustible?", "expected_keyword": "tapa del deposito"},
    #{"query": "¿La guantera puede afectar la ergonomía del vehículo?", "expected_keyword": "guantera"},
    #{"query": "¿Qué materiales son comunes en las cerraduras de maleteros?", "expected_keyword": "material"},
    #{"query": "¿Se pueden encontrar interruptores de ventanas eléctricas en vehículos antiguos?", "expected_keyword": "interruptores"},
    #{"query": "¿Qué otras luces de advertencia pueden estar relacionadas con el sistema de frenos?", "expected_keyword": "luces"},
    #{"query": "¿Qué medidas de seguridad se destacan en la imagen?", "expected_keyword": "medidas"},
    #{"query": "¿Qué acción inmediata debería tomar un conductor al ver el icono de batería encendido?", "expected_keyword": "accion"},
    #{"query": "¿Es seguro continuar conduciendo si se enciende el icono de advertencia de aceite?", "expected_keyword": "icono aceite"},

    #{"query": "¿El vehículo cuenta con llanta de repuesto?", "expected_keyword": "repuesto"},
    #{"query": "¿Como enciendo el aire acondicionado?", "expected_keyword": "aire acondicionado"},
    #{"query": "¿En que parte activo el claxon?", "expected_keyword": "claxon"},
    #{"query": "¿El carro cuenta con guantera?", "expected_keyword": "guantera"},
    #{"query": "¿Com oinstalo un asiento de bebe en el carro?", "expected_keyword": "asiento de bebe"},
    #{"query": "¿Como activo las luces exteriores del carro?", "expected_keyword": "luces"},
    #{"query": "¿Que tipo de aceite usa el motor?", "expected_keyword": "aceite"},
    #{"query": "¿Donde se enceuntra ubicado el filtro de polvo?", "expected_keyword": "filtro"},
    #{"query": "¿Donde puedo llenar el agua para los limpiaparabrisas?", "expected_keyword": "limpiaparabrisas"},
    #{"query": "¿Como activo las luces de emergencia?", "expected_keyword": "luces"},

    # Puedes agregar más ejemplos según tu caso
]

def evaluate(evaluation_data, rm, top_k=5):
    """
    Ejecuta la evaluación comparando el ranking inicial (FAISS) y el ranking tras aplicar re-ranking con LLM.
    Calcula para cada consulta:
      - Hit Rate y MRR para el ranking inicial.
      - Hit Rate y MRR para el candidato seleccionado tras re-ranking.
    """
    total_queries = len(evaluation_data)
    hits_baseline = 0
    hits_rerank = 0
    reciprocal_ranks_baseline = []
    reciprocal_ranks_rerank = []
    
    for item in evaluation_data:
        query = item["query"]
        expected_keyword = item["expected_keyword"]
        candidates = get_candidates(rm, query, top_k=top_k)
        
        # Evaluación del ranking inicial (baseline)
        rank_found_baseline = None
        for cand in candidates:
            caption = cand["caption"]
            # Se normalizan los textos para evitar problemas con acentos
            expected_norm = remove_accents(expected_keyword.lower())
            caption_norm = remove_accents(caption.lower())
            if expected_norm in caption_norm:
                rank_found_baseline = cand["rank"]
                break
        
        if rank_found_baseline is not None:
            hits_baseline += 1
            rr_baseline = 1.0 / rank_found_baseline
        else:
            rr_baseline = 0.0
        reciprocal_ranks_baseline.append(rr_baseline)
        
        # Evaluación usando re-ranking con LLM
        # Se usa el método re_rank_candidates que devuelve el índice (0-based) del candidato seleccionado
        best_idx_rerank = rm.re_rank_candidates(query, candidates)
        # Para re-ranking, consideramos que el candidato seleccionado es rank 1 si es correcto.
        caption_rerank = candidates[best_idx_rerank]["caption"] if candidates else ""
        expected_norm = remove_accents(expected_keyword.lower())
        caption_rerank_norm = remove_accents(caption_rerank.lower())
        if expected_norm in caption_rerank_norm:
            rr_rerank = 1.0  # candidato correcto
            hits_rerank += 1
        else:
            rr_rerank = 0.0
        reciprocal_ranks_rerank.append(rr_rerank)
        
        # Mostrar resultados para la consulta
        print(f"Query: {query}")
        print("Candidatos Baseline:")
        for cand in candidates:
            print(f"  Rank {cand['rank']}: {cand['caption']}")
        print(f"Baseline - Keyword esperada: {expected_keyword}, Rank encontrado: {rank_found_baseline}, Reciprocal Rank: {rr_baseline:.2f}")
        print("Re-ranking:")
        print(f"  Seleccionado (Rank original {candidates[best_idx_rerank]['rank']}): {candidates[best_idx_rerank]['caption']}")
        print(f"  Re-ranking - Keyword esperada: {expected_keyword}, {'Correcto' if rr_rerank==1.0 else 'Incorrecto'}, Reciprocal Rank: {rr_rerank:.2f}")
        print("-" * 50)
    
    hit_rate_baseline = hits_baseline / total_queries
    mrr_baseline = sum(reciprocal_ranks_baseline) / total_queries
    hit_rate_rerank = hits_rerank / total_queries
    mrr_rerank = sum(reciprocal_ranks_rerank) / total_queries
    
    print("Resultados de la Evaluación:")
    print("Baseline Ranking - Hit Rate: {:.2f}, MRR: {:.2f}".format(hit_rate_baseline, mrr_baseline))
    print("Re-ranking (LLM)  - Hit Rate: {:.2f}, MRR: {:.2f}".format(hit_rate_rerank, mrr_rerank))
    return hit_rate_baseline, mrr_baseline, hit_rate_rerank, mrr_rerank

if __name__ == "__main__":
    evaluate(evaluation_data, retrieval_manager, top_k=5)
