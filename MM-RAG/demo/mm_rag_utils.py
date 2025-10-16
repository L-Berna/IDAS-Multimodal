"""
Multimodal RAG Utilities for IDAS

This module provides utility functions for multimodal retrieval augmented generation,
including BridgeTower embedding generation, LanceDB search, and session management.

Authors:
- Luis Bernardo Hernandez Salinas
- Juan R. Terven
"""

import torch
import numpy as np
from PIL import Image
import lancedb
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Global store for managing chat history across sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create chat history for a given session.
    
    Args:
        session_id (str): Unique identifier for the conversation session.
    
    Returns:
        BaseChatMessageHistory: Chat history object for the session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# System prompt for contextualizing questions based on chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def initialize_bridgetower(cache_dir="./model_cache"):
    """
    Initialize BridgeTower model and processor.
    
    Args:
        cache_dir (str): Directory to cache the model files.
    
    Returns:
        tuple: (processor, model, device)
    """
    # Configure device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading BridgeTower model on device: {device}")
    
    try:
        # Load processor
        processor = BridgeTowerProcessor.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir=cache_dir
        )
        
        # Load model with error handling
        model = BridgeTowerForContrastiveLearning.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            cache_dir=cache_dir,
            local_files_only=False,  # Allow downloading if not cached
            force_download=False,    # Don't force download if cached
            resume_download=True     # Resume if download was interrupted
        ).to(device)
        
        model.eval()
        print("BridgeTower model loaded successfully!")
        
        return processor, model, device
        
    except Exception as e:
        print(f"Error loading BridgeTower model: {e}")
        print("Trying to download model from scratch...")
        
        try:
            # Try downloading without cache
            processor = BridgeTowerProcessor.from_pretrained(
                "BridgeTower/bridgetower-large-itm-mlm-itc",
                cache_dir=cache_dir,
                local_files_only=False,
                force_download=True
            )
            
            model = BridgeTowerForContrastiveLearning.from_pretrained(
                "BridgeTower/bridgetower-large-itm-mlm-itc",
                cache_dir=cache_dir,
                local_files_only=False,
                force_download=True
            ).to(device)
            
            model.eval()
            print("BridgeTower model downloaded and loaded successfully!")
            
            return processor, model, device
            
        except Exception as e2:
            print(f"Failed to load BridgeTower model: {e2}")
            print("Please check your internet connection and try again.")
            raise e2


def generate_query_embedding(query_text, processor, model, device):
    """
    Generate text embedding for a query using BridgeTower.
    
    BridgeTower requires both image and text inputs, so we use a dummy
    white image for text-only queries.
    
    Args:
        query_text (str): The query text.
        processor: BridgeTower processor.
        model: BridgeTower model.
        device: PyTorch device (CPU or CUDA).
    
    Returns:
        list: Normalized text embedding as a list of floats.
    """
    # Create a dummy white image (BridgeTower requires image input)
    dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
    
    # Process inputs
    encoding = processor(
        images=dummy_image,
        text=query_text,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoding)
        # L2 normalize text embeddings
        text_embeds = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=1)
    
    # Convert to list and return
    return text_embeds.cpu().numpy().flatten().astype('float32').tolist()


def connect_lancedb(db_path):
    """
    Connect to LanceDB and open the embeddings table.
    
    Args:
        db_path (str): Path to the LanceDB database.
    
    Returns:
        tuple: (db connection, table)
    """
    db = lancedb.connect(db_path)
    table = db.open_table("embeddings")
    print(f"Connected to LanceDB at: {db_path}")
    print(f"Number of records: {table.count_rows()}")
    return db, table


def search_multimodal(query_text, table, processor, model, device, top_k=5, llm=None, use_llm_reranking=True, text_weight=0.7, image_weight=0.3, aggregation_count=3):
    """
    Search for relevant images and captions using a text query with hybrid re-ranking and LLM-based selection.
    
    Args:
        query_text (str): The user's query.
        table: LanceDB table with embeddings.
        processor: BridgeTower processor.
        model: BridgeTower model.
        device: PyTorch device.
        top_k (int): Number of results to retrieve.
        llm: Language model for re-ranking (optional).
        use_llm_reranking (bool): Whether to use LLM-based re-ranking.
    
    Returns:
        dict: Dictionary containing:
            - image_caption (str): Caption of the best match
            - image_bytes (bytes): Image data of the best match
            - all_results (DataFrame): All search results
            - aggregated_context (str): Context from multiple top results
    """
    # Generate query embedding
    query_embedding = generate_query_embedding(query_text, processor, model, device)
    query_vector = np.array(query_embedding, dtype='float32')
    
    # Search in LanceDB using cosine similarity on text_vector column
    results_df = table.search(query_vector.tolist(), vector_column_name="text_vector") \
                      .metric("cosine") \
                      .limit(top_k) \
                      .to_pandas()
    
    if len(results_df) > 0:
        # Apply hybrid re-ranking combining text and image vectors
        reranked_results = apply_hybrid_reranking(results_df, query_vector, top_k, text_weight=text_weight, image_weight=image_weight)
        
        # Apply LLM-based re-ranking if enabled and LLM is available
        if use_llm_reranking and llm is not None:
            query_type = classify_query_type(query_text)
            reranked_results = apply_llm_reranking(query_text, reranked_results, llm, query_type)
        
        # Get best result after re-ranking
        best_result = reranked_results.iloc[0]
        
        # Create aggregated context from multiple top results
        aggregated_context = create_aggregated_context(reranked_results, aggregation_count=aggregation_count)
        
        return {
            "image_caption": best_result.get("texto", "No caption available"),
            "image_bytes": best_result.get("imagen", None),
            "all_results": reranked_results,
            "aggregated_context": aggregated_context
        }
    else:
        return {
            "image_caption": "No relevant information found in the manual.",
            "image_bytes": None,
            "all_results": None,
            "aggregated_context": "No relevant information found in the manual."
        }


def apply_hybrid_reranking(results_df, query_vector, top_k, text_weight=0.7, image_weight=0.3):
    """
    Apply hybrid re-ranking combining text and image vector similarities.
    
    Args:
        results_df (DataFrame): Initial search results
        query_vector (np.array): Query embedding vector
        top_k (int): Number of results to return
    
    Returns:
        DataFrame: Re-ranked results
    """
    if len(results_df) == 0:
        return results_df
    
    # Normalize query vector
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    
    # Extract and normalize vectors
    text_vectors = np.vstack(results_df["text_vector"].to_list()).astype("float32")
    image_vectors = np.vstack(results_df["image_vector"].to_list()).astype("float32")
    
    # L2 normalize vectors
    def l2_normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms
    
    text_vectors = l2_normalize(text_vectors)
    image_vectors = l2_normalize(image_vectors)
    
    # Calculate similarities
    text_similarity = text_vectors @ query_norm
    image_similarity = image_vectors @ query_norm
    
    # Hybrid scoring: configurable weights
    hybrid_scores = text_weight * text_similarity + image_weight * image_similarity
    
    # Add scores to dataframe
    results_df = results_df.copy()
    results_df['hybrid_score'] = hybrid_scores
    
    # Sort by hybrid score and return top_k
    reranked_df = results_df.sort_values('hybrid_score', ascending=False).head(top_k)
    
    return reranked_df


def create_aggregated_context(results_df, aggregation_count=3):
    """
    Create aggregated context from multiple top results.
    
    Args:
        results_df (DataFrame): Re-ranked results
        top_k (int): Number of results to aggregate
    
    Returns:
        str: Aggregated context string
    """
    if len(results_df) == 0:
        return "No relevant information found."
    
    # Take top results
    top_results = results_df.head(aggregation_count)
    
    # Create context from multiple captions
    contexts = []
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        caption = row.get("texto", "")
        if caption and caption.strip():
            contexts.append(f"Contexto {i}: {caption.strip()}")
    
    if contexts:
        return "\n\n".join(contexts)
    else:
        return "No relevant information found."


def apply_llm_reranking(query_text, results_df, llm, query_type="general"):
    """
    Apply LLM-based re-ranking for intelligent result selection.
    
    Args:
        query_text (str): User's query
        results_df (DataFrame): Results to re-rank
        llm: Language model for re-ranking
        query_type (str): Type of query (diagnostic, maintenance, etc.)
    
    Returns:
        DataFrame: Re-ranked results with LLM scores
    """
    if len(results_df) == 0:
        return results_df
    
    # Instructions based on query type
    type_instructions = {
        "diagnostic": "Prioriza resultados que muestren síntomas, problemas o fallas específicas del vehículo.",
        "maintenance": "Prioriza resultados sobre mantenimiento, cambios de partes o inspecciones del vehículo.",
        "procedure": "Prioriza resultados que muestren pasos, procedimientos o instrucciones detalladas.",
        "specification": "Prioriza resultados con datos técnicos, especificaciones o medidas del vehículo.",
        "general": "Selecciona el resultado más relevante para la consulta general sobre el vehículo."
    }
    
    instruction = type_instructions.get(query_type, type_instructions["general"])
    
    # Create prompt for LLM re-ranking
    prompt = f"""Como experto automotriz, evalúa estos resultados para la consulta:
"{query_text}"

Tipo de consulta: {query_type}
Instrucción específica: {instruction}

Resultados disponibles:
"""
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        caption = row.get("texto", "")
        score = row.get("hybrid_score", 0.0)
        prompt += f"{i}. {caption} (Score híbrido: {score:.3f})\n"
    
    prompt += f"\nConsiderando el tipo de consulta ({query_type}) y la instrucción específica, evalúa cada resultado del 1 al 10 basándote en qué tan bien responde a la consulta. Responde con el formato: 'Resultado X: Puntuación Y' para cada resultado."
    
    try:
        # Get LLM response
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse LLM scores
        llm_scores = {}
        for line in response_text.split('\n'):
            if 'Resultado' in line and ':' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        result_num = int(parts[0].split()[-1])
                        score = float(parts[1].strip())
                        llm_scores[result_num] = score
                except (ValueError, IndexError):
                    continue
        
        # Add LLM scores to dataframe
        results_df = results_df.copy()
        results_df['llm_score'] = 0.0
        
        for i, (idx, row) in enumerate(results_df.iterrows()):
            if i + 1 in llm_scores:
                results_df.at[idx, 'llm_score'] = llm_scores[i + 1]
        
        # Combine hybrid and LLM scores (70% hybrid, 30% LLM)
        results_df['final_score'] = 0.7 * results_df['hybrid_score'] + 0.3 * results_df['llm_score']
        
        # Sort by final score
        final_ranked = results_df.sort_values('final_score', ascending=False)
        
        return final_ranked
        
    except Exception as e:
        print(f"Error in LLM re-ranking: {e}")
        # Return original results if LLM fails
        return results_df


def classify_query_type(query_text):
    """
    Classify the type of query to improve result selection.
    
    Args:
        query_text (str): User's query
    
    Returns:
        str: Query type (diagnostic, maintenance, procedure, specification, general)
    """
    query_lower = query_text.lower()
    
    # Keywords for different query types
    diagnostic_keywords = ['problema', 'falla', 'error', 'síntoma', 'diagnóstico', 'revisar', 'verificar', 'daño']
    maintenance_keywords = ['mantenimiento', 'cambio', 'reemplazar', 'revisar', 'inspección', 'servicio', 'aceite', 'filtro']
    procedure_keywords = ['cómo', 'pasos', 'procedimiento', 'instrucción', 'hacer', 'realizar', 'activar', 'desactivar']
    specification_keywords = ['especificación', 'medida', 'tamaño', 'capacidad', 'voltaje', 'presión', 'temperatura', 'datos']
    
    # Count keyword matches
    scores = {
        'diagnostic': sum(1 for kw in diagnostic_keywords if kw in query_lower),
        'maintenance': sum(1 for kw in maintenance_keywords if kw in query_lower),
        'procedure': sum(1 for kw in procedure_keywords if kw in query_lower),
        'specification': sum(1 for kw in specification_keywords if kw in query_lower)
    }
    
    # Return type with highest score, or 'general' if no matches
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        return 'general'


def retrieve_context_with_image(query_text, table, processor, model, device, top_k=5):
    """
    Retrieve visual context (caption and image) for a query.
    
    This is a convenience function that wraps search_multimodal and returns
    just the caption and image bytes.
    
    Args:
        query_text (str): The user's query.
        table: LanceDB table with embeddings.
        processor: BridgeTower processor.
        model: BridgeTower model.
        device: PyTorch device.
        top_k (int): Number of results to retrieve.
    
    Returns:
        tuple: (image_caption, image_bytes)
    """
    result = search_multimodal(query_text, table, processor, model, device, top_k)
    return result["image_caption"], result["image_bytes"]
