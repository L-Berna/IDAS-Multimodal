"""
Main Entry Point for Multimodal Vehicle Assistant

This module serves as the entry point for the application and orchestrates
all the different components (retrieval, voice, and UI management).
"""

from config import (
    logger, device, LANCEDB_PATH, VECTOR_DB_DIR, LLM_MODEL, 
    OPENAI_API_KEY, ELEVEN_API_KEY
)
from retrieval_manager import RetrievalManager
from voice_manager import VoiceManager
from vehicle_assistant_ui import VehicleAssistantUI


def main():
    """
    Main function that initializes all components and starts the application.
    """
    try:
        logger.info("Iniciando Asistente Multimodal y de Voz para Vehículos...")
        
        # Initialize Retrieval Manager
        logger.info("Inicializando RetrievalManager...")
        retrieval_manager = RetrievalManager(
            device=device,
            db_path=LANCEDB_PATH,
            vector_db_dir=VECTOR_DB_DIR,
            llm_model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("RetrievalManager inicializado correctamente.")
        
        # Initialize Voice Manager
        logger.info("Inicializando VoiceManager...")
        voice_manager = VoiceManager(
            eleven_api_key=ELEVEN_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("VoiceManager inicializado correctamente.")
        
        # Initialize and start UI
        logger.info("Inicializando interfaz de usuario...")
        app_ui = VehicleAssistantUI(retrieval_manager, voice_manager)
        logger.info("Aplicación iniciada correctamente.")
        
    except Exception as e:
        logger.exception("Error al iniciar la aplicación: %s", e)
        raise


if __name__ == "__main__":
    main() 