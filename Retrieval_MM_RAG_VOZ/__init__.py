"""
Multimodal Vehicle Assistant Package

A modular voice-enabled vehicle assistant that combines:
- Multimodal information retrieval (text + image)
- Voice interaction (speech-to-text and text-to-speech)
- Structured UI for easy interaction

Modules:
- config: Configuration parameters and constants
- retrieval_manager: Information retrieval and search operations
- voice_manager: Voice recording, transcription, and synthesis
- vehicle_assistant_ui: User interface and interaction management
- main: Entry point and application orchestration
"""

__version__ = "2.0.0"
__author__ = "Vehicle Assistant Team"

# Import main components for easy access
from .config import *
from .retrieval_manager import RetrievalManager
from .voice_manager import VoiceManager
from .vehicle_assistant_ui import VehicleAssistantUI

__all__ = [
    'RetrievalManager',
    'VoiceManager', 
    'VehicleAssistantUI'
] 