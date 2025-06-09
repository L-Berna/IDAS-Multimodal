# Multimodal Vehicle Assistant - Modular Version

Este proyecto es una versión modularizada del asistente multimodal de vehículos que integra capacidades de voz y búsqueda multimodal.

## 📁 Estructura del Proyecto

```
Retrieval_MM_RAG_VOZ/
├── __init__.py              # Package initialization
├── config.py                # Configuraciones globales y constantes
├── retrieval_manager.py     # Gestión de búsqueda y recuperación de información
├── voice_manager.py         # Gestión de voz (grabación, transcripción, síntesis)
├── vehicle_assistant_ui.py  # Interfaz gráfica de usuario
├── main.py                  # Punto de entrada principal
└── README.md               # Este archivo
```

## 🔧 Módulos

### 1. `config.py`
**Responsabilidades:**
- Variables de entorno y API keys
- Parámetros de configuración (audio, modelos, UI)
- Configuración de dispositivos y logging

### 2. `retrieval_manager.py`
**Responsabilidades:**
- Conexión a LanceDB y construcción del índice FAISS
- Inicialización de RetrievalQA con Chroma
- Generación de embeddings multimodales con BridgeTower
- Búsqueda y re-ranking de candidatos
- Filtrado de consultas relacionadas con vehículos

### 3. `voice_manager.py`
**Responsabilidades:**
- Grabación de audio con PyAudio
- Transcripción con Whisper de OpenAI
- Síntesis de voz con ElevenLabs
- Reproducción de audio
- Gestión de archivos temporales

### 4. `vehicle_assistant_ui.py`
**Responsabilidades:**
- Interfaz gráfica con Tkinter
- Coordinación entre módulos de voz y búsqueda
- Gestión del historial de conversación
- Manejo de eventos de la UI

### 5. `main.py`
**Responsabilidades:**
- Punto de entrada de la aplicación
- Inicialización y orquestación de todos los módulos
- Manejo de excepciones globales

## 🚀 Uso

### Ejecución desde la carpeta del proyecto:
```bash
cd Retrieval_MM_RAG_VOZ
python main.py
```

### Importación como paquete:
```python
from Retrieval_MM_RAG_VOZ import RetrievalManager, VoiceManager, VehicleAssistantUI
from Retrieval_MM_RAG_VOZ.config import *
```

## 📋 Requisitos

Las mismas dependencias del proyecto original:
- torch
- transformers
- langchain-openai
- lancedb
- faiss-cpu/faiss-gpu
- pyaudio
- elevenlabs
- pydub
- PIL
- tkinter

## 🔧 Variables de Entorno

```bash
export OPENAI_API_KEY="tu_clave_openai"
export ELEVEN_LABS_KEY="tu_clave_elevenlabs"
```

## ✅ Ventajas de la Modularización

1. **Separación de responsabilidades**: Cada módulo tiene una función específica
2. **Mantenibilidad**: Cambios en un módulo no afectan otros
3. **Reutilización**: Los módulos pueden importarse independientemente
4. **Testeo**: Más fácil crear unit tests para cada componente
5. **Configuración centralizada**: Parámetros en un solo lugar
6. **Colaboración**: Diferentes desarrolladores pueden trabajar en módulos distintos

## 📈 Mejoras Implementadas

- **Cache LRU** para embeddings (mejora rendimiento)
- **Índice FAISS optimizado** (M=64, efConstruction=200, efSearch=100)
- **Normalización solo en FAISS** (mejor precisión)
- **Prompts estructurados** para respuestas más concisas
- **Filtrado de consultas** relacionadas con vehículos
- **Precarga de modelos** para reducir latencia

## 🔄 Migración desde Versión Monolítica

El archivo original `Retriever_MM_RAG_VOZ_V2_original.py` se mantiene como referencia. Esta versión modular incluye todas las optimizaciones y mejoras del original. 