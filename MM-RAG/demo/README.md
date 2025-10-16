# MM-IDAS: Multimodal Intelligent Driving Assistance System

A voice-based conversational assistant for vehicles that uses multimodal RAG (Retrieval Augmented Generation) with BridgeTower embeddings to provide answers with visual context from the car manual.

## 🎯 Features

- **Voice Input**: Speech-to-text using OpenAI Whisper
- **Multimodal Retrieval**: BridgeTower embeddings + LanceDB for semantic search
- **Visual Context**: Displays relevant images from the car manual
- **Conversational Memory**: Maintains chat history and context
- **Voice Output**: Text-to-speech using ElevenLabs
- **GUI Interface**: User-friendly Tkinter interface

## 📁 Files

```
demo/
├── mm_rag_utils.py          # Multimodal RAG utility functions
├── mm_idas_gui.py           # Main GUI application
├── config_mm_idas.ini       # Configuration file
└── README.md                # This file
```

## 🔧 Prerequisites

### 1. Python Dependencies

Install required packages:

```bash
pip install torch torchvision
pip install transformers
pip install lancedb
pip install langchain langchain-openai langchain-community langchain-anthropic langchain-mistralai
pip install openai elevenlabs
pip install pyaudio pillow
pip install colorama
```

### 2. API Keys

Set the following environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
export ELEVEN_LABS_KEY="your-elevenlabs-api-key"

# Optional (depending on LLM choice)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ELEVEN_LABS_KEY="your-elevenlabs-api-key"
```

### 3. LanceDB Database

Ensure you have generated the multimodal vector database using the notebook:
- `1_Embeddings_Generator_Bridgetower.py` or
- `notebooks/1_Vector_Database_BridgeTower.py`

The database should be located at the path specified in `config_mm_idas.ini`.

### 4. Additional Dependencies (Windows)

For audio streaming on Windows, you need **mpv**:
- Download from: https://mpv.io/installation/
- Extract to parent directory of the project

## ⚙️ Configuration

Edit `config_mm_idas.ini` to customize:

```ini
[DEFAULT]
vehicle_name = kia_sorento
llm_name = gpt-4o-mini
lancedb_path = C:\path\to\your\LanceDB_KIA-Simple_Mejorado
voice_name = Rachel
top_k = 5
```

### Available Options

**LLM Models:**
- OpenAI: `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-5-sonnet-20240620`
- Mistral: `open-mistral-7b`, `open-mixtral-8x7b`, `mistral-large-2407`

**Voice Names (ElevenLabs):**
- Rachel, Drew, Clyde, Paul, Domi, Dave, Fin, Sarah, Antoni, Thomas

## 🚀 Usage

### Running the Application

From the demo directory:

```bash
python mm_idas_gui.py --config config_mm_idas.ini
```

### Using the Interface

1. **Start Listening**: Click the green "Start listening" button and speak your question
2. **Stop Listening**: Click again to stop recording and process your question
3. **View Results**: 
   - The conversation appears in the text box
   - Relevant manual image is displayed below
   - Audio response plays automatically
4. **New Chat**: Click "New Chat" to start a fresh conversation

### Example Conversation

```
👤 You: "How do I change a flat tire?"
🤖 IDAS: [Shows tire change image and explains the procedure]

👤 You: "Where is the jack located?"
🤖 IDAS: [Shows jack location image and explains, understanding context from previous question]

👤 You: "And the spare tire?"
🤖 IDAS: [Continues the conversation with context awareness]
```

## 🏗️ Architecture

```
User Voice Input
    ↓
Whisper Transcription (OpenAI)
    ↓
Question Reformulation (if chat history exists)
    ↓
BridgeTower Embedding Generation
    ↓
LanceDB Similarity Search
    ↓
Retrieve Image + Caption
    ↓
LLM Answer Generation (with visual context)
    ↓
ElevenLabs Text-to-Speech
    ↓
Display Text + Image in GUI
```

## 🔍 How It Works

### Multimodal RAG Pipeline

1. **BridgeTower Embeddings**: Generates unified embeddings from text and images
2. **LanceDB Search**: Fast vector similarity search with cosine metric
3. **Context Integration**: Combines visual information (image + caption) with LLM
4. **Conversational Memory**: Maintains session history for follow-up questions

### Key Functions (mm_rag_utils.py)

- `initialize_bridgetower()`: Load the multimodal model
- `generate_query_embedding()`: Create query embeddings
- `search_multimodal()`: Search the vector database
- `get_session_history()`: Manage conversation history

## 🐛 Troubleshooting

### Issue: "No module named 'mm_rag_utils'"
**Solution**: Make sure you're running from the demo directory or add it to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "CUDA out of memory"
**Solution**: 
- Use CPU instead (automatic fallback)
- Reduce batch size or use a smaller model
- Close other GPU-intensive applications

### Issue: "LanceDB not found"
**Solution**: Check the `lancedb_path` in config file points to the correct location

### Issue: Audio not working (Windows)
**Solution**: 
- Install mpv player
- Check microphone permissions
- Verify ElevenLabs API key

### Issue: "Port already in use" or async errors
**Solution**: Close any other instances of the application

## 📊 Performance

**Typical Response Times:**
- Audio transcription (Whisper): ~1-2 seconds
- Multimodal search (BridgeTower + LanceDB): ~0.5-1 second
- LLM generation (GPT-4o-mini): ~1-2 seconds
- TTS (ElevenLabs): Streaming (starts immediately)

**Total**: ~3-5 seconds from speech to response

**Hardware Requirements:**
- **Minimum**: 8GB RAM, CPU (slower)
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Disk**: ~2GB for BridgeTower model

## 📝 Differences from Regular IDAS

| Feature | IDAS (rag_utils) | MM-IDAS (mm_rag_utils) |
|---------|------------------|------------------------|
| Embeddings | OpenAI text embeddings | BridgeTower multimodal |
| Vector DB | Chroma | LanceDB |
| Context | Text only | Text + Images |
| Retrieval | Standard RAG chain | Custom multimodal search |
| GUI | Text display | Text + Image display |

## 🎓 Authors

- Luis Bernardo Hernandez Salinas
- Juan R. Terven

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows existing style
2. Functions are documented
3. Configuration changes are reflected in config file
4. Test with different queries before submitting

---

**Note**: This system requires active internet connection for:
- OpenAI Whisper API (transcription)
- OpenAI/Anthropic/Mistral API (LLM)
- ElevenLabs API (text-to-speech)

The BridgeTower model and LanceDB operate locally.
