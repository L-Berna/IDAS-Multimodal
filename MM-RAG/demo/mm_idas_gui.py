"""
Multimodal Intelligent Driving Assistance System (MM-IDAS) with GUI

A voice-based conversational assistant for vehicles that uses multimodal RAG
(Retrieval Augmented Generation) with BridgeTower embeddings and LanceDB
to provide answers with visual context from the car manual.

Features:
- Voice input (Whisper ASR)
- Multimodal retrieval (BridgeTower + LanceDB)
- Conversational memory
- Voice output (ElevenLabs TTS)
- Visual display of manual images
- GUI interface (Tkinter)

Authors:
- Luis Bernardo Hernandez Salinas
- Juan R. Terven
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import re
import asyncio
import threading
import configparser
import argparse
import io
import datetime
from pathlib import Path

from openai import OpenAI
import pyaudio
import wave
import platform
from ctypes import *
from PIL import Image, ImageTk
import tkinter as tk
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import logging
from colorama import init, Fore

# Import multimodal RAG utilities
from mm_rag_utils import (
    get_session_history,
    contextualize_q_system_prompt,
    initialize_bridgetower,
    connect_lancedb,
    search_multimodal
)

logging.getLogger().setLevel(logging.ERROR)  # Hide warning logs
os_name = platform.system()  # Get the name of the OS
init(autoreset=True)  # Initialize colorama

# Audio output streaming needs mpv in Windows
# https://mpv.io/installation/
if os_name == "Windows":
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    os.environ['PATH'] += os.pathsep + parent_directory + '/mpv/'

# Global Variables
event_loop = None   # Store the event loop
app_running = True  # Flag to indicate if the app is running
rec_stream = None   # Audio recording
state = "waiting"   # System state
wav_file = None     # Wave file object
session_id = 1

# Audio file characteristics
temp_file = "prompt_recording.wav"  # File to store the prompt audio
sample_rate = 16000
bits_per_sample = 16
chunk_size = 1024
audio_format = pyaudio.paInt16
channels = 1

# Global variables for multimodal components
processor = None
model_bridgetower = None
device = None
lancedb_table = None
current_image_bytes = None  # Store current image for display

# Global variables for image saving
session_folder = None
response_counter = 0

# Global variables for configuration
vehicle_name = None
llm_name = None
system_prompt = None
lancedb_path = None
voice_name = None
top_k = None
use_llm_reranking = None
hybrid_text_weight = None
hybrid_image_weight = None
context_aggregation_count = None
save_topk_images = None
save_metadata = None
llm = None
openai_client = None
elabs_client = None
contextualize_chain = None
enable_question_reformulation = None


def initialize_system():
    """Initialize the system components including models and database connections."""
    global processor, model_bridgetower, device, lancedb_table
    global vehicle_name, llm_name, system_prompt, lancedb_path, voice_name, top_k
    global use_llm_reranking, hybrid_text_weight, hybrid_image_weight, context_aggregation_count
    global save_topk_images, save_metadata
    global llm, openai_client, elabs_client, contextualize_chain
    global enable_question_reformulation
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Load configuration for MM-IDAS.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Resolve config path (allow passing base name without .ini)
    config_path = args.config
    if not os.path.isfile(config_path):
        if not config_path.lower().endswith('.ini') and os.path.isfile(config_path + '.ini'):
            config_path = config_path + '.ini'
    if not os.path.isfile(config_path):
        available = [f for f in os.listdir(os.path.dirname(__file__)) if f.lower().endswith('.ini')]
        print(Fore.RED + f"Config file not found: {args.config}")
        if available:
            print(Fore.YELLOW + "Available .ini files in this folder:")
            for f in available:
                print(Fore.YELLOW + f"  - {f}")
        sys.exit(1)

    # Read the configuration file with UTF-8 encoding
    config.read(config_path, encoding='utf-8')

    # Access values from config file
    vehicle_name = config.get('DEFAULT', 'vehicle_name')
    llm_name = config.get('DEFAULT', 'llm_name')
    system_prompt = config.get('DEFAULT', 'system_prompt')
    lancedb_path = config.get('DEFAULT', 'lancedb_path')
    voice_name = config.get('DEFAULT', 'voice_name')
    top_k = config.getint('DEFAULT', 'top_k', fallback=5)
    
    # Advanced retrieval settings
    use_llm_reranking = config.getboolean('DEFAULT', 'use_llm_reranking', fallback=True)
    hybrid_text_weight = config.getfloat('DEFAULT', 'hybrid_text_weight', fallback=0.7)
    hybrid_image_weight = config.getfloat('DEFAULT', 'hybrid_image_weight', fallback=0.3)
    context_aggregation_count = config.getint('DEFAULT', 'context_aggregation_count', fallback=3)
    enable_question_reformulation = config.getboolean('DEFAULT', 'enable_question_reformulation', fallback=True)
    
    # Image saving settings
    save_topk_images = config.getboolean('DEFAULT', 'save_topk_images', fallback=True)
    save_metadata = config.getboolean('DEFAULT', 'save_metadata', fallback=True)

    # Get the proper LLM API key
    if "gpt" in llm_name:
        llm_api_key = os.environ['OPENAI_API_KEY']
    elif "claude" in llm_name:
        llm_api_key = os.environ['ANTHROPIC_API_KEY']
    elif "mistral" in llm_name or "mixtral" in llm_name:
        llm_api_key = os.environ['MISTRAL_API_KEY']
    elif "llama" in llm_name or "gemma" in llm_name:
        llm_api_key = os.environ['LLAMA_API_KEY']
    else:
        print(Fore.RED + "INVALID MODEL!")
        sys.exit(0)

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    eleven_api_key = os.environ.get('ELEVEN_LABS_KEY')

    # Set the LLM
    if "gpt" in llm_name:
        llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0)
    elif "claude" in llm_name:
        llm = ChatAnthropic(model_name=llm_name, api_key=llm_api_key, temperature=0)
    elif "mistral" in llm_name or "mixtral" in llm_name:
        llm = ChatMistralAI(model=llm_name, api_key=llm_api_key, temperature=0)
    elif "llama" in llm_name or "gemma" in llm_name:
        llm = ChatOpenAI(model_name=llm_name, api_key=llm_api_key, temperature=0,
                         base_url="https://api.llama-api.com")
    else:
        print(Fore.RED + "UNABLE TO SET THE LLM")
        sys.exit(0)

    if hasattr(llm, "model_name"):
        print(Fore.GREEN + f"Using Model: {llm.model_name}")
    else:
        print(Fore.GREEN + f"Using Model: {llm_name}")

    # Configure OpenAI and Text-to-speech API keys
    openai_client = OpenAI(api_key=openai_api_key)  # For Whisper
    elabs_client = ElevenLabs(api_key=eleven_api_key)

    # Initialize BridgeTower model
    print(Fore.CYAN + "Initializing BridgeTower model...")
    processor, model_bridgetower, device = initialize_bridgetower()

    # Connect to LanceDB
    print(Fore.CYAN + "Connecting to LanceDB...")
    if not os.path.exists(lancedb_path):
        print(Fore.RED + f"LanceDB not found at: {lancedb_path}")
        print(Fore.RED + "Exiting!")
        sys.exit(0)
    else:
        print(Fore.GREEN + f"Using LanceDB: {lancedb_path}")

    db, lancedb_table = connect_lancedb(lancedb_path)

    # Create contextualization prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create contextualization chain
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # Create initial session folder
    create_session_folder()


def get_contextualized_question(input_text, session_id):
    """
    Reformulate question based on chat history if available.
    
    Args:
        input_text (str): User's input question.
        session_id (str): Session identifier.
    
    Returns:
        str: Standalone question.
    """
    # If reformulation is disabled, return as-is
    if enable_question_reformulation is False:
        return input_text

    chat_history = get_session_history(session_id)
    
    if len(chat_history.messages) > 0:
        # There's history, reformulate the question
        result = contextualize_chain.invoke({
            "input": input_text,
            "chat_history": chat_history.messages
        })
        return result
    else:
        # No history, return question as is
        return input_text


def generate_answer_with_visual_context(query, image_caption, session_id):
    """
    Generate answer using LLM with visual context from manual.
    
    Args:
        query (str): User's original query.
        image_caption (str): Caption describing the retrieved image.
        session_id (str): Session identifier.
    
    Returns:
        str: Generated answer.
    """
    # Create prompt with visual context
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nVisual Information from Manual: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Get chat history
    chat_history = get_session_history(session_id)
    
    # Generate answer
    chain = qa_prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "input": query,
        "context": image_caption,
        "chat_history": chat_history.messages
    })
    
    # Add to history
    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)
    
    return answer


# Suppress ALSA warnings on Linux
if os_name == 'Linux':
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        return
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

# Initialize PyAudio
audio = pyaudio.PyAudio()


def write_text_safe(text, color):
    """Write colored text to the Text widget safely from any thread."""
    root.after(0, write_text, text, color)


def write_text(text, color):
    """Write colored text to the Text widget."""
    tag_name = f"color_{color}"
    
    if tag_name not in text_box.tag_names():
        text_box.tag_configure(tag_name, foreground=color)
    
    text_box.insert(tk.END, text, tag_name)
    text_box.see(tk.END)


def clear_text_box():
    """Clear the content of the text box."""
    text_box.delete("1.0", tk.END)


def clear_image():
    """Clear the image display."""
    image_label.config(image="")
    image_label.image = None


def display_image_in_gui(image_bytes):
    """
    Display retrieved image in the GUI.
    
    Args:
        image_bytes (bytes): Image data.
    """
    global current_image_bytes
    current_image_bytes = image_bytes
    
    if image_bytes is not None:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Resize to fit in display area
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(image)
            image_label.config(image=tk_image)
            image_label.image = tk_image  # Keep a reference
        except Exception as e:
            print(Fore.RED + f"Error displaying image: {e}")
            image_label.config(image="")
            image_label.image = None
    else:
        clear_image()


def create_session_folder():
    """Create a folder for the current session to save images."""
    global session_folder, response_counter
    
    # Create main folder if it doesn't exist
    main_folder = Path("mm_idas_responses")
    main_folder.mkdir(exist_ok=True)
    
    # Create session folder with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = main_folder / f"session_{session_id}_{timestamp}"
    session_folder.mkdir(exist_ok=True)
    
    response_counter = 0
    print(Fore.GREEN + f"Session folder created: {session_folder}")


def save_topk_images_to_disk(mm_result, query_text):
    """Save all images from top-k results to organized folders."""
    global response_counter
    
    if not session_folder:
        create_session_folder()
    
    response_counter += 1
    
    # Create response folder
    safe_query = re.sub(r'[^\w\s-]', '', query_text)[:50]  # Safe filename
    response_folder = session_folder / f"response_{response_counter:03d}_{safe_query}"
    response_folder.mkdir(exist_ok=True)
    
    # Save images from all results
    all_results = mm_result.get("all_results")
    if all_results is not None and len(all_results) > 0:
        for i, (_, row) in enumerate(all_results.iterrows()):
            image_bytes = row.get("imagen")
            caption = row.get("texto", f"Result {i+1}")
            
            if image_bytes is not None:
                try:
                    # Create safe filename
                    safe_caption = re.sub(r'[^\w\s-]', '', caption)[:30]
                    image_filename = f"rank_{i+1:02d}_{safe_caption}.jpg"
                    image_path = response_folder / image_filename
                    
                    # Save image
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    print(Fore.CYAN + f"  Saved: {image_filename}")
                    
                except Exception as e:
                    print(Fore.RED + f"  Error saving image {i+1}: {e}")
        
        # Save metadata (if enabled)
        if save_metadata:
            metadata_path = response_folder / "metadata.txt"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query_text}\n")
                f.write(f"Response Number: {response_counter}\n")
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Total Results: {len(all_results)}\n\n")
                
                for i, (_, row) in enumerate(all_results.iterrows()):
                    f.write(f"Rank {i+1}:\n")
                    f.write(f"  Caption: {row.get('texto', 'N/A')}\n")
                    f.write(f"  Hybrid Score: {row.get('hybrid_score', 'N/A')}\n")
                    f.write(f"  LLM Score: {row.get('llm_score', 'N/A')}\n")
                    f.write(f"  Final Score: {row.get('final_score', 'N/A')}\n\n")
        
        print(Fore.GREEN + f"Saved {len(all_results)} images to: {response_folder}")
    else:
        print(Fore.YELLOW + f"No results to save for response {response_counter}")


def new_session():
    """Start a new conversation session."""
    global session_id, response_counter
    session_id += 1
    response_counter = 0
    clear_text_box()
    clear_image()
    create_session_folder()
    print(Fore.CYAN + f"New session started: {session_id}")


async def process_audio():
    """
    Process recorded audio: transcribe, search multimodal DB, generate response.
    """
    global session_id, current_image_bytes
    try:
        # Step 1: Transcribe audio with Whisper
        audio_file = open(temp_file, "rb")
        query_transcription = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        query = query_transcription.text
        audio_file.close()
        
        print(Fore.CYAN + f"\nYou said: {query}")
        write_text_safe(f"You: {query}\n", "blue")
        
        # Step 2: Contextualize question (reformulate if needed)
        standalone_question = get_contextualized_question(query, str(session_id))
        if standalone_question != query:
            print(Fore.YELLOW + f"Reformulated: {standalone_question}")
        
        # Step 3: Search multimodal database with improved retrieval
        mm_result = search_multimodal(
            standalone_question,
            lancedb_table,
            processor,
            model_bridgetower,
            device,
            top_k=top_k,
            llm=llm,
            use_llm_reranking=use_llm_reranking,
            text_weight=hybrid_text_weight,
            image_weight=hybrid_image_weight,
            aggregation_count=context_aggregation_count
        )
        
        image_caption = mm_result["image_caption"]
        image_bytes = mm_result["image_bytes"]
        aggregated_context = mm_result.get("aggregated_context", image_caption)
        
        # Step 4: Save top-k images to organized folders (if enabled)
        if save_topk_images:
            root.after(0, save_topk_images_to_disk, mm_result, query)
        
        # Step 5: Generate answer with enhanced visual context
        response_text = generate_answer_with_visual_context(
            query,
            aggregated_context,  # Use aggregated context instead of single caption
            str(session_id)
        )
        
        # Step 6: Display image in GUI
        root.after(0, display_image_in_gui, image_bytes)
        
        # Step 6: Convert response to audio and stream
        audio_stream = elabs_client.generate(
            text=text_stream(response_text),
            voice=voice_name,
            model="eleven_multilingual_v2",
            stream=True
        )
        
        await asyncio.to_thread(stream, audio_stream)
        root.after(0, reset_button)
        
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        write_text_safe(f"\nError: {str(e)}\n", "red")
        root.after(0, reset_button)
        root.after(0, write_text, "\n", "")
    finally:
        root.after(0, enable_listening_button)


def enable_listening_button():
    """Enable the listening button."""
    btn_listening.config(state=tk.NORMAL)


def reset_button():
    """Reset the listening button to initial state."""
    global state
    state = "waiting"
    btn_listening["text"] = "Start listening"
    btn_listening.config(bg="green", fg="black")
    write_text("\n", "")


def toggle_state():
    """Toggle between listening and processing states."""
    global rec_stream, state, wav_file
    
    if state == "waiting":
        # Start listening
        btn_listening["text"] = "Stop listening"
        btn_listening.config(bg="lightcoral", fg="black")
        state = "listening"
        write_text("Listening...\n", "green")
        
        # Initialize wave file
        initialize_wave_file()
        
        # Start recording audio
        rec_stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            stream_callback=audio_recording_callback
        )
    
    elif state == "listening":
        # Stop listening and process
        btn_listening["text"] = "Processing ..."
        btn_listening.config(bg="#a3a300", fg="black")
        state = "processing"
        
        # Stop and close the audio stream
        rec_stream.stop_stream()
        rec_stream.close()
        
        # Close the wave file
        wav_file.close()
        wav_file = None
        
        # Start processing audio asynchronously
        asyncio.run_coroutine_threadsafe(process_audio(), event_loop)
    
    elif state == "processing":
        pass  # Do nothing while processing


def initialize_wave_file():
    """Initialize the wave file for recording."""
    global wav_file
    wav_file = wave.open(temp_file, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)


def audio_recording_callback(in_data, frame_count, time_info, status):
    """Callback for audio recording."""
    if wav_file is not None:
        wav_file.writeframes(in_data)
    return None, pyaudio.paContinue


def text_stream(text):
    """
    Stream text sentence by sentence for TTS and display.
    
    Args:
        text (str): Text to stream.
    
    Yields:
        str: Individual sentences.
    """
    sentences = re.split(r'(?<=[.!?]) +|\n', text)
    for sentence in sentences:
        if sentence.strip():
            print(Fore.YELLOW + sentence)
            write_text(f"{sentence}\n", "green")
            yield sentence


def on_closing():
    """Handle application closing."""
    global app_running
    
    app_running = False
    
    # Close audio stream
    try:
        if rec_stream and rec_stream._is_running:
            if rec_stream.is_active():
                rec_stream.stop_stream()
            rec_stream.close()
    except OSError:
        pass
    
    # Close wave file
    if wav_file:
        wav_file.close()
    
    # Terminate PyAudio
    if audio:
        audio.terminate()
    
    # Stop asyncio event loop
    if event_loop and event_loop.is_running():
        event_loop.call_soon_threadsafe(event_loop.stop)
    
    # Close Tkinter window
    root.destroy()


def main():
    """Initialize system and run the Tkinter main loop."""
    # Initialize system components
    initialize_system()
    
    # Run the Tkinter main loop
    root.mainloop()


def start_async_loop(loop):
    """Initialize and start the background event loop."""
    global event_loop
    event_loop = loop
    asyncio.set_event_loop(loop)
    loop.run_forever()


# ============================================================================
# GUI SETUP
# ============================================================================

# Create the main window
root = tk.Tk()
root.title("Multimodal Intelligent Driving Assistance System (MM-IDAS)")
root.configure(bg="#2c2f35")

# Set the protocol for handling window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window to full screen
root.geometry(f"{screen_width}x{screen_height}")

# Button frame
button_frame = tk.Frame(root, bg="#2c2f35")
button_frame.pack(pady=30)

# "New Chat" button
btn_new_conversation = tk.Button(
    button_frame,
    text="New\nChat",
    font=("Verdana", 32),
    command=new_session,
    height=8,
    width=6,
    bg="#406ce5",
    fg="black"
)
btn_new_conversation.pack(side=tk.LEFT, padx=10)

# "Start listening" button
btn_listening = tk.Button(
    button_frame,
    text="Start listening",
    font=("Verdana", 64),
    command=toggle_state,
    height=4,
    width=20,
    bg="#3f704d",
    fg="black"
)
btn_listening.pack(side=tk.LEFT, padx=20)

# Text frame (for conversation history)
text_frame = tk.Frame(root, bg="#2c2f35")
text_frame.pack(pady=10)

# Text box with scrollbar
text_box = tk.Text(
    text_frame,
    font=("Verdana", 14),
    height=8,
    width=80,
    wrap="word",
    bg="#1e1e1e",
    fg="white"
)
text_box.pack(side=tk.LEFT)

scrollbar = tk.Scrollbar(text_frame, command=text_box.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_box.config(yscrollcommand=scrollbar.set)

# Image frame (for manual images)
image_frame = tk.Frame(root, bg="#2c2f35")
image_frame.pack(pady=20)

# Label for image display
image_label = tk.Label(
    image_frame,
    bg="#1e1e1e",
    width=200,
    height=200
)
image_label.pack()

# Run the application
if __name__ == "__main__":
    # Required for Windows multiprocessing support
    import multiprocessing
    multiprocessing.freeze_support()
    
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_async_loop, args=(loop,), daemon=True).start()
    main()
