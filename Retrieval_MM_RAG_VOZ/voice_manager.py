"""
Voice Manager Module for Multimodal Vehicle Assistant

This module handles all voice-related operations including:
- Audio recording with PyAudio
- Speech-to-text transcription with OpenAI Whisper
- Text-to-speech synthesis with ElevenLabs
- Audio playback and temporary file management
"""

import os
import wave
import pyaudio
import requests
import tempfile
import pydub
from pydub import playback
import openai

from elevenlabs.client import ElevenLabs

from config import (
    logger, ELEVEN_API_KEY, OPENAI_API_KEY, SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
    ELEVENLABS_MODEL_ID, VOICE_STABILITY, VOICE_SIMILARITY_BOOST
)

# Define AUDIO_FORMAT here since it requires pyaudio
AUDIO_FORMAT = pyaudio.paInt16


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
        if self.voices and len(self.voices) > 23:
            self.voice_id = self.voices[23]["voice_id"]
            logger.info("Voz seleccionada: %s (ID: %s)", self.voices[23]["name"], self.voice_id)
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
            with open(audio_filepath, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
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
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": VOICE_STABILITY,
                "similarity_boost": VOICE_SIMILARITY_BOOST
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