import os
from io import BytesIO
import pyaudio
import wave
from pynput import keyboard
import speech_recognition as sr
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import simpleaudio as sa  # Added import statement
import numpy as np
import pygame
import threading
# Retrieve API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Ensure the key is set
if ELEVENLABS_API_KEY is None:
    raise ValueError("API key not found. Please set ELEVENLABS_API_KEY as an environment variable.")
# Initialize ElevenLabs client
ttsClient = ElevenLabs(api_key=ELEVENLABS_API_KEY)
# Initialize PyAudio
p = pyaudio.PyAudio()
pygame.mixer.init()
# Function to list available audio input devices
def list_audio_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    devices = []
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        devices.append(device_info)
        print(f"Device {i}: {device_info['name']}")
    return devices
# Select an appropriate audio input device
def select_audio_device():
    devices = list_audio_devices()
    # Select the first device (or any other logic to select the correct device)
    device_index = 0
    print(f"Selected Device {device_index}: {devices[device_index]['name']}")
    return device_index
# Function to record audio
def record_audio(device_index):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "input.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    # Recording loop
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
# Global variable to control recording state
recording = False
recording_thread = None
# Key press handler
def on_press(key):
    global recording, recording_thread
    if key == keyboard.Key.ctrl_r and not recording:
        recording = True
        print("Recording started...")
        device_index = select_audio_device()
        recording_thread = threading.Thread(target=record_audio, args=(device_index,))
        recording_thread.start()
# Key release handler
def on_release(key):
    global recording, recording_thread
    if key == keyboard.Key.shift_l and recording:
        recording = False
        print("Recording stopped...")
        if recording_thread is not None:
            recording_thread.join()  # Ensure the recording thread finishes
            print("Recording thread joined")
def ensure_correct_buffer_size(audio_data, num_channels, bytes_per_sample):
    buffer_size = len(audio_data)
    correct_size = (buffer_size // (num_channels * bytes_per_sample)) * (num_channels * bytes_per_sample)
    return audio_data[:correct_size]
# Function to play audio
def play_audio(audio_stream):
    audio_stream.seek(0)
    audio_data = audio_stream.read()
    wave_obj = sa.WaveObject(audio_data, num_channels=1, bytes_per_sample=2, sample_rate=44100)
    play_obj = wave_obj.play()
    play_obj.wait_done()
# Function to convert text to speech and play it
def text_to_speech_stream(text: str):
    print("Converting text to speech:", text)
    # Dummy sine wave for example purposes
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate, False)
    wave = np.sin(440 * t * 2 * np.pi)
    audio = wave * (2**15 - 1) / np.max(np.abs(wave))
    audio = audio.astype(np.int16)

    # Ensure the buffer size is correct
    audio = ensure_correct_buffer_size(audio, 1, 2)

    # Create an audio stream
    audio_stream = BytesIO()
    audio_stream.write(audio.tobytes())
    audio_stream.seek(0)

    play_audio(audio_stream)

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"
# Basic rule-based response generation
def generate_response(user_input):
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(user_input.lower(), "Sorry, I didn't understand that.")
# Function to handle call initialization
def initialize_call():
    # Placeholder for actual call initialization logic
    print("Call initiated")
    # Play a greeting message
    text_to_speech_stream("Welcome to the AI Telephone service. How can I help you today?")
# Function to handle call termination
def terminate_call():
    # Placeholder for actual call termination logic
    print("Call terminated")
    # Play a thank you message
    text_to_speech_stream("Thank you for using our service. Goodbye!")
def authenticate_user():
    global recording, recording_thread
    print("Please say your name to authenticate.")
    with keyboard.Listener(on_press = on_press, on_release = on_release) as listener:
        listener.join()
    transcript_result = transcribe_audio("input.wav")
    print("\nYou: ", transcript_result)
    # Placeholder for actual authentication logic
    authorized_users = ["Rob Smith", "Roger White"]
    if transcript_result in authorized_users:
        text_to_speech_stream(f"Welcome, {transcript_result}. Authentication successful.")
        return True
    else:
        text_to_speech_stream("Authentication failed. Please try again.")
        return False
def handle_conversation():
    if authenticate_user():
        initialize_call()
        text = "Hey, welcome to sports media's speechbot, I am svaz, how can I help you today"
        text_to_speech_stream(text)
        print("\nAI:", text)
        record = "AI: " + text + "\n"
        while True:
            # Wait for user to start recording
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
            transcript_result = transcribe_audio("input.wav")
            record += "User: " + transcript_result + "\n"
            print("\nYou: ", transcript_result)
            if transcript_result.lower() == "end":
                break
            ai_response = generate_response(transcript_result)
            record += "AI: " + ai_response + "\n"
            print("\nAI: ", ai_response)
            text_to_speech_stream(ai_response)
        terminate_call()

list_audio_devices()
# Start conversation
handle_conversation()