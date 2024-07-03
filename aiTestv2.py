import os
from io import BytesIO
import pyaudio
import wave
from pynput import keyboard
import speech_recognition as sr
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import simpleaudio as sa
import numpy as np
import pygame
import threading

# Retrieve API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if ELEVENLABS_API_KEY is None:
    raise ValueError("API key not found. Please set ELEVENLABS_API_KEY as an environment variable.")

# Initialize ElevenLabs client
ttsClient = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize PyAudio and pygame
p = pyaudio.PyAudio()
pygame.mixer.init()

# Global variables
recording = False
recording_thread = None

def list_audio_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    devices = []
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        devices.append(device_info)
        print(f"Device {i}: {device_info['name']}")
    return devices

def select_audio_device():
    devices = list_audio_devices()
    device_index = 0
    print(f"Selected Device {device_index}: {devices[device_index]['name']}")
    return device_index

def record_audio():
    global recording
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "input.wav"
    
    device_index = select_audio_device()
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
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

def on_press(key):
    global recording, recording_thread
    if key == keyboard.Key.ctrl_r and not recording:
        recording = True
        print("Recording started...")
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

def on_release(key):
    global recording, recording_thread
    if key == keyboard.Key.shift_l and recording:
        recording = False
        print("Recording stopped...")
        if recording_thread is not None:
            recording_thread.join()
            print("Recording thread joined")

def ensure_correct_buffer_size(audio_data, num_channels, bytes_per_sample):
    buffer_size = len(audio_data)
    correct_size = (buffer_size // (num_channels * bytes_per_sample)) * (num_channels * bytes_per_sample)
    return audio_data[:correct_size]

def play_audio(audio_stream):
    audio_stream.seek(0)
    audio_data = audio_stream.read()
    wave_obj = sa.WaveObject(audio_data, num_channels=1, bytes_per_sample=2, sample_rate=44100)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def text_to_speech_stream(text: str):
    print("Converting text to speech:", text)
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate, False)
    wave = np.sin(440 * t * 2 * np.pi)
    audio = wave * (2**15 - 1) / np.max(np.abs(wave))
    audio = audio.astype(np.int16)
    audio = ensure_correct_buffer_size(audio, 1, 2)
    audio_stream = BytesIO()
    audio_stream.write(audio.tobytes())
    audio_stream.seek(0)
    play_audio(audio_stream)

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

def generate_response(user_input):
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(user_input.lower(), "Sorry, I didn't understand that.")

def handle_conversation():
    text = "Hey, welcome to sports media's speechbot, I am svaz, how can I help you today"
    text_to_speech_stream(text)
    print("\nAI:", text)
    record = "AI: " + text + "\n"

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            print("\nPress Right Ctrl to start recording, Left Shift to stop.")
            while not recording:
                pass
            while recording:
                pass

            transcript_result = transcribe_audio("input.wav")
            record += "User: " + transcript_result + "\n"
            print("\nYou:", transcript_result)

            if transcript_result.lower() == "end":
                break

            ai_response = generate_response(transcript_result)
            record += "AI: " + ai_response + "\n"
            print("\nAI:", ai_response)
            text_to_speech_stream(ai_response)

    finally:
        listener.stop()

if __name__ == "__main__":
    list_audio_devices()
    handle_conversation()
