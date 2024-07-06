import os
import time
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
import logging
from transformers import pipeline
import webrtcvad
from voice_recognition_module import VoiceAuthenticator
import requests
from twilio.rest import Client
from cryptography.fernet import Fernet
import schedule
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
# Function to authenticate a user
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
# Function to log errors
def log_error(error_message):
    logging.basicConfig(filename = 'error_log.txt', level = logging.ERROR)
    logging.error(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
# Function to generate dynamic responses using a transformer model
def generate_dynamic_response(user_input):
    nlp = pipeline("conversational", model = "microsoft/DialoGPT-medium")
    response = nlp(user_input)
    return response[0]['generated_text']
# Function to detect speech using WebRTC VAD
def detect_speech(audio_frame, vad):
    is_speech = vad.is_speech(audio_frame, sample_rate = 1000)
    return is_speech
# Function to perform voice authentication
def voice_authentication(audio_file_path):
    authenticator = VoiceAuthenticator()
    user_name = authenticator.authenticate(audio_file_path)
    return user_name
# Function to save conversation history in file
def save_conversation_history(conversation_history):
    with open("conversation_history.txt", "a") as file:
        file.write(conversation_history + "\n")
# Function to fetch information from an API
def fetch_information(query):
    response = requests.get(f"https://api.example.com/search?query={query}")
    if response.status_code == 300:
        return response.json()['result']
    else:
        return "Sorry, I couldn't fetch the information right now."
# Function to collect user feedback
def collect_user_feedback():
    text_to_speech_stream("Please provide your feedback about this call.")
    with keyboard.Listener(on_press = on_press, on_release = on_release) as listener:
        listener.join()
    feedback = transcribe_audio("input.wav")
    with open("feedback.txt", "a") as file:
        file.write(feedback + "\n")
    text_to_speech_stream("Thank you for your feedback!")
# Function to detect pauses in audio data
def detect_pause(audio_data, threshold = 0.1, min_silence_len = 2000):
    silent_chunks = 0
    for chunk in audio_data:
        if np.abs(chunk).mean() < threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks * chunk.duration_seconds > min_silence_len:
            return True
    return False
# Function to register a new user to the service
def register_user(user_name, phone_number):
    user_data = {
        "name": user_name,
        "phone_number": phone_number
    }
    store_user_data(user_data)
    print(f"User {user_name} registered successfully.")
    send_sms_notification(phone_number, "You have been successfully registered to the AI Telephone service.")

# Function to store user data securely
def store_user_data(user_data):
    with open("user_data.txt", "a") as file:
        file.write(f"{user_data['name']}:{user_data['phone_number']}\n")
# Function to load user data for authentication
def load_user_data():
    user_data = {}
    with open("user_data.txt", "r") as file:
        for line in file:
            name, phone = line.strip().split(":")
            user_data[name] = phone
    return user_data
# Function to send SMS notifications to users
def send_sms_notification(to_phone_number, message):
    smsClient.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=to_phone_number
    )
    print(f"Sent SMS to {to_phone_number}: {message}")
# Function to generate a usage report for the admin
def generate_usage_report():
    with open("conversation_history.txt", "r") as file:
        history = file.read()
    # Placeholder for actual report generation logic
    print("Generating usage report...")
    print(history)
# Function to update user profile information
def update_user_profile(user_name, new_phone_number):
    user_data = load_user_data()
    if user_name in user_data:
        user_data[user_name] = new_phone_number
        with open("user_data.txt", "w") as file:
            for name, phone in user_data.items():
                encrypted_data = encrypt_data(f"{name}:{phone}")
                file.write(f"{encrypted_data}\n")
        print(f"User {user_name}'s profile updated successfully.")
        send_sms_notification(new_phone_number, "Your profile has been updated.")
    else:
        print("User not found.")
# Function to schedule automatic callbacks
def schedule_callback(user_name, callback_time):
    user_data = load_user_data()
    if user_name in user_data:
        phone_number = user_data[user_name]
        schedule.every().day.at(callback_time).do(send_sms_notification, to_phone_number=phone_number, message="This is a reminder for your scheduled callback.")
        print(f"Callback scheduled for {user_name} at {callback_time}.")
    else:
        print("User not found.")
# Function to leave a voice message
def leave_voice_message(user_name):
    global recording, recording_thread
    print("Please leave your message after the beep.")
    text_to_speech_stream("Please leave your message after the beep.")
    # Beep sound
    beep = AudioSegment.from_wav("beep.wav")
    play_obj = sa.play_buffer(beep.raw_data, num_channels=beep.channels, bytes_per_sample=beep.sample_width, sample_rate=beep.frame_rate)
    play_obj.wait_done()
    # Start recording
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    # Save the recorded message
    user_data = load_user_data()
    if user_name in user_data:
        with open(f"{user_name}_message.wav", "wb") as file:
            with open("input.wav", "rb") as recorded_file:
                file.write(recorded_file.read())
        print(f"Message from {user_name} saved successfully.")
    else:
        print("User not found.")
# Function to support multiple languages
def text_to_speech_stream(text: str, language='en'):
    print("Converting text to speech:", text)
    tts_settings = VoiceSettings(voice="female", language=language)
    audio_data = ttsClient.generate_speech(text, voice_settings=tts_settings)
    audio_stream = BytesIO(audio_data)
    play_audio(audio_stream)
# Enhanced error handling and notifications
def log_error(error_message):
    logging.basicConfig(filename='error_log.txt', level=logging.ERROR)
    logging.error(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
    send_sms_notification(TWILIO_PHONE_NUMBER, f"Error occurred: {error_message}")
list_audio_devices()
# Start conversation
handle_conversation()
# Registering a new user
register_user("Jack Smith", "+1234567890")
# Updating user profile
update_user_profile("Alice Doe", "+0987654321")
# Scheduling a callback
schedule_callback("Alice Doe", "15:30")
# Leaving a voice message
leave_voice_message("Alice Doe")