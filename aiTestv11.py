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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random

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

    # Function to collect user feedback
    def collect_user_feedback():
        text_to_speech_stream("Please provide your feedback about this call.")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        feedback = transcribe_audio("input.wav")
        with open("feedback.txt", "a") as file:
            file.write(feedback + "\n")
        text_to_speech_stream("Thank you for your feedback!")

    # Function to detect pauses in audio data
    def detect_pause(audio_data, threshold=0.1, min_silence_len=2000):
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
            schedule.every().day.at(callback_time).do(send_sms_notification, to_phone_number=phone_number,
                                                      message="This is a reminder for your scheduled callback.")
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
        play_obj = sa.play_buffer(beep.raw_data, channels=beep.channels, sample_width=beep.sample_width,
                                  frame_rate=beep.frame_rate)
        play_obj.wait_done()
        # Record message
        recording = True
        device_index = select_audio_device()
        recording_thread = threading.Thread(target=record_audio, args=(device_index,))
        recording_thread.start()
        time.sleep(10)  # Allow 10 seconds for message recording
        recording = False
        recording_thread.join()  # Ensure the recording thread finishes
        print("Recording stopped...")
        text_to_speech_stream("Message recorded. Thank you.")

    # Function to save conversation history in file
    def save_conversation_history(conversation_history):
        with open("conversation_history.txt", "a") as file:
            file.write(conversation_history + "\n")

    # Function to send SMS
    def send_sms(to_number, message):
        smsClient.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )

    # Function to transcribe voice messages
    def transcribe_voice_message(audio_file_path):
        return transcribe_audio(audio_file_path)

    # Encrypt user data
    def encrypt_data(data):
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        return encrypted_data, key

    # Decrypt user data
    def decrypt_data(encrypted_data, key):
        cipher_suite = Fernet(key)
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data.decode()

    # Schedule follow-up calls
    def schedule_follow_up(call_time, message):
        schedule.every().day.at(call_time).do(send_sms, "+1234567890", message)

    def send_voice_message(user_name, to_phone_number):
        # Construct the file name for the user's voice message
        voice_message_file = f"{user_name}_message.wav"
        # Check if the voice message file exists
        if os.path.exists(voice_message_file):
            # Read the audio data from the file
            audio_data = open(voice_message_file, 'rb').read()
            # Convert the audio data to a BytesIO stream for playback
            audio_stream = BytesIO(audio_data)
            # Play the audio stream (assuming play_audio is a predefined function)
            play_audio(audio_stream)
            # Print confirmation of the sent voice message
            print(f"Sent voice message from {user_name} to {to_phone_number}.")
        else:
            # Print an error message if no recorded message is found
            print("No recorded message found.")

    def perform_sentiment_analysis(user_input):
        # Initialize a sentiment analysis pipeline using the Hugging Face transformers library
        analyzer = pipeline("sentiment-analysis")
        # Analyze the sentiment of the user input and get the result
        result = analyzer(user_input)[0]
        # Return the sentiment analysis result
        return result

    def secure_data_storage(data):
        # Generate a key for encryption using the Fernet symmetric encryption algorithm
        key = Fernet.generate_key()
        # Initialize the Fernet cipher suite with the generated key
        cipher_suite = Fernet(key)
        # Encrypt the data using the cipher suite
        encrypted_data = cipher_suite.encrypt(data.encode())
        # Store the encrypted data in a file
        with open("encrypted_data.txt", "wb") as file:
            file.write(encrypted_data)
        # Print confirmation of secure data storage
        print("Data encrypted and stored securely.")
        # Return the encryption key
        return key

    def retrieve_secure_data(key):
        # Initialize the Fernet cipher suite with the provided key
        cipher_suite = Fernet(key)
        # Read the encrypted data from the file
        with open("encrypted_data.txt", "rb") as file:
            encrypted_data = file.read()
        # Decrypt the data using the cipher suite
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        # Print confirmation of data retrieval and decryption
        print("Data retrieved and decrypted successfully.")
        # Return the decrypted data as a string
        return decrypted_data.decode()

    # Function to send OTP for Multi-Factor Authentication (MFA)
    def send_otp(email):
        otp = random.randint(100000, 999999)
        msg = MIMEMultipart()
        msg['From'] = "noreply@yourdomain.com"
        msg['To'] = email
        msg['Subject'] = "Your OTP Code"
        body = f"Your OTP code is {otp}. Please use this code to verify your identity."
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("your-email@gmail.com", "your-password")
        text = msg.as_string()
        server.sendmail("noreply@yourdomain.com", email, text)
        server.quit()
        print(f"OTP sent to {email}")
        return otp

    # Function to integrate with a CRM system (Dummy Implementation)
    def sync_with_crm(user_data):
        # Simulate syncing with a CRM system
        print(f"Syncing user data with CRM: {user_data}")
        # Dummy return value
        return True

    # Function for Call Analytics (Dummy Implementation)
    def analyze_call_data(call_data):
        print(f"Analyzing call data: {call_data}")
        # Simulate some analysis
        call_duration = len(call_data) / 60  # Example: converting data length to minutes
        print(f"Call duration: {call_duration} minutes")
        # Dummy return value
        return {"call_duration": call_duration}

    # Function for AI-driven recommendations based on transcribed text
    def provide_recommendations(transcribed_text):
        print(f"Providing recommendations based on transcribed text: {transcribed_text}")
        # Dummy implementation of recommendation logic
        if "refund" in transcribed_text.lower():
            return "Would you like assistance with processing a refund?"
        elif "help" in transcribed_text.lower():
            return "It seems you need help. How can I assist you further?"
        else:
            return "Thank you for your input. Is there anything else I can do for you?"

        import speech_recognition as sr

        def recognize_command():
            recognizer = sr.Recognizer()
            microphone = sr.Microphone()

            with microphone as source:
                print("Listening for command...")
                audio = recognizer.listen(source)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Command recognized: {command}")

                if "send message" in command:
                    # Trigger send message functionality
                    send_message()
                elif "set reminder" in command:
                    # Trigger set reminder functionality
                    set_reminder()
                elif "make call" in command:
                    # Trigger make call functionality
                    make_call()
                else:
                    print("Command not recognized.")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        def send_message():
            print("Sending message...")

        def set_reminder():
            print("Setting reminder...")

        def make_call():
            print("Making call...")


import wave
import speech_recognition as sr
def record_call(filename="call_recording.wav", duration=60):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    frames = []

    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe_call(filename="call_recording.wav"):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(filename)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


# Usage
record_call()  # Record a call for 60 seconds by default
transcribe_call()  # Transcribe the recorded call

from elevenlabs import generate, play, VoiceSettings

def customize_tts(text, voice='default', pitch=0, speed=1.0):
    settings = VoiceSettings(pitch=pitch, speed=speed)
    audio = generate(text=text, voice=voice, settings=settings)
    play(audio)

# Usage
customize_tts("Hello, this is your custom voice speaking.", pitch=-2, speed=1.2)

# Provides support for multiple languages in TTS and speech recognition.
def recognize_multilingual_command(language="en-US"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print(f"Listening for command in {language}...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio, language=language).lower()
        print(f"Command recognized: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

# Usage
recognize_multilingual_command(language="es-ES")  # Listening for Spanish commands

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