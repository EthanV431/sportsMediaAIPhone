import os
from io import BytesIO
import pyaudio
import wave
from pynput import keyboard
import speech_recognition as sr
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import simpleaudio as sa  # Added import statement

# Retrieve API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Ensure the key is set
if ELEVENLABS_API_KEY is None:
    raise ValueError("API key not found. Please set ELEVENLABS_API_KEY as an environment variable.")

# Initialize ElevenLabs client
ttsClient = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize PyAudio
p = pyaudio.PyAudio()

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

# Key press handler
def on_press(key):
    global recording
    if key == keyboard.Key.ctrl_r and not recording:  # Use alt_l for left Alt key
        recording = True
        print("Recording started...")
        record_audio(select_audio_device())
    elif key == keyboard.Key.esc:
        recording = False
        print("Recording stopped...")
        return False  # Stop listener

# Key release handler
def on_release(key):
    global recording
    if key == keyboard.Key.alt_l:
        recording = False
        print("Recording stopped...")

# Function to play audio
def play_audio(audio_stream):
    audio_data = audio_stream.read()
    wave_obj = sa.WaveObject(audio_data, num_channels=1, bytes_per_sample=2, sample_rate=44100)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Function to convert text to speech and play it
def text_to_speech_stream(text: str):
    response = ttsClient.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
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

# Conversation loop
def handle_conversation():
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

# Start conversation
handle_conversation()
