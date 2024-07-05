from dotenv import load_dotenv
import pyaudio
import wave
import requests
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import io
import os


load_dotenv()

WAVE_OUTPUT_FILENAME = "output.wav"

def record_audio(record_seconds=5, wave_output_filename=WAVE_OUTPUT_FILENAME):
    # Audio stream parameters
    FORMAT = pyaudio.paInt16  # Sample format
    CHANNELS = 1  # Mono audio
    RATE = 44100  # Sample rate
    CHUNK = 1024  # Frames per buffer

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Record data in chunks for the duration of RECORD_SECONDS
    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")

    # Save the recorded data as a WAV file
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    
# Function to convert audio to text
def text_to_audio(text):
    voice_id = "XB0fDUnXU5powFXDhCwa"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.5,
            "use_speaker_boost": True
        },
    }

    headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv('ELEVENLABS_API_KEY')
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # Convert the response content (audio data) into an audio segment
        audio_segment = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        # Export the audio segment to an MP3 file
        audio_segment.export("output.mp3", format="mp3")
        # Use afplay to play the exported MP3 file
        os.system("afplay output.mp3")
        print("Audio played successfully.")
    else:
        print(f"Error: {response.text}")

# Function to get response from GPT-4
def get_gpt_response(text):
    api_key = os.getenv('OPENAI_API_KEY')
    gpt_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    response = requests.post(gpt_url, headers=headers, json=data)
    
    # print("Request to GPT-4 API:")
    # print(text)
    # print("Response from GPT-4 API:")
    # print(response.json())

    # Check if the response contains an error
    if 'error' in response.json():
        print("Error from GPT-4 API:", response.json()['error'])
        return "Error in generating response."
    
    # Safely access 'choices' if no error
    try:
        gpt_response = response.json()['choices'][0]['message']['content']
        return gpt_response
    except KeyError:
        print("KeyError: 'choices' not found in the response.")
        return "Failed to generate response."

# Main function to handle the chatbot logic
def chatbot(audio_path):
    while True:
        record_audio()

        text = audio_to_text(audio_path)

        if "goodbye" in text.lower() or not text.strip():
            text_to_audio("Goodbye! Have a great day!")
            break

        response = get_gpt_response(text)
        print (response)
        text_to_audio(response)

        

# Use the recorded audio file
audio_path = WAVE_OUTPUT_FILENAME

chatbot(audio_path)
