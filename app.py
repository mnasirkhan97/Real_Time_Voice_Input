import streamlit as st
from transformers import pipeline
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the Hugging Face model on the correct device
qa_pipeline = pipeline("text-generation", model="gpt2", device=device)

st.title("Real-Time Voice-Enabled Interview Chatbot")
st.write("Speak your questions or type them to get interview preparation answers.")

# Function to get live audio input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            question = recognizer.recognize_google(audio)
            st.write(f"You asked: {question}")
            return question
        except sr.UnknownValueError:
            st.write("Could not understand audio.")
        except sr.RequestError:
            st.write("Voice service unavailable. Please check your connection.")
    return None

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text)
    audio_output = BytesIO()
    tts.write_to_fp(audio_output)
    st.audio(audio_output, format="audio/mp3")

# Select input type
input_type = st.radio("Choose your input method:", ("Text", "Voice"))

question = None
if input_type == "Text":
    question = st.text_input("Type your interview question here:")
elif input_type == "Voice":
    if st.button("Record Question"):
        question = get_voice_input()

# If a question is asked, generate and display the answer
if st.button("Get Answer") and question:
    # Generate response based on the question
    response = qa_pipeline(question, max_length=50, do_sample=True)
    answer_text = response[0]['generated_text']
    st.write(f"**Answer:** {answer_text}")
    
    # Option to play the answer as audio
    if st.button("Play Answer"):
        text_to_speech(answer_text)
