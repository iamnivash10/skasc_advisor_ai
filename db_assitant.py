import pysqlite3 as sqlite3
import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from few_shots import few_shots  # Import the few-shot examples
from gtts import gTTS
import tempfile
import os
import sqlite3
print(sqlite3.sqlite_version)



# Load API key securely
GROQ_API_KEY = st.secrets["api"]["key"]

# Display image at the top center
st.markdown("""
    <div style="text-align: center;">
        <img src="OIP (2).jpeg" width="150">
    </div>
""", unsafe_allow_html=True)

# Center all elements using Streamlit
st.markdown("""
    <div style="text-align: center;">
        <h1>🎤 SKASC Drug Addiction Counselor AI</h1>
    </div>
""", unsafe_allow_html=True)

username = st.text_input("Enter your name:", "John", key="username", help="Enter your name")
response_placeholder = st.empty()


def text_to_speech(text):
    """Convert text to speech and return the file path."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_path = temp_audio.name
        tts.save(temp_path)
    return temp_path


def voice_input():
    """Listen for user input and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            user_text = recognizer.recognize_google(audio)
            return user_text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the speech."
        except sr.RequestError:
            return "Error with the speech recognition service."


# Initialize LLM and Vectorstore once
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=2,
)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
documents = [
    Document(
        page_content=f"User: {example['UserQuery']}\nAdvisor: {example['AdvisorResponse']}",
        metadata=example
    ) for example in few_shots
]
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")


def get_advice(user_input, username):
    """Generate AI response strictly based on examples."""
    selected_examples = vectorstore.similarity_search(user_input, k=1)

    if selected_examples:
        example_query = selected_examples[0].metadata["UserQuery"]
        example_response = selected_examples[0].metadata["AdvisorResponse"]
    else:
        example_query, example_response = "No example found.", "Provide a general response."

    formatted_prompt = (
        f"You are a compassionate drug addiction counselor.\n"
        f"Always structure your response exactly like the example provided below.\n"
        f"If unsure, generate a response similar to the closest example.\n\n"
        f"Example:\nUser: {example_query}\nAdvisor: {example_response}\n\n"
        f"User ({username}): {user_input}\nAdvisor:"
    )

    response = llm.invoke(formatted_prompt)
    final_response = f"{response.content}"
    audio_path = text_to_speech(final_response)
    return final_response, audio_path


if "recording" not in st.session_state:
    st.session_state.recording = False
if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = ""

if st.button("🎙️ Start Listening", key="start-listening"):
    user_query = voice_input()

    if user_query.strip():
        response_placeholder.markdown(f"<div style='text-align: center;'><strong>You:</strong> {user_query}</div>",
                                      unsafe_allow_html=True)
        advice, audio_path = get_advice(user_query, username)
        response_placeholder.markdown(f"<div style='text-align: center;'><strong>Advisor:</strong> {advice}</div>",
                                      unsafe_allow_html=True)

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        # Automatically play the audio
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        # Provide a download button for the audio
        st.markdown("""
            <div style="text-align: center;">
                <a href="data:audio/mp3;base64," download="advisor_response.mp3">📥 Download Advice Audio</a>
            </div>
        """, unsafe_allow_html=True)
        os.remove(audio_path)
