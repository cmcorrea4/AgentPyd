import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
import numpy as np
from typing import List, Union
import json
import time
import paho.mqtt.client as mqtt
from gtts import gTTS

# Configuración MQTT
MQTT_BROKER = "157.230.214.127"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Variables de estado
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = None

class ClaudeEmbeddings:
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.claude = ChatAnthropic(model=model)
    
    def embed_text(self, text: str) -> List[float]:
        """Genera un embedding usando Claude"""
        prompt = f"""Por favor, analiza el siguiente texto y genera un vector de embedding de 1536 dimensiones. 
        El vector debe capturar la esencia semántica del texto. Responde SOLO con los números del vector, 
        separados por comas.
        
        Texto: {text}
        """
        
        try:
            response = self.claude.invoke(prompt)
            # Limpiar y convertir la respuesta a una lista de números
            vector_str = response.content.strip().replace('[', '').replace(']', '')
            vector = [float(x) for x in vector_str.split(',')]
            # Normalizar el vector
            vector = np.array(vector)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
        except Exception as e:
            st.error(f"Error al generar embedding: {e}")
            # Retornar un vector de ceros como fallback
            return [0.0] * 1536

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para una lista de textos"""
        return [self.embed_text(text) for text in texts]

def process_pdf(pdf_path: str) -> List[str]:
    """Procesa el PDF y retorna chunks de texto"""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.session_state.document_chunks = chunks
        return chunks
    except Exception as e:
        st.error(f"Error al procesar PDF: {e}")
        return []

def semantic_search(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Realiza búsqueda semántica usando Claude"""
    embeddings = ClaudeEmbeddings()
    query_embedding = embeddings.embed_text(query)
    
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    # Calcular similitud coseno
    similarities = []
    for chunk_emb in chunk_embeddings:
        similarity = np.dot(query_embedding, chunk_emb)
        similarities.append(similarity)
    
    # Obtener los top_k chunks más relevantes
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def get_mqtt_message():
    """Obtiene datos del sensor MQTT"""
    message_received = {"received": False, "payload": None}
    
    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
            message_received["payload"] = payload
            message_received["received"] = True
        except Exception as e:
            st.error(f"Error al procesar mensaje: {e}")
    
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        
        timeout = time.time() + 5
        while not message_received["received"] and time.time() < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        return message_received["payload"]
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def analyze_with_claude(query: str, context: List[str], sensor_data: dict = None) -> str:
    """Analiza la consulta usando Claude con contexto y datos del sensor"""
    llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
    
    prompt = f"""Analiza la siguiente consulta usando el contexto proporcionado y los datos del sensor.
    
    Consulta: {query}
    
    Contexto relevante del documento:
    {' '.join(context)}
    
    Datos del sensor: {sensor_data if sensor_data else 'No disponibles'}
    
    Por favor:
    1. Proporciona un análisis detallado
    2. Relaciona la información del documento con los datos del sensor cuando sea relevante
    3. Ofrece recomendaciones específicas
    4. Destaca cualquier patrón o anomalía importante
    """
    
    response = llm.invoke(prompt)
    return response.content

def text_to_speech(text: str, tld: str = 'es-es') -> tuple:
    """Convierte texto a voz"""
    tts = gTTS(text=text, lang='es', tld=tld, slow=False)
    file_name = text[:20]
    file_path = f"temp/{file_name}.mp3"
    tts.save(file_path)
    return file_name, text

# Configuración de la página
st.set_page_config(page_title="Asistente Claude", layout="wide")
st.title('Asistente Inteligente con Claude 🤖')

# Crear directorio temporal
os.makedirs("temp", exist_ok=True)

# Input para API key
api_key = st.text_input('Ingresa tu API Key de Anthropic:', type='password')
if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

# Procesamiento inicial del PDF
pdf_path = 'plantas.pdf'
if os.path.exists(pdf_path) and st.session_state.document_chunks is None:
    with st.spinner('Procesando documento base...'):
        st.session_state.document_chunks = process_pdf(pdf_path)

# Interfaz principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            if sensor_data:
                st.session_state.sensor_data = sensor_data
                st.success("Datos recibidos")
                
                for key, value in sensor_data.items():
                    st.metric(key, value)
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Consulta al Asistente")
    st.info("""
    Puedes preguntar sobre:
    - Información del documento base
    - Datos del sensor
    - Recomendaciones basadas en las condiciones actuales
    """)
    
    user_question = st.text_area("¿Qué deseas saber?")
    
    if user_question and api_key:
        with st.spinner('Analizando tu consulta...'):
            try:
                # Buscar contexto relevante
                relevant_chunks = []
                if st.session_state.document_chunks:
                    relevant_chunks = semantic_search(user_question, st.session_state.document_chunks)
                
                # Obtener respuesta
                response = analyze_with_claude(
                    user_question, 
                    relevant_chunks, 
                    st.session_state.sensor_data
                )
                
                st.session_state.last_response = response
                
                st.write("### Respuesta:")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error al procesar la consulta: {str(e)}")

    # Botón de audio
    if st.session_state.last_response:
        if st.button("Escuchar"):
            try:
                result_audio, _ = text_to_speech(st.session_state.last_response)
                audio_file = open(f"temp/{result_audio}.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
            except Exception as e:
                st.error(f"Error al generar el audio: {str(e)}")

# Información en la barra lateral
with st.sidebar:
    st.subheader("Acerca del Asistente")
    st.write("""
    Este asistente utiliza exclusivamente Claude para:
    - Procesar y entender documentos
    - Generar embeddings para búsqueda semántica
    - Analizar datos de sensores
    - Proporcionar respuestas contextuales
    - Generar recomendaciones personalizadas
    
    Powered by Claude 3.5 Sonnet
    """)
