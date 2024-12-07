import os
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import paho.mqtt.client as mqtt
import time
from streamlit_lottie import st_lottie
from gtts import gTTS
import glob

# Configuraciones MQTT
MQTT_BROKER = "157.230.214.127"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Modelos de datos
class Tool(BaseModel):
    name: str
    description: str
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class TemperatureAnalysisTool(Tool):
    name: str = "analyze_temperature"
    description: str = "Analiza datos de temperatura"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        temp = kwargs.get('temperature', 0)
        operation = kwargs.get('operation', 'convert')
        
        if operation == 'convert':
            fahrenheit = (temp * 9/5) + 32
            kelvin = temp + 273.15
            return {
                "celsius": temp,
                "fahrenheit": fahrenheit,
                "kelvin": kelvin
            }
        elif operation == 'comfort':
            if temp < 18:
                return {"status": "Frío", "recommendation": "Considere aumentar la temperatura"}
            elif temp > 26:
                return {"status": "Caliente", "recommendation": "Considere reducir la temperatura"}
            else:
                return {"status": "Confortable", "recommendation": "Temperatura ideal"}

class HumidityAnalysisTool(Tool):
    name: str = "analyze_humidity"
    description: str = "Analiza datos de humedad"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        humidity = kwargs.get('humidity', 0)
        
        if humidity < 30:
            status = "Baja"
            recommendation = "Considere usar un humidificador"
        elif humidity > 60:
            status = "Alta"
            recommendation = "Considere usar un deshumidificador"
        else:
            status = "Óptima"
            recommendation = "Nivel de humedad ideal"
            
        return {
            "humidity": humidity,
            "status": status,
            "recommendation": recommendation,
            "comfort_index": min(100, max(0, (humidity - 30) * 100 / 30))
        }

# Funciones MQTT
def get_mqtt_message():
    """Función para obtener un único mensaje MQTT"""
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

# Función de texto a voz
def text_to_speech(text, tld):
    tts = gTTS(text, "es", tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text

# Configuración de la página
st.set_page_config(page_title="UMI - Asistente Inteligente", layout="wide")

# Configuración de OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"]

# UI Principal
st.title("🤖 UMI - Asistente Inteligente")

# Cargar animación
with open('umbird.json') as source:
    animation = json.load(source)
st_lottie(animation, width=350)

# Crear directorio temporal
try:
    os.mkdir("temp")
except:
    pass

# Inicialización de herramientas
temp_tool = TemperatureAnalysisTool()
humidity_tool = HumidityAnalysisTool()

# Procesar PDF
pdf_path = 'plantas.pdf'
if os.path.exists(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # Crear embeddings y base de conocimiento
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Inicializar embeddings y vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Crear cadena de RAG
    template = """Pregunta: {question}
    Contexto: {context}
    Respuesta útil:"""
    
    prompt = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Layout principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            if sensor_data:
                st.success("Datos recibidos")
                st.metric("Temperatura", f"{sensor_data.get('Temp', 'N/A')}°C")
                st.metric("Humedad", f"{sensor_data.get('Hum', 'N/A')}%")
                
                # Análisis de temperatura
                temp_analysis = temp_tool.execute(
                    temperature=float(sensor_data['Temp']),
                    operation='comfort'
                )
                # Análisis de humedad
                hum_analysis = humidity_tool.execute(
                    humidity=float(sensor_data['Hum'])
                )
                
                st.write("### Análisis de Temperatura")
                st.write(f"Estado: {temp_analysis['status']}")
                st.write(f"Recomendación: {temp_analysis['recommendation']}")
                
                st.write("### Análisis de Humedad")
                st.write(f"Estado: {hum_analysis['status']}")
                st.write(f"Recomendación: {hum_analysis['recommendation']}")
                st.progress(int(hum_analysis['comfort_index']))
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Realiza tu consulta")
    user_question = st.text_area("Escribe tu pregunta aquí:")
    
    if user_question and 'rag_chain' in locals():
        with st.spinner('Analizando tu pregunta...'):
            response = rag_chain.invoke(user_question)
            
            st.write("### Respuesta:")
            st.write(response)
            
            # Botón de audio
            if st.button("Escuchar"):
                result_audio, _ = text_to_speech(response, 'es-es')
                audio_file = open(f"temp/{result_audio}.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
