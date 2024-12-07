import streamlit as st
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import load_qa_chain
from langchain.callbacks import get_openai_callback
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

class RAGTool(Tool):
    name: str = "rag_query"
    description: str = "Consulta documentos usando RAG"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        knowledge_base = kwargs.get('knowledge_base')
        question = kwargs.get('question')
        sensor_data = kwargs.get('sensor_data', {})
        
        if not knowledge_base or not question:
            return {"error": "Falta la base de conocimiento o la pregunta"}
        
        # Enriquecer la pregunta con datos del sensor
        if sensor_data:
            enhanced_question = f"""
            Contexto actual del sensor:
            - Temperatura: {sensor_data.get('Temp', 'N/A')}°C
            - Humedad: {sensor_data.get('Hum', 'N/A')}%
            
            Pregunta del usuario:
            {question}
            """
        else:
            enhanced_question = question
            
        docs = knowledge_base.similarity_search(enhanced_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=enhanced_question)
            usage = str(cb)
        
        return {
            "response": response,
            "usage": usage
        }

class SmartAgent(BaseModel):
    name: str
    description: str
    tools: Dict[str, Tool] = {}
    history: List[Dict[str, Any]] = []
    knowledge_base: Optional[Any] = None
    sensor_data: Optional[Dict[str, Any]] = None
    
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": "Herramienta no encontrada"}
        
        result = self.tools[tool_name].execute(**kwargs)
        
        operation = {
            "tool": tool_name,
            "inputs": kwargs,
            "result": result
        }
        self.history.append(operation)
        
        return result
    
    def set_knowledge_base(self, kb):
        self.knowledge_base = kb
        
    def set_sensor_data(self, data):
        self.sensor_data = data

# Funciones auxiliares
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

# Inicialización del agente
if 'agent' not in st.session_state:
    agent = SmartAgent(
        name="UMI",
        description="Asistente inteligente para análisis de datos y consultas"
    )
    agent.add_tool(TemperatureAnalysisTool())
    agent.add_tool(HumidityAnalysisTool())
    agent.add_tool(RAGTool())
    st.session_state.agent = agent

# Configuración de OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"]

# UI Principal
st.title("🤖 UMI - Asistente Inteligente")

# Cargar animación
with open('umbirdp.json') as source:
    animation = json.load(source)
st_lottie(animation, width=350)

# Crear directorio temporal si no existe
try:
    os.mkdir("temp")
except:
    pass

# Cargar y procesar PDF
pdfFileObj = open('plantas.pdf', 'rb')
pdf_reader = PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Crear base de conocimiento
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=20,
    length_function=len
)
chunks = text_splitter.split_text(text)
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)
st.session_state.agent.set_knowledge_base(knowledge_base)

# Layout principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            if sensor_data:
                st.session_state.agent.set_sensor_data(sensor_data)
                st.success("Datos recibidos")
                st.metric("Temperatura", f"{sensor_data.get('Temp', 'N/A')}°C")
                st.metric("Humedad", f"{sensor_data.get('Hum', 'N/A')}%")
                
                # Análisis automático
                temp_analysis = st.session_state.agent.use_tool(
                    "analyze_temperature",
                    temperature=float(sensor_data['Temp']),
                    operation='comfort'
                )
                hum_analysis = st.session_state.agent.use_tool(
                    "analyze_humidity",
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
    
    if user_question:
        with st.spinner('Analizando tu pregunta...'):
            result = st.session_state.agent.use_tool(
                "rag_query",
                knowledge_base=knowledge_base,
                question=user_question,
                sensor_data=st.session_state.agent.sensor_data
            )
            
            st.write("### Respuesta:")
            st.write(result["response"])
            
            # Botón de audio
            if st.button("Escuchar"):
                result_audio, _ = text_to_speech(result["response"], 'es-es')
                audio_file = open(f"temp/{result_audio}.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)

# Historial
if st.checkbox("Mostrar historial de operaciones"):
    st.write("### Historial")
    for operation in st.session_state.agent.history:
        st.write(f"Operación: {operation['tool']}")
        st.write(f"Resultado: {operation['result']}")
        st.write("---")

# Cerrar archivo PDF
pdfFileObj.close()

