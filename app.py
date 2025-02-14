import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
import paho.mqtt.client as mqtt
import json
import time
from typing import List, Union
import re
from PIL import Image
from gtts import gTTS

# Configuración MQTT
MQTT_BROKER = "broker.mqttdashboard.com"
MQTT_PORT = 1883
MQTT_TOPIC = "Sensor/THP"

# Variables de estado
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.get("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAcción: {action}\nObservación: {observation}\n"
        
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        if "input" not in kwargs:
            kwargs["input"] = ""
            
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Acción Final:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Acción Final:")[-1].strip()},
                log=llm_output
            )
        
        match = re.search(r"Acción:\s*(.*?)\nEntrada:\s*(.*)", llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output
            )
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def get_mqtt_message():
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

def analyze_data(data: dict) -> dict:
    """Analiza los datos proporcionados usando Claude"""
    llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
    response = llm.invoke(
        f"""Analiza los siguientes datos y proporciona insights relevantes: {data}
        Por favor incluye:
        1. Patrones principales
        2. Anomalías
        3. Recomendaciones
        """
    )
    return response

def text_to_speech(text, tld='es-es'):
    tts = gTTS(text=text, lang='es', tld=tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text

# Configuración de la página
st.set_page_config(page_title="Asistente de Análisis con Claude", layout="wide")
st.title('Asistente de Análisis con Claude 🤖')

# Crear directorio temporal si no existe
os.makedirs("temp", exist_ok=True)

# Input para la API key de Anthropic
api_key = st.secrets["settings"]["key"]
if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

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
                
                # Análisis automático con Claude
                analysis = analyze_data(sensor_data)
                st.write("### Análisis")
                st.write(analysis)
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Consulta al Asistente")
    st.info("""
    Asegúrate de:
    1. Obtener una lectura del sensor antes de hacer preguntas
     
    Puedes preguntar sobre:
    - Análisis de patrones en los datos
    - Recomendaciones basadas en las lecturas
    - Comparaciones con valores óptimos
    """)
    
    user_question = st.text_area("¿Qué deseas analizar?")
    
    if user_question and api_key:
        if not st.session_state.sensor_data:
            st.warning("Por favor, obtén primero una lectura del sensor")
        else:
            with st.spinner('Analizando tu consulta...'):
                try:
                    llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
                    response = llm.invoke(
                        f"""Analiza la siguiente pregunta sobre estos datos: {st.session_state.sensor_data}
                        Pregunta: {user_question}
                        
                        Por favor:
                        1. Proporciona un análisis detallado
                        2. Incluye recomendaciones específicas
                        3. Destaca cualquier patrón o anomalía relevante
                        """
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
    Este asistente puede:
    - Conectarse a sensores MQTT
    - Analizar datos en tiempo real
    - Proporcionar insights usando Claude
    - Convertir respuestas a audio
    - Generar recomendaciones personalizadas
    
    Utiliza el modelo Claude 3.5 Sonnet para análisis avanzado.
    """)
