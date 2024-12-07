import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
import PyPDF2
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import time
from streamlit_lottie import st_lottie
import json
import paho.mqtt.client as mqtt
import pytz
from typing import List, Union, Optional
import re

# Configuración MQTT
MQTT_BROKER = "157.230.214.127"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Variables de estado para los datos del sensor
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

# Clase para el template del prompt
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

# Clase para procesar la salida del agente
class CustomOutputParser:
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Acción Final:" in text:
            return AgentFinish(
                return_values={"output": text.split("Acción Final:")[-1].strip()},
                log=text
            )
            
        match = re.search(r"Acción:\s*(.*?)\nEntrada:\s*(.*)", text, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": text.strip()},
                log=text
            )
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=text)

# Funciones de utilidad
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

def analyze_temperature(temp: float) -> dict:
    temp = float(temp)
    if temp < 18:
        return {"status": "Frío", "recommendation": "Considere aumentar la temperatura"}
    elif temp > 26:
        return {"status": "Caliente", "recommendation": "Considere reducir la temperatura"}
    return {"status": "Confortable", "recommendation": "Temperatura ideal"}

def analyze_humidity(humidity: float) -> dict:
    humidity = float(humidity)
    if humidity < 30:
        return {"status": "Baja", "recomendación": "Use un humidificador"}
    elif humidity > 60:
        return {"status": "Alta", "recomendación": "Use un deshumidificador"}
    return {"status": "Óptima", "recomendación": "Nivel ideal"}

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

# Título y animación
st.title('UMI - Asistente Inteligente 💬')
with open('umbirdp.json') as source:
    animation = json.load(source)
st_lottie(animation, width=350)

# Crear directorio temporal
try:
    os.mkdir("temp")
except:
    pass

# Configuración de OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"]

# Procesamiento del PDF y configuración del agente
pdf_path = 'plantas.pdf'
if os.path.exists(pdf_path):
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
    
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # Herramientas del agente
    tools = [
        Tool(
            name="Consultar_Documento",
            func=lambda q: knowledge_base.similarity_search(q)[0].page_content,
            description="Útil para buscar información en el documento sobre plantas"
        ),
        Tool(
            name="Analizar_Temperatura",
            func=analyze_temperature,
            description="Analiza si la temperatura es adecuada"
        ),
        Tool(
            name="Analizar_Humedad",
            func=analyze_humidity,
            description="Analiza si la humedad es adecuada"
        )
    ]
    
    # Template para el prompt
    template = """Eres un asistente experto en plantas y condiciones ambientales.
    
    Tienes acceso a las siguientes herramientas:
    {tools}
    
    Usa el siguiente formato:
    Pregunta: la pregunta que debes responder
    Pensamiento: piensa paso a paso qué debes hacer
    Acción: la acción a tomar (una de {tool_names})
    Entrada: la entrada para la herramienta
    Observación: el resultado de la acción
    ... (este patrón Pensamiento/Acción/Entrada/Observación puede repetirse N veces)
    Pensamiento: Ahora sé la respuesta final
    Acción Final: la respuesta final

    Asegúrate de incluir información relevante sobre las condiciones ambientales actuales en tu respuesta.
    
    {agent_scratchpad}
    
    Pregunta: {input}
    Pensamiento:"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps", "agent_scratchpad"]
    )
    
    output_parser = CustomOutputParser()
    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservación:"],
        allowed_tools=[tool.name for tool in tools],
        input_keys=["input", "agent_scratchpad"]
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

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
                temp = sensor_data.get('Temp')
                hum = sensor_data.get('Hum')
                
                st.metric("Temperatura", f"{temp}°C")
                st.metric("Humedad", f"{hum}%")
                
                temp_analysis = analyze_temperature(float(temp))
                hum_analysis = analyze_humidity(float(hum))
                
                st.write("### Análisis")
                st.write(f"Temperatura: {temp_analysis['status']}")
                st.write(f"Recomendación: {temp_analysis['recommendation']}")
                st.write(f"Humedad: {hum_analysis['status']}")
                st.write(f"Recomendación: {hum_analysis['recomendación']}")
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Consulta al Asistente")
    st.info("""
    Ejemplos de preguntas que puedes hacer:
    - ¿Qué plantas son buenas para la temperatura actual?
    - ¿Cómo afecta la humedad actual a las plantas?
    - ¿Qué cuidados necesitan las plantas en estas condiciones?
    """)
    
    user_question = st.text_area("¿Qué deseas saber?")
    
    if user_question and 'agent_executor' in locals():
        with st.spinner('Procesando tu consulta...'):
            try:
                response = agent_executor.run({
                    'input': user_question,
                    'agent_scratchpad': ''
                })
                
                st.write("### Respuesta:")
                st.write(response)
                
                if st.button("Escuchar"):
                    result_audio, _ = text_to_speech(response, 'es-es')
                    audio_file = open(f"temp/{result_audio}.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3", start_time=0)
                    
            except Exception as e:
                st.error(f"Error al procesar la consulta: {str(e)}")

# Información en la barra lateral
with st.sidebar:
    st.subheader("Acerca del Asistente")
    st.write("""
    Este asistente puede:
    - Responder preguntas sobre plantas
    - Analizar condiciones ambientales
    - Proporcionar recomendaciones personalizadas
    - Convertir respuestas a audio
    
    Basado en los datos del sensor y la información del documento.
    """)
