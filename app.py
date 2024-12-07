import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
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

# Variables de estado
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'token_info' not in st.session_state:
    st.session_state.token_info = None
if 'intermediate_steps' not in st.session_state:
    st.session_state.intermediate_steps = None

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
        
    @property
    def _type(self) -> str:
        return "custom_output_parser"

def get_current_conditions() -> str:
    """Obtiene las condiciones actuales del sensor."""
    if 'sensor_data' in st.session_state and st.session_state.sensor_data:
        temp = st.session_state.sensor_data.get('Temp', 'N/A')
        hum = st.session_state.sensor_data.get('Hum', 'N/A')
        return f"Temperatura actual: {temp}°C, Humedad actual: {hum}%"
    return "No hay datos disponibles del sensor. Por favor, obtén una lectura primero."

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

def analyze_temperature(temp: str) -> dict:
    try:
        temp = float(temp)
        if temp < 18:
            return {"status": "Frío", "recommendation": "Considere aumentar la temperatura"}
        elif temp > 26:
            return {"status": "Caliente", "recommendation": "Considere reducir la temperatura"}
        return {"status": "Confortable", "recommendation": "Temperatura ideal"}
    except:
        return {"status": "Error", "recommendation": "No se pudo analizar la temperatura"}

def analyze_humidity(humidity: str) -> dict:
    try:
        humidity = float(humidity)
        if humidity < 30:
            return {"status": "Baja", "recomendación": "Use un humidificador"}
        elif humidity > 60:
            return {"status": "Alta", "recomendación": "Use un deshumidificador"}
        return {"status": "Óptima", "recomendación": "Nivel ideal"}
    except:
        return {"status": "Error", "recomendación": "No se pudo analizar la humedad"}

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

try:
    with open('umbird.json') as source:
        animation = json.load(source)
    st_lottie(animation, width=350)
except Exception as e:
    st.error(f"Error al cargar la animación: {e}")

# Crear directorio temporal
try:
    os.mkdir("temp")
except:
    pass

# Configuración de OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"]

# Procesamiento del PDF
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
            func=lambda x: analyze_temperature(st.session_state.sensor_data.get('Temp', 0)) if st.session_state.sensor_data else {"status": "Error", "recommendation": "No hay datos del sensor"},
            description="Analiza la temperatura actual del ambiente"
        ),
        Tool(
            name="Analizar_Humedad",
            func=lambda x: analyze_humidity(st.session_state.sensor_data.get('Hum', 0)) if st.session_state.sensor_data else {"status": "Error", "recomendación": "No hay datos del sensor"},
            description="Analiza la humedad actual del ambiente"
        ),
        Tool(
            name="Consultar_Condiciones_Actuales",
            func=lambda _: get_current_conditions(),
            description="Obtiene las condiciones actuales de temperatura y humedad"
        )
    ]
    
    # Template para el prompt
    template = """Eres un asistente experto en plantas y condiciones ambientales.
    
    Tienes acceso a las siguientes herramientas:
    {tools}
    
    IMPORTANTE: Antes de responder cualquier pregunta sobre condiciones ambientales, 
    SIEMPRE usa primero la herramienta Consultar_Condiciones_Actuales para obtener los datos más recientes.
    
    Usa el siguiente formato:
    Pregunta: la pregunta que debes responder
    Pensamiento: piensa paso a paso qué debes hacer
    Acción: la acción a tomar (una de {tool_names})
    Entrada: la entrada para la herramienta
    Observación: el resultado de la acción
    ... (este patrón Pensamiento/Acción/Entrada/Observación puede repetirse N veces)
    Pensamiento: Ahora sé la respuesta final
    Acción Final: la respuesta final
    
    Asegúrate de:
    1. Consultar siempre las condiciones actuales primero
    2. Analizar si las condiciones son adecuadas usando las herramientas correspondientes
    3. Buscar información específica sobre plantas en el documento
    4. Proporcionar recomendaciones basadas en las condiciones actuales
    
    {agent_scratchpad}
    
    Pregunta: {input}
    Pensamiento:"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps", "agent_scratchpad"]
    )
    
    output_parser = CustomOutputParser()
    llm = OpenAI(temperature=0, model_name="gpt-4o-mini")
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
                
                temp_analysis = analyze_temperature(temp)
                hum_analysis = analyze_humidity(hum)
                
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
    Asegúrate de obtener una lectura del sensor antes de hacer preguntas.
    
    Ejemplos de preguntas que puedes hacer:
    - Considerando la temperatura y humedad actuales, ¿qué plantas me recomiendas?
    - ¿Las condiciones actuales son buenas para plantas tropicales?
    - ¿Qué ajustes necesito hacer en el ambiente para mejorar las condiciones?
    """)
    
    user_question = st.text_area("¿Qué deseas saber?")
    
    if user_question and 'agent_executor' in locals():
        if not st.session_state.sensor_data:
            st.warning("Por favor, obtén primero una lectura del sensor usando el botón 'Obtener Lectura'")
        else:
            # Botón específico para realizar la consulta
            if st.button("Consultar"):
                with st.spinner('Procesando tu consulta...'):
                    try:
                        with get_openai_callback() as cb:
                            result = agent_executor(
                                {
                                    'input': user_question,
                                    'agent_scratchpad': ''
                                }
                            )
                            
                            st.session_state.last_response = result['output']
                            st.session_state.intermediate_steps = result['intermediate_steps']
                            
                            # Guardar información de tokens
                            st.session_state.token_info = {
                                'total_tokens': cb.total_tokens,
                                'prompt_tokens': cb.prompt_tokens,
                                'completion_tokens': cb.completion_tokens,
                                'total_cost': cb.total_cost
                            }
                            
                    except Exception as e:
                        st.error(f"Error al procesar la consulta: {str(e)}")
                        st.error("Por favor, intenta reformular tu pregunta")
    
    # Mostrar la respuesta si existe
    if st.session_state.last_response:
        st.write("### Respuesta:")
        st.write(st.session_state.last_response)
        
        # Mostrar información de tokens
        if st.session_state.token_info:
            st.write("### Información de uso:")
            st.write(f"Total de Tokens: {st.session_state.token_info['total_tokens']}")
            st.write(f"Tokens de Prompt: {st.session_state.token_info['prompt_tokens']}")
            st.write(f"Tokens de Completion: {st.session_state.token_info['completion_tokens']}")
            st.write(f"Costo Total: ${st.session_state.token_info['total_cost']:.4f}")
        
        # Mostrar proceso de razonamiento
        if st.checkbox("Mostrar proceso de razonamiento"):
            st.write("### Proceso de razonamiento:")
            for step in st.session_state.intermediate_steps:
                st.write(f"**Acción:** {step[0]}")
                st.write(f"**Resultado:** {step[1]}")
                st.write("---")
        
        # Botón de audio separado
        if st.button("Escuchar", key="audio_button"):
            try:
                result_audio, _ = text_to_speech(st.session_state.last_response, 'es-es')
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
    - Responder preguntas sobre plantas
    - Analizar condiciones ambientales en tiempo real
    - Proporcionar recomendaciones personalizadas
    - Convertir respuestas a audio
    - Mostrar el proceso de razonamiento
    - Mostrar uso de tokens y costos
    
    Basado en los datos del sensor y la información del documento.
    """)
