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

[... El resto del código hasta la sección de la interfaz permanece igual ...]

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
