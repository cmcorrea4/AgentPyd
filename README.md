# AgentPyd
# UMI - Asistente Inteligente para Plantas 🌿

UMI es un asistente inteligente que combina IoT y procesamiento de lenguaje natural para proporcionar recomendaciones personalizadas sobre el cuidado de plantas basadas en condiciones ambientales en tiempo real.

## 🌟 Características

- Monitoreo en tiempo real de temperatura y humedad
- Análisis de condiciones ambientales
- Recomendaciones personalizadas basadas en datos del sensor
- Integración con base de conocimientos sobre plantas
- Conversión de texto a voz
- Interfaz intuitiva con Streamlit
- Visualización del proceso de razonamiento del agente

## 🛠️ Tecnologías Utilizadas

- Python 3.9
- Streamlit
- LangChain
- OpenAI GPT-4o-mini
- MQTT
- FAISS para búsqueda vectorial
- PyPDF2 para procesamiento de documentos
- gTTS para conversión de texto a voz

## 📋 Requisitos Previos

```bash
# Instalar las dependencias
langchain==0.0.154
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.18.1
faiss-cpu==1.7.4
streamlit-extras
openai==0.28
tiktoken
gTTS==2.2.2
googletrans==3.1.0a0
streamlit-lottie
altair==4.2.0
paho-mqtt==1.6.1
pytz
```

## ⚙️ Configuración

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/umi-assistant.git
cd umi-assistant
```

2. Crear un archivo `.streamlit/secrets.toml` con tu API key de OpenAI:
```toml
[settings]
key = "tu-api-key-de-openai"
```

3. Configurar el broker MQTT:
```python
MQTT_BROKER = "tu-broker-mqtt"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"
```

4. Colocar tu base de conocimientos en un archivo PDF llamado `plantas.pdf`

## 🚀 Uso

1. Iniciar la aplicación:
```bash
streamlit run app.py
```

2. En la interfaz web:
   - Clic en "Obtener Lectura" para conseguir datos actuales del sensor
   - Hacer preguntas sobre plantas considerando las condiciones actuales
   - Ver el análisis detallado y recomendaciones

## 💡 Ejemplos de Preguntas

- "¿Qué plantas son recomendables para la temperatura actual?"
- "Con esta humedad, ¿qué cuidados especiales necesitan las plantas?"
- "¿Las condiciones actuales son buenas para cactus?"
- "¿Qué ajustes necesito hacer en el ambiente para mejorar las condiciones?"

## 🤖 Cómo Funciona el Agente

El agente utiliza cuatro herramientas principales:

1. **Consultar_Documento**: Busca información en la base de conocimientos
2. **Analizar_Temperatura**: Evalúa las condiciones de temperatura
3. **Analizar_Humedad**: Evalúa las condiciones de humedad
4. **Consultar_Condiciones_Actuales**: Obtiene datos en tiempo real

Para cada consulta, el agente:
1. Obtiene las condiciones actuales
2. Analiza si son adecuadas
3. Busca información relevante
4. Proporciona recomendaciones personalizadas

## 📊 Estructura del Proyecto

```
umi-assistant/
├── app.py              # Aplicación principal
├── plantas.pdf         # Base de conocimientos
├── requirements.txt    # Dependencias
├── umbird.json        # Archivo de animación
├── .streamlit/        # Configuración de Streamlit
│   └── secrets.toml   # Secretos (API keys)
└── temp/              # Archivos temporales de audio
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del repositorio
2. Crea una rama para tu función
3. Realiza tus cambios
4. Envía un pull request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- [Tu Nombre](https://github.com/cmcorrea4)

## 🙏 Agradecimientos

- OpenAI por el modelo GPT-4o-mini
- Comunidad de Streamlit por sus herramientas y documentación
- Contribuidores del proyecto LangChain

## 📞 Contacto

Para preguntas y soporte, por favor crea un issue en el repositorio o contacta a [cmcorrea4@gmail.com]
