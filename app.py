import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json

# Modelos de datos
class Tool(BaseModel):
    name: str
    description: str
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class SumTool(Tool):
    name: str = "sum_numbers"
    description: str = "Suma dos números"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        num1 = kwargs.get('num1', 0)
        num2 = kwargs.get('num2', 0)
        result = float(num1) + float(num2)
        return {"result": result}

class MultiplyTool(Tool):
    name: str = "multiply_numbers"
    description: str = "Multiplica dos números"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        num1 = kwargs.get('num1', 0)
        num2 = kwargs.get('num2', 0)
        result = float(num1) * float(num2)
        return {"result": result}

class MathAgent(BaseModel):
    name: str
    description: str
    tools: Dict[str, Tool] = {}
    history: List[Dict[str, Any]] = []
    
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": "Herramienta no encontrada"}
        
        result = self.tools[tool_name].execute(**kwargs)
        
        # Guardar en el historial
        operation = {
            "tool": tool_name,
            "inputs": kwargs,
            "result": result
        }
        self.history.append(operation)
        
        return result

# Configuración de la página
st.set_page_config(page_title="Agente Matemático", layout="wide")

# Inicialización del agente (solo una vez)
if 'agent' not in st.session_state:
    agent = MathAgent(
        name="MathBot",
        description="Un agente que realiza operaciones matemáticas"
    )
    agent.add_tool(SumTool())
    agent.add_tool(MultiplyTool())
    st.session_state.agent = agent

# Interfaz de usuario
st.title("🤖 Agente Matemático")
st.write("Este agente puede realizar operaciones matemáticas básicas")

# Selección de herramienta
tool_name = st.selectbox(
    "Selecciona la operación",
    options=list(st.session_state.agent.tools.keys()),
    format_func=lambda x: "Suma" if x == "sum_numbers" else "Multiplicación"
)

# Entrada de números
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("Primer número", value=0.0)
with col2:
    num2 = st.number_input("Segundo número", value=0.0)

# Botón para calcular
if st.button("Calcular"):
    result = st.session_state.agent.use_tool(tool_name, num1=num1, num2=num2)
    st.write("### Resultado:")
    st.write(result["result"])

# Historial de operaciones
if st.checkbox("Mostrar historial de operaciones"):
    st.write("### Historial")
    for operation in st.session_state.agent.history:
        st.write(f"Operación: {operation['tool']}")
        st.write(f"Números: {operation['inputs']}")
        st.write(f"Resultado: {operation['result']}")
        st.write("---")

