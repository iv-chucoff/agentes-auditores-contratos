import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template(
    """Eres un analista legal especializado en comparación de contratos.
Se te proporcionan dos documentos: el contrato original y su adenda (contrato modificado).

--- CONTRATO ORIGINAL ---
{original_text}

--- CONTRATO MODIFICADO (ADENDA) ---
{amendment_text}

Tu tarea es construir un mapa contextual comparado. Para cada documento identifica:
1. Las secciones o cláusulas presentes (con su numeración o nombre)
2. El propósito o tema principal de cada sección
3. Cómo se corresponden las secciones entre ambos documentos

No extraigas los cambios todavía. Solo construye el mapa estructural que permita entender
qué sección del original corresponde a qué sección de la adenda, y cuál es el contenido esperado de cada bloque.

Responde con texto estructurado claro."""
)


def run_contextualization_agent(original_text: str, amendment_text: str, model: str):
    """Analiza la estructura de ambos contratos y devuelve el AIMessage con content y usage_metadata."""
    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
    )

    chain = _PROMPT | llm
    return chain.invoke(
        {
            "original_text": original_text,
            "amendment_text": amendment_text,
        }
    )
