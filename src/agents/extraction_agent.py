import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import ContractChangeOutput

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template(
    """Eres un analista legal experto en identificar cambios entre versiones de contratos.

Se te proporciona un mapa contextual que describe la estructura de ambos documentos, \
y los textos completos del contrato original y la adenda.

--- MAPA CONTEXTUAL ---
{contextual_map}

--- CONTRATO ORIGINAL ---
{original_text}

--- CONTRATO MODIFICADO (ADENDA) ---
{amendment_text}

Tu tarea es identificar y describir TODOS los cambios introducidos por la adenda:
- ADICIONES: texto o cláusulas nuevas que no existían en el original
- ELIMINACIONES: texto o cláusulas del original que fueron removidas
- MODIFICACIONES: texto o cláusulas que existían pero fueron alteradas

Devuelve los resultados en el siguiente formato:
- sections_changed: lista de identificadores de las secciones modificadas (ej: "Cláusula 3", "Anexo A")
- topics_touched: lista de categorías legales/comerciales afectadas (ej: "precio", "plazo", "responsabilidad")
- summary_of_the_change: descripción detallada y precisa de todos los cambios aplicados"""
)


def run_extraction_agent(
    original_text: str,
    amendment_text: str,
    contextual_map: str,
    model: str
) -> dict:
    """Identifica y describe todos los cambios entre el contrato original y la adenda.

    Retorna el dict crudo de include_raw=True con keys 'parsed', 'raw' y 'parsing_error'.
    """
    llm = ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
    ).with_structured_output(ContractChangeOutput, include_raw=True)

    chain = _PROMPT | llm
    return chain.invoke(
        {
            "contextual_map": contextual_map,
            "original_text": original_text,
            "amendment_text": amendment_text,
        }
    )
