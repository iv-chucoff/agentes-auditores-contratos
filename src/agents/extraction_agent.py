import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import ContractChangeOutput

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template(
"""
    Eres un auditor legal experto en identificar cambios entre versiones de contratos.

    Se te proporciona un mapa contextual que describe la estructura de ambos documentos,
    y los textos completos del contrato original y la adenda.

    --- MAPA CONTEXTUAL ---
    {contextual_map}

    --- CONTRATO ORIGINAL ---
    {original_text}

    --- CONTRATO MODIFICADO (ADENDA) ---
    {amendment_text}

    --- TAREA ---
    Identificá y describí TODOS los cambios introducidos por la adenda:
    - ADICIONES: texto o cláusulas nuevas que no existían en el original
    - ELIMINACIONES: texto o cláusulas del original que fueron removidas
    - MODIFICACIONES: texto o cláusulas que existían en el original pero fueron alteradas

    Si no hay cambios en alguna categoría, indicalo explícitamente con "Sin cambios."

    --- REGLAS ---
    - Sé exhaustivo: no omitas ningún cambio, por menor que parezca.
    - Sé preciso: citá los identificadores exactos de cada cláusula, sección o anexo afectado.
    - No interpretes la intención legal de los cambios, solo describilos objetivamente.
    - Si un párrafo fue reordenado sin cambios de contenido, mencionalo como cambio estructural menor.
    - Basá tu análisis únicamente en los textos proporcionados, sin asumir información externa.

    --- FORMATO DE RESPUESTA ---
    Devolvé tu análisis en el siguiente formato. No agregues texto fuera de él.

    sections_changed: ["Cláusula 3", "Anexo A"]
    topics_touched: ["precio", "plazo", "responsabilidad"]

    summary_of_the_change:
      1. ADICIONES:
         - [Identificador]: [Descripción objetiva del contenido agregado.]
      2. ELIMINACIONES:
         - [Identificador]: [Descripción objetiva del contenido removido.]
      3. MODIFICACIONES:
         - [Identificador]: [Texto anterior → Texto nuevo.]
"""
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
