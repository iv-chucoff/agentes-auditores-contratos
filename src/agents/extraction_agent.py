import os

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from exceptions import APIError, ParsedOutputError
from logger import get_logger
from models import ContractChangeOutput

load_dotenv()
logger = get_logger("extraction-agent")

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un auditor legal experto en identificar cambios entre versiones de contratos.

Se te proporcionará un mapa contextual que describe la estructura de ambos documentos,
y los textos completos del contrato original y la adenda.

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
- Para MODIFICACIONES, siempre usá el formato: valor original → valor nuevo. Nunca cites solo el valor nuevo.
""",
        ),
        (
            "human",
            """--- MAPA CONTEXTUAL ---
{contextual_map}

--- CONTRATO ORIGINAL ---
{original_text}

--- CONTRATO MODIFICADO (ADENDA) ---
{amendment_text}""",
        ),
    ]
)


def run_extraction_agent(
    original_text: str,
    amendment_text: str,
    contextual_map: str,
    model: str
) -> ContractChangeOutput:
    """Identifica y describe todos los cambios entre el contrato original y la adenda.

    Retorna un ContractChangeOutput con los cambios detectados.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        timeout=60,
        max_retries=2,
    ).with_structured_output(ContractChangeOutput)

    logger.info("Iniciando extracción de cambios.")

    chain = _PROMPT | llm
    try:
        result = chain.invoke(
            {
                "contextual_map": contextual_map,
                "original_text": original_text,
                "amendment_text": amendment_text,
            }
        )
    except OutputParserException as e:
        raise ParsedOutputError(str(e)) from e
    except Exception as e:
        raise APIError("openai", str(e)) from e

    logger.info(f"Extracción completada. Secciones modificadas: {len(result.sections_changed)}.")
    return result
