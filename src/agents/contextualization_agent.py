from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from exceptions import APIError
from logger import get_logger

load_dotenv()
logger = get_logger("contextualization-agent")

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un analista legal especializado en comparación estructural de contratos.

Se te proporcionarán dos documentos: el contrato original y su adenda.

--- TAREA ---

Debes realizar un mapeo estructural EXHAUSTIVO entre ambos documentos.

### PASO 1 — IDENTIFICACIÓN DE CLÁUSULAS
- Identificá todas las secciones o cláusulas de cada documento.

### PASO 2 — LISTADO COMPLETO
- Construí mentalmente dos listas:
- Lista completa de cláusulas del contrato original
- Lista completa de cláusulas del contrato modificado
- NO omitir ninguna cláusula, incluso si no tiene equivalente en el otro documento.

### PASO 3 — TABLA FINAL
Devolvé exclusivamente una tabla Markdown con estas columnas:

| **SECCIÓN / CLÁUSULA** | **CONTRATO ORIGINAL** | **CONTRATO MODIFICADO** | **TEMA PRINCIPAL** | **CORRESPONDENCIA** |
|---|---|---|---|---|


SECCIÓN / CLÁUSULA: Nombre de la seccion / cláusula
CONTRATO ORIGINAL:  Escribir "Presente" si la cláusula existe en el contrato original.
                    Escribir "N/A" si NO existe en el contrato original.

CONTRATO MODIFICADO: Escribir "Presente" si la cláusula existe en el contrato modificado.
                     Escribir "N/A" si NO existe en el contrato modificado.

TEMA PRINCIPAL: tema principal

CORRESPONDENCIA: Completar con una de estas categorías:
    - Idéntica: mismo contenido en ambos.
    - Similar: mismo contenido en ambos pero con cambios menores en el contrato modificado de redacción ó estilo, cuyo signficado semántico es el mismo.
    - Modificada: cambios relevantes en condiciones, obligaciones ó términos en el contrato modificado.
    - Nueva: solo existe en el contrato modificado.
    - Eliminada: solo existe en el contrato original.


EJEMPLO:

OBJETO | Presente | Presente | Objeto del contrato | Idéntica
PROHIBICIONES | Presente | N/A |Restricciones de uso | Eliminada
SEGURO CONTRA INCENDIOS | N/A | Presenete | Obligación de seguro | Nueva


--- REGLAS ---
- Incluir TODAS las cláusulas de ambos documentos.
- No agrupar cláusulas distintas.
- No omitir cláusulas eliminadas o nuevas.
- Mantener nombres EXACTOS de las cláusulas.
- Si una cláusula no existe en uno de los documentos → usar "N/A".
- No agregar texto fuera de la tabla.

Antes de generar la tabla, verificá que la cantidad de filas sea igual a la suma de:
- cláusulas del original
- cláusulas nuevas del modificado
- cláusulas eliminadas
- verificá la columna correspondencia de la tabla contra las clásulas del contrato original y el modificado""",
        ),
        (
            "human",
            """--- CONTRATO ORIGINAL ---
{original_text}

--- CONTRATO MODIFICADO (ADENDA) ---
{amendment_text}""",
        ),
    ]
)


def run_contextualization_agent(original_text: str, amendment_text: str, model: str):
    """Analiza la estructura de ambos contratos y devuelve el AIMessage con content y usage_metadata."""
    logger.info("Iniciando generación del mapa contextual.")

    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )

    chain = _PROMPT | llm
    try:
        response = chain.invoke(
            {
                "original_text": original_text,
                "amendment_text": amendment_text,
            }
        )
    except Exception as e:
        raise APIError("openai", str(e)) from e

    logger.info("Mapa contextual generado.")
    return response
