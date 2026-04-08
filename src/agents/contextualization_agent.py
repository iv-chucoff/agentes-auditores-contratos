import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_PROMPT = ChatPromptTemplate.from_template(
"""
    Eres un analista legal especializado en comparación de contratos.
    Se te proporcionan dos documentos: el contrato original y su adenda.

    --- CONTRATO ORIGINAL ---
    {original_text}

    --- CONTRATO MODIFICADO (ADENDA) ---
    {amendment_text}

    --- TAREA ---
    Construí un mapa estructural comparado entre ambos documentos.
    Para cada sección o cláusula identificá:
    1. Su nombre exacto tal como aparece en el documento.
    2. El número de cláusula en el contrato original (o N/A si no existe).
    3. El número de cláusula en el contrato modificado (o N/A si no existe).
    4. Un resumen breve (1 oración) de su propósito o tema principal.
    5. Una descripción de cómo se corresponden ambas versiones: si son idénticas, 
       similares con cambios, renombradas, nuevas, o eliminadas.

    No analices ni describas los cambios de contenido todavía.
    Solo mapeá la estructura.

    --- FORMATO DE RESPUESTA ---
    Devolvé exclusivamente una tabla Markdown con exactamente estas columnas y en este orden. 
    No agregues ningún texto antes ni después.

    La tabla debe tener estas columnas: 
    
    | **SECCIÓN / CLÁUSULA** | **CONTRATO ORIGINAL** | **CONTRATO MODIFICADO** | **TEMA PRINCIPAL** | **CORRESPONDENCIA** |
    |---|---|---|---|---|
    | 1. OBJETO | 1 | 1 | Descripción del inmueble objeto de la locación. | Ambas secciones son idénticas, describiendo el mismo inmueble. |
    | 8. SEGURO CONTRA INCENDIOS | N/A | 8 | Nueva obligación de contratar un seguro. | Esta sección es nueva en la adenda y no tiene equivalente en el original. |
    | 9. PROHIBICIONES | 9 | N/A | Restricciones sobre el uso del inmueble. | Esta sección no está presente en la adenda. |
"""
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
