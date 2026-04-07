import base64
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_contract_image(image_paths: list[str], model: str) -> AIMessage:
    """Recibe una lista de paths de imágenes (páginas de un contrato),
    las codifica en base64 y las envía a GPT-4o para extraer el texto completo.

    Retorna el AIMessage de LangChain con .content y .usage_metadata.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
    )

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Transcribe todo el texto visible en estas imágenes"
            ),
        }
    ]

    for path in image_paths:
        base64_img = _encode_image(path)
        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64_img}",
                    "detail": "high",
                },
            }
        )

    response: AIMessage = llm.invoke([
        SystemMessage(content="""Tu función es extraer texto de imágenes.
Comienza desde la primera imagen y continúa hasta la última. No omitas ninguna parte de la imagen.

- Respeta mayúsculas, minúsculas, signos de puntuación y saltos de línea.
- Mantén la estructura original (párrafos, listas, alineaciones básicas).

Reglas especiales:

- Los campos en blanco o espacios para completar deben representarse como: [________].
- Si encuentras texto ilegible o borroso, escribe: [ILEGIBLE].
- Las firmas deben indicarse como: [FIRMA].
- Los sellos deben indicarse como: [SELLO].

Restricciones:

- No agregues formato Markdown.
- No interpretes el contenido.
- No resumas.
- No corrijas errores tipográficos.
- No agregues información."""),
        HumanMessage(content=content),
    ])
    return response
