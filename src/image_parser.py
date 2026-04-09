import base64

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from exceptions import APIError, ImageReadError, InvalidImageFormatError
from logger import get_logger

load_dotenv()
logger = get_logger("image-parser")

_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise ImageReadError(image_path, "archivo no encontrado")
    except PermissionError:
        raise ImageReadError(image_path, "sin permisos de lectura")
    except OSError as e:
        raise ImageReadError(image_path, str(e))


def _validate_image_format(path: str) -> str:
    """Valida la extensión y retorna el MIME type correspondiente."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext not in _MIME_TYPES:
        raise InvalidImageFormatError(path, ext)
    return _MIME_TYPES[ext]


def parse_contract_image(image_paths: list[str], model: str, contract_type: str) -> AIMessage:
    """Recibe una lista de paths de imágenes (páginas de un contrato),
    las codifica en base64 y las envía a GPT-4o para extraer el texto completo.

    Retorna el AIMessage de LangChain con .content y .usage_metadata.
    """
    logger.info(f"Iniciando transcripción de contrato {contract_type}")

    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        timeout=60,
        max_retries=2,
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
        #Valido que la imagen sea de un formato correcto
        mime = _validate_image_format(path)
        #Base 64, en caso de no poder leer los archivos, capturo el
        base64_img = _encode_image(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64_img}",
                    "detail": "high",
                },
            }
        )

    try:
        response: AIMessage = llm.invoke([
            SystemMessage(content="""Tu función es extraer texto de imágenes.
            Comienza desde la primera imagen y continúa hasta la última. No omitas ninguna parte de la imagen.

            - Respetá mayúsculas, minúsculas, signos de puntuación y saltos de línea.
            - Mantén la estructura original (párrafos, listas, alineaciones básicas).
            - Todas las imágenes pertenecen al MISMO documento y deben ser interpretadas como un texto continuo.

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
    except Exception as e:
        raise APIError("openai", str(e)) from e

    logger.info(f"Fin de transcripción de contrato {contract_type}")
    return response
