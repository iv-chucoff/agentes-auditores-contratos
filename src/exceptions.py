"""Excepciones personalizadas para el sistema de auditoría de contratos."""


class ContractAuditError(Exception):
    """Excepción base del sistema. Captura cualquier error con un solo except."""
    pass


# ---------------------------------------------------------------------------
# Entrada / carpetas
# ---------------------------------------------------------------------------

class ContractDirError(ContractAuditError):
    """La carpeta del contrato no existe, no es un directorio, o le faltan subcarpetas."""
    pass


class EmptyContractError(ContractAuditError):
    """Una carpeta de contrato (original o modificado) no contiene archivos."""
    pass


class TooManyImagesError(ContractAuditError):
    """La carpeta contiene más imágenes que el límite permitido."""

    MAX_IMAGES = 20

    def __init__(self, contract_type: str, count: int) -> None:
        self.contract_type = contract_type
        self.count = count
        super().__init__(
            f"El contrato '{contract_type}' tiene {count} imágenes. "
            f"El límite máximo es {self.MAX_IMAGES}."
        )


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------

class InvalidImageFormatError(ContractAuditError):
    """El archivo no tiene un formato de imagen soportado por OpenAI Vision."""

    ALLOWED = {"png", "jpg", "jpeg", "gif", "webp"}

    def __init__(self, path: str, ext: str) -> None:
        self.path = path
        self.ext = ext
        super().__init__(
            f"Formato '{ext}' no soportado en '{path}'. "
            f"Formatos permitidos: {', '.join(sorted(self.ALLOWED))}"
        )


class ImageReadError(ContractAuditError):
    """No se pudo leer o abrir el archivo de imagen."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        super().__init__(f"No se pudo leer la imagen '{path}': {reason}")


# ---------------------------------------------------------------------------
# Transcripción
# ---------------------------------------------------------------------------

class TranscriptionTooShortError(ContractAuditError):
    """La transcripción obtenida es demasiado corta, lo que indica que el modelo
    no pudo extraer texto significativo de las imágenes."""

    MIN_CHARS = 500

    def __init__(self, contract_type: str, char_count: int) -> None:
        self.contract_type = contract_type
        self.char_count = char_count
        super().__init__(
            f"Transcripción del contrato '{contract_type}' demasiado corta "
            f"({char_count} caracteres). Mínimo requerido: {self.MIN_CHARS}. "
            "Verificá que las imágenes sean legibles."
        )


# ---------------------------------------------------------------------------
# Agentes / LLM
# ---------------------------------------------------------------------------

class MissingAPIKeyError(ContractAuditError):
    """Una API key requerida no está configurada en las variables de entorno."""

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        super().__init__(
            f"La variable de entorno '{key_name}' no está configurada. "
            "Verificá tu archivo .env."
        )


class APIError(ContractAuditError):
    """Error al llamar a una API externa (OpenAI, Langfuse): clave inválida,
    timeout, servicio caído, etc."""

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        super().__init__(f"[{service.upper()}] {message}")


class ParsedOutputError(ContractAuditError):
    """El agente de extracción no pudo parsear la respuesta del LLM al modelo
    estructurado ContractChangeOutput."""

    def __init__(self, parsing_error: str) -> None:
        super().__init__(
            f"No se pudo parsear el output del agente de extracción: {parsing_error}"
        )


# ---------------------------------------------------------------------------
# Guardado de output
# ---------------------------------------------------------------------------

class OutputSaveError(ContractAuditError):
    """No se pudo guardar el archivo JSON de resultado."""

    def __init__(self, filepath: str, reason: str) -> None:
        self.filepath = filepath
        super().__init__(f"No se pudo guardar el output en '{filepath}': {reason}")
