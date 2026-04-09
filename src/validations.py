"""Validaciones de entrada para el pipeline de auditoría de contratos."""

import os

from exceptions import (
    ContractDirError,
    EmptyContractError,
    MissingAPIKeyError,
    TooManyImagesError,
    TranscriptionTooShortError,
)


def validate_contract_dir(contract_dir: str) -> tuple[str, str]:
    """Valida que la carpeta del contrato y sus subcarpetas existan.
    Retorna (original_dir, modified_dir)."""
    if not os.path.exists(contract_dir):
        raise ContractDirError(f"La carpeta '{contract_dir}' no existe.")
    if not os.path.isdir(contract_dir):
        raise ContractDirError(f"'{contract_dir}' no es una carpeta.")

    original_dir = os.path.join(contract_dir, "original")
    modified_dir = os.path.join(contract_dir, "modificado")

    if not os.path.isdir(original_dir):
        raise ContractDirError(f"No se encontró la subcarpeta 'original' en '{contract_dir}'.")
    if not os.path.isdir(modified_dir):
        raise ContractDirError(f"No se encontró la subcarpeta 'modificado' en '{contract_dir}'.")

    return original_dir, modified_dir


def validate_contract_files(originals: list[str], amendments: list[str], original_dir: str, modified_dir: str) -> None:
    """Valida que las carpetas no estén vacías y no superen el límite de imágenes."""
    if not originals:
        raise EmptyContractError(f"No se encontraron archivos en '{original_dir}'.")
    if not amendments:
        raise EmptyContractError(f"No se encontraron archivos en '{modified_dir}'.")

    if len(originals) > TooManyImagesError.MAX_IMAGES:
        raise TooManyImagesError("original", len(originals))
    if len(amendments) > TooManyImagesError.MAX_IMAGES:
        raise TooManyImagesError("modificado", len(amendments))


def validate_api_key(*key_names: str) -> None:
    """Verifica que las variables de entorno requeridas estén configuradas."""
    for key_name in key_names:
        if not os.getenv(key_name):
            raise MissingAPIKeyError(key_name)


def validate_transcription(contract_type: str, text: str) -> None:
    """Valida que la transcripción tenga al menos el mínimo de caracteres requerido."""
    if len(text) < TranscriptionTooShortError.MIN_CHARS:
        raise TranscriptionTooShortError(contract_type, len(text))
