""" Tests unitarios para validations.py. """

import pytest

from exceptions import (
    ContractDirError,
    EmptyContractError,
    MissingAPIKeyError,
    TooManyImagesError,
    TranscriptionTooShortError,
)
from validations import (
    validate_api_key,
    validate_contract_dir,
    validate_contract_files,
    validate_transcription,
)


# ---------------------------------------------------------------------------
# validate_contract_dir
# ---------------------------------------------------------------------------

def test_validate_contract_dir_ok(tmp_path):
    """Carpeta válida con subcarpetas original y modificado."""
    (tmp_path / "original").mkdir()
    (tmp_path / "modificado").mkdir()

    original, modificado = validate_contract_dir(str(tmp_path))

    assert original == str(tmp_path / "original")
    assert modificado == str(tmp_path / "modificado")


def test_validate_contract_dir_no_existe():
    """Lanza error si la carpeta no existe."""
    with pytest.raises(ContractDirError):
        validate_contract_dir("/ruta/que/no/existe")


def test_validate_contract_dir_sin_subcarpeta_original(tmp_path):
    """Lanza error si falta la subcarpeta original."""
    (tmp_path / "modificado").mkdir()

    with pytest.raises(ContractDirError):
        validate_contract_dir(str(tmp_path))


def test_validate_contract_dir_sin_subcarpeta_modificado(tmp_path):
    """Lanza error si falta la subcarpeta modificado."""
    (tmp_path / "original").mkdir()

    with pytest.raises(ContractDirError):
        validate_contract_dir(str(tmp_path))


# ---------------------------------------------------------------------------
# validate_contract_files
# ---------------------------------------------------------------------------

def test_validate_contract_files_ok():
    """No lanza error con listas válidas."""
    validate_contract_files(["img1.png"], ["img2.png"], "original/", "modificado/")


def test_validate_contract_files_originales_vacios():
    """Lanza error si la lista de originales está vacía."""
    with pytest.raises(EmptyContractError):
        validate_contract_files([], ["img.png"], "original/", "modificado/")


def test_validate_contract_files_modificados_vacios():
    """Lanza error si la lista de modificados está vacía."""
    with pytest.raises(EmptyContractError):
        validate_contract_files(["img.png"], [], "original/", "modificado/")


def test_validate_contract_files_demasiadas_imagenes_originales():
    """Lanza error si hay más de 20 imágenes en originales."""
    imagenes = [f"img_{i}.png" for i in range(21)]
    with pytest.raises(TooManyImagesError):
        validate_contract_files(imagenes, ["img.png"], "original/", "modificado/")


def test_validate_contract_files_demasiadas_imagenes_modificadas():
    """Lanza error si hay más de 20 imágenes en modificados."""
    imagenes = [f"img_{i}.png" for i in range(21)]
    with pytest.raises(TooManyImagesError):
        validate_contract_files(["img.png"], imagenes, "original/", "modificado/")


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------

def test_validate_api_key_ok(monkeypatch):
    """No lanza error si la variable de entorno existe."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    validate_api_key("OPENAI_API_KEY")


def test_validate_api_key_faltante(monkeypatch):
    """Lanza error si la variable de entorno no está configurada."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        validate_api_key("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# validate_transcription
# ---------------------------------------------------------------------------

def test_validate_transcription_ok():
    """No lanza error si el texto supera el mínimo de caracteres."""
    texto_largo = "a" * 500
    validate_transcription("original", texto_largo)


def test_validate_transcription_muy_corta():
    """Lanza error si la transcripción tiene menos de 500 caracteres."""
    with pytest.raises(TranscriptionTooShortError):
        validate_transcription("original", "texto corto")
