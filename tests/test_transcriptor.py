"""
Golden case tests para el transcriptor.
Llaman a la API real de OpenAI — requieren OPENAI_API_KEY en el entorno.
"""

import difflib
from pathlib import Path

import pytest

from image_parser import parse_contract_image

DATA_DIR = Path(__file__).parent.parent / "data" / "test_contracts"
GOLDEN_DIR = Path(__file__).parent / "golden"
SIMILARITY_THRESHOLD = 0.98

@pytest.mark.skip(reason="golden de contrato_1 pendiente")
def test_transcriptor_contrato_1_original():
    """Golden Case: contrato_1 original debe tener ≥90% de similitud con el golden."""
    image_paths = sorted(str(p) for p in (DATA_DIR / "contrato_1" / "original").glob("*.png"))
    golden = (GOLDEN_DIR / "contrato_1" / "original" / "contrato_1_orignal.txt").read_text(encoding="utf-8").strip()

    result = parse_contract_image(image_paths, model="gpt-4o", contract_type="original")
    transcription = result.content.strip()

    similarity = difflib.SequenceMatcher(None, golden.lower(), transcription.lower()).ratio()
    assert similarity >= SIMILARITY_THRESHOLD, f"Similitud: {similarity:.2%} — esperado ≥{SIMILARITY_THRESHOLD:.0%}"


@pytest.mark.skip(reason="golden de contrato_1 pendiente")
def test_transcriptor_contrato_1_modificado():
    """Golden Case: contrato_1 modificado debe tener ≥90% de similitud con el golden."""
    image_paths = sorted(str(p) for p in (DATA_DIR / "contrato_1" / "modificado").glob("*.png"))
    golden = (GOLDEN_DIR / "contrato_1" / "modificado" / "contrato_1_modificado.txt").read_text(encoding="utf-8").strip()

    result = parse_contract_image(image_paths, model="gpt-4o", contract_type="modificado")
    transcription = result.content.strip()

    similarity = difflib.SequenceMatcher(None, golden.lower(), transcription.lower()).ratio()
    assert similarity >= SIMILARITY_THRESHOLD, f"Similitud: {similarity:.2%} — esperado ≥{SIMILARITY_THRESHOLD:.0%}"


def test_transcriptor_contrato_3_original():
    """Golden Case: contrato_3 original debe tener ≥90% de similitud con el golden."""
    image_paths = [str(DATA_DIR / "contrato_3" / "original" / "documento_2__original.jpg")]
    golden = (GOLDEN_DIR / "contrato_3" / "original" / "contrato_3_original.txt").read_text(encoding="utf-8").strip()

    result = parse_contract_image(image_paths, model="gpt-4o", contract_type="original")
    transcription = result.content.strip()

    similarity = difflib.SequenceMatcher(None, golden.lower(), transcription.lower()).ratio()
    assert similarity >= SIMILARITY_THRESHOLD, f"Similitud: {similarity:.2%} — esperado ≥{SIMILARITY_THRESHOLD:.0%}"


def test_transcriptor_contrato_3_modificado():
    """Golden Case: contrato_3 modificado debe tener ≥90% de similitud con el golden."""
    image_paths = [str(DATA_DIR / "contrato_3" / "modificado" / "documento_2__enmienda.jpg")]
    golden = (GOLDEN_DIR / "contrato_3" / "modificado" / "contrato_3_modificado.txt").read_text(encoding="utf-8").strip()

    result = parse_contract_image(image_paths, model="gpt-4o", contract_type="modificado")
    transcription = result.content.strip()

    similarity = difflib.SequenceMatcher(None, golden.lower(), transcription.lower()).ratio()
    assert similarity >= SIMILARITY_THRESHOLD, f"Similitud: {similarity:.2%} — esperado ≥{SIMILARITY_THRESHOLD:.0%}"
