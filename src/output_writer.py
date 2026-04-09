import json
import os
from datetime import datetime

from exceptions import OutputSaveError
from models import ContractChangeOutput


def save_output(output: ContractChangeOutput, contract_dir: str) -> str:
    """Guarda el output en output/<folder_name>_<fecha>_<hora>.json y retorna la ruta del archivo."""
    folder_name = os.path.basename(os.path.normpath(contract_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_name}_{timestamp}.json"

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, ensure_ascii=False, indent=2)
    except PermissionError as e:
        raise OutputSaveError(os.path.join(output_dir, filename), f"sin permisos de escritura: {e}")
    except OSError as e:
        raise OutputSaveError(os.path.join(output_dir, filename), str(e))

    return filepath
