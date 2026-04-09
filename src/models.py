import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field


class ContractChangeOutput(BaseModel):
    sections_changed: list[str] = Field(
        min_length=1,
        description="Nombres exactos de las secciones o cláusulas afectadas.",
        examples=[["Alcance del Servicio", "Cláusula 3", "Anexo A"]],
    )
    topics_touched: list[str] = Field(
        min_length=1,
        description="Temas jurídicos o contractuales involucrados en los cambios.",
        examples=[["Precio", "Plazo", "Responsabilidad"]],
    )
    summary_of_the_change: str = Field(
        min_length=1,
        description="Resumen breve de los cambios, estructurado en tres secciones. En caso de modificaciones mencionar el valor anterior y el nuevo con una flecha",
        examples=[
            "1. ADICIONES:\n   - Cláusula 5: Se incorpora obligación de seguro contra incendios.\n\n"
            "2. ELIMINACIONES:\n   - Sin cambios.\n\n"
            "3. MODIFICACIONES:\n   - Cláusula 3: Plazo de entrega de 30 días → 45 días.\n   - Cláusula 4 'Soporte': Medios de soporte técnico: email y teléfono → email, teléfono y chat."
        ],
    )


class ContractState(TypedDict):
    # Inputs
    path_original_contract: List[str]
    path_amendment_contract: List[str]

    # Paso 1 - Vision parsing
    text_extract_original_contract: str
    text_extract_amendment_contract: str

    # Paso 2 - Agent 1 output
    contextual_map: str

    # Paso 3 - Agent 2 output
    final_output: ContractChangeOutput

    # Tracking de pasos acumulados
    pipeline_steps: Annotated[List[dict], operator.add]
