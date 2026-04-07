import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel


class ContractChangeOutput(BaseModel):
    sections_changed: list[str]
    topics_touched: list[str]
    summary_of_the_change: str


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
