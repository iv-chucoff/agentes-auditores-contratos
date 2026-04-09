import os
import sys
import glob

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langfuse import Langfuse

from agents.contextualization_agent import run_contextualization_agent
from agents.extraction_agent import run_extraction_agent
from exceptions import ContractAuditError
from validations import (
    validate_api_key,
    validate_contract_dir,
    validate_contract_files,
    validate_transcription,
)
from image_parser import parse_contract_image
from logger import get_logger
from models import ContractChangeOutput, ContractState
from output_writer import save_output

load_dotenv()
logger = get_logger("contract-analysis")

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_original_node(state: ContractState, config: RunnableConfig) -> dict:
    langfuse: Langfuse = config["configurable"]["langfuse"]
    trace_id: str = config["configurable"]["trace_id"]
    model = "gpt-4o"

    gen = langfuse.start_observation(
        name="parse_original_contract",
        as_type="generation",
        trace_context={"trace_id": trace_id},
        model=model,
        input={"paths": state["path_original_contract"]},
    )

    response = parse_contract_image(state["path_original_contract"], model=model, contract_type = "original")

    contract_text = response.content
    validate_transcription("original", contract_text)

    gen.update(output={"text_extract_original_contract": contract_text})
    gen.end()

    return {
        "text_extract_original_contract": contract_text,
        "pipeline_steps": ["parse_original_contract"],
    }


def parse_amendment_node(state: ContractState, config: RunnableConfig) -> dict:
    langfuse: Langfuse = config["configurable"]["langfuse"]
    trace_id: str = config["configurable"]["trace_id"]
    model = "gpt-4o"

    gen = langfuse.start_observation(
        name="parse_amendment_contract",
        as_type="generation",
        trace_context={"trace_id": trace_id},
        model=model,
        input={"paths": state["path_amendment_contract"]},
    )

    response = parse_contract_image(state["path_amendment_contract"], model=model, contract_type = "modificado")

    contract_text = response.content
    validate_transcription("modificado", contract_text)

    gen.update(output={"text_extract_amendment_contract": contract_text})
    gen.end()

    return {
        "text_extract_amendment_contract": contract_text,
        "pipeline_steps": ["parse_amendment_contract"],
    }


def contextualization_node(state: ContractState, config: RunnableConfig) -> dict:
    langfuse: Langfuse = config["configurable"]["langfuse"]
    trace_id: str = config["configurable"]["trace_id"]
    model = "gpt-4o-mini"

    gen = langfuse.start_observation(
        name="contextualization_agent",
        as_type="generation",
        trace_context={"trace_id": trace_id},
        model=model,
        input={
            "text_extract_original_contract": state["text_extract_original_contract"],
            "text_extract_amendment_contract": state["text_extract_amendment_contract"],
        },
    )

    response = run_contextualization_agent(
        original_text=state["text_extract_original_contract"],
        amendment_text=state["text_extract_amendment_contract"],
        model=model,
    )

    contextual_map = response.content

    gen.update(output={"contextual_map": contextual_map})
    gen.end()

    return {
        "contextual_map": contextual_map,
        "pipeline_steps": ["contextualization_agent"],
    }


def extraction_node(state: ContractState, config: RunnableConfig) -> dict:
    langfuse: Langfuse = config["configurable"]["langfuse"]
    trace_id: str = config["configurable"]["trace_id"]
    model = "gpt-4o-mini"

    gen = langfuse.start_observation(
        name="extraction_agent",
        as_type="generation",
        trace_context={"trace_id": trace_id},
        model=model,
        input={"contextual_map": state["contextual_map"],
               "text_extract_original_contract": state["text_extract_original_contract"],
               "text_extract_amendment_contract": state["text_extract_amendment_contract"],
               },
    )

    response = run_extraction_agent(
        original_text=state["text_extract_original_contract"],
        amendment_text=state["text_extract_amendment_contract"],
        contextual_map=state["contextual_map"],
        model=model
    )

    gen.update(output=response.model_dump())
    gen.end()

    return {
        "final_output": response,
        "pipeline_steps": ["extraction_agent"],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ContractState)

    graph.add_node("parse_original", parse_original_node)
    graph.add_node("parse_amendment", parse_amendment_node)
    graph.add_node("contextualization", contextualization_node)
    graph.add_node("extraction", extraction_node)

    graph.add_edge(START, "parse_original")
    graph.add_edge("parse_original", "parse_amendment")
    graph.add_edge("parse_amendment", "contextualization")
    graph.add_edge("contextualization", "extraction")
    graph.add_edge("extraction", END)

    return graph.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(original_paths: list[str], amendment_paths: list[str]) -> ContractChangeOutput:
    langfuse = Langfuse()
    trace_id = langfuse.create_trace_id()

    root = langfuse.start_observation(
        name="contract-analysis",
        as_type="span",
        trace_context={"trace_id": trace_id},
        input={
            "original_paths": original_paths,
            "amendment_paths": amendment_paths,
        },
    )

    result = graph.invoke(
        input={
            "path_original_contract": original_paths,
            "path_amendment_contract": amendment_paths,
            "pipeline_steps": [],
        },
        config={
            "configurable": {
                "langfuse": langfuse,
                "trace_id": trace_id,
            }
        },
    )

    root.update(
        output=result["final_output"].model_dump(),
        metadata={"pipeline_steps": result["pipeline_steps"]},
    )
    root.end()
    langfuse.flush()

    return result["final_output"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py <carpeta_contrato>")
        print("Ejemplo: python main.py data/test_contracts/contrato_1")
        sys.exit(1)

    try:
        validate_api_key(
            "OPENAI_API_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_BASE_URL",
        )

        contract_dir = sys.argv[1]

        #Valido que exista el contrato original y el modificado y devuelvo sus rutas
        original_dir, modified_dir = validate_contract_dir(contract_dir)

        #Ordeno los paths en caso de que un contrato tenga mas de una hoja y obtengo cada archivo que esta en la carpeta
        originals = sorted(glob.glob(os.path.join(original_dir, "*")))
        amendments = sorted(glob.glob(os.path.join(modified_dir, "*")))

        #Valido que las carpetas no esten vacias y que el contrato no supere las 20 hojas
        validate_contract_files(originals, amendments, original_dir, modified_dir)

        output = main(originals, amendments)

        filepath = save_output(output, contract_dir)
        logger.info(f"Output guardado en: {filepath}")

        logger.info("Pipeline completado exitosamente.")

    except ContractAuditError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)
