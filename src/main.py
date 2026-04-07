import os
import sys
import glob

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langfuse import Langfuse

from agents.contextualization_agent import run_contextualization_agent
from agents.extraction_agent import run_extraction_agent
from image_parser import parse_contract_image
from logger import get_logger
from models import ContractChangeOutput, ContractState

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

    response = parse_contract_image(state["path_original_contract"], model=model)

    gen.update(output={"text": response.content})
    gen.end()

    logger.info("Contrato original parseado.")
    return {
        "text_extract_original_contract": response.content,
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

    response = parse_contract_image(state["path_amendment_contract"], model=model)

    gen.update(output={"text": response.content})
    gen.end()

    logger.info("Contrato modificado parseado.")
    return {
        "text_extract_amendment_contract": response.content,
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
            "original_text_preview": state["text_extract_original_contract"],
            "amendment_text_preview": state["text_extract_amendment_contract"],
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

    logger.info("Mapa contextual generado.")
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
        input={"contextual_map_preview": state["contextual_map"]},
    )

    response = run_extraction_agent(
        original_text=state["text_extract_original_contract"],
        amendment_text=state["text_extract_amendment_contract"],
        contextual_map=state["contextual_map"],
        model=model
    )

    result = response["parsed"]

    gen.update(output=result.model_dump())
    gen.end()

    logger.info("Extracción de cambios completada.")
    return {
        "final_output": result,
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

    logger.info("Pipeline completado.")
    return result["final_output"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py <carpeta_contrato>")
        print("Ejemplo: python main.py data/test_contracts/contrato_1")
        sys.exit(1)

    contract_dir = sys.argv[1]
    original_dir = os.path.join(contract_dir, "original")
    modified_dir = os.path.join(contract_dir, "modificado")

    originals = sorted(glob.glob(os.path.join(original_dir, "*")))
    amendments = sorted(glob.glob(os.path.join(modified_dir, "*")))

    if not originals:
        print(f"No se encontraron archivos en {original_dir}")
        sys.exit(1)
    if not amendments:
        print(f"No se encontraron archivos en {modified_dir}")
        sys.exit(1)

    output = main(originals, amendments)
    print("\n--- RESULTADO ---")
    print(output.model_dump_json(indent=2))
