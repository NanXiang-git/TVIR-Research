# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

# Import from the new modular structure
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from src.logging.task_logger import bootstrap_logger

# Configure logger and get the configured instance
logger = bootstrap_logger()


def load_query_from_json(
    query_id: str, json_path: str = "../benchmark/data/query.json"
) -> dict:
    """Load query information from JSON file by query_id.

    Args:
        query_id: The ID of the query to load (e.g., "002001")
        json_path: Path to the query JSON file

    Returns:
        Dictionary containing query information

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If query_id not found in JSON
    """
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"Query JSON file not found: {json_path}")

    with open(json_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Find the query with matching ID
    for query_item in queries:
        if query_item.get("id") == query_id:
            logger.info(
                f"Loaded query {query_id}: {query_item.get('query', '')[:100]}..."
            )
            return query_item

    raise ValueError(f"Query ID '{query_id}' not found in {json_path}")


async def amain(cfg: DictConfig) -> None:
    """Asynchronous main function."""

    logger.info(OmegaConf.to_yaml(cfg))

    query_id = str(os.environ.get("QUERY_ID"))
    llm_name = cfg.llm.model_name

    if query_id and llm_name:
        # Load query from JSON file
        try:
            query_data = load_query_from_json(query_id)
            task_description = query_data.get("query", "")
            task_id = f"{llm_name}/{query_id}"

            logger.info(f"Running task for query ID: {query_id}")
            logger.info(f"Task description: {task_description[:200]}...")

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading query: {e}")
            return
    else:
        # Fallback to default task (for backward compatibility)
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_description = ""

    task_file_name = ""

    # Create pipeline components using the factory function
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg, custom_env={"TASK_ID": task_id})
    )

    # Execute task using the pipeline
    final_summary, final_boxed_answer, log_file_path = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    main()
