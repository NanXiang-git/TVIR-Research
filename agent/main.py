# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import bootstrap_logger

load_dotenv()
logger = bootstrap_logger()


def build_cfg(llm_config: str, agent_config: str) -> DictConfig:
    agent_dir = Path(__file__).resolve().parent
    conf_dir = agent_dir / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"llm={llm_config}", f"agent={agent_config}"],
        )
    debug_dir = Path(str(cfg.debug_dir))
    if not debug_dir.is_absolute():
        cfg.debug_dir = str((agent_dir / debug_dir).resolve())
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured automation tasks with the TVIR multi-agent pipeline."
    )
    parser.add_argument("--domain", required=True, help="Task domain, for example energy transition")
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Output language",
    )
    parser.add_argument(
        "--complexity",
        choices=["low", "medium", "high"],
        required=True,
        help="Task complexity",
    )
    parser.add_argument(
        "--llm-config",
        default="default",
        help="Hydra llm config name, e.g. default / claude-4-5 / qwen-3",
    )
    parser.add_argument(
        "--agent-config",
        default="tvir_agent",
        help="Hydra agent config name",
    )
    return parser.parse_args()


async def amain(cfg: DictConfig, domain: str, language: str, complexity: str) -> dict:
    logger.info(OmegaConf.to_yaml(cfg))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_name = cfg.llm.model_name
    task_id = f"{llm_name}/automation_{timestamp}"
    task_input = {
        "domain": domain,
        "language": language,
        "complexity": complexity,
    }

    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg, custom_env={"TASK_ID": task_id})
    )

    result, _log_file_path = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_input=task_input,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )
    return result


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args.llm_config, args.agent_config)
    result = asyncio.run(amain(cfg, args.domain, args.language, args.complexity))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
