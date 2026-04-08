from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable


AUTOMATION_AGENT_PROMPTS = {
    "agent-topic-generator": "topic_generation.json",
    "agent-query-builder": "query_construction.json",
    "agent-refiner": "refine_validation.json",
}


def is_automation_agent(agent_type: str) -> bool:
    return agent_type in AUTOMATION_AGENT_PROMPTS


@lru_cache(maxsize=None)
def _load_prompt_file(agent_type: str) -> Dict[str, Any]:
    if agent_type not in AUTOMATION_AGENT_PROMPTS:
        raise ValueError(f"Unsupported automation agent type: {agent_type}")

    prompt_dir = Path(__file__).resolve().parents[2] / "prompts"
    prompt_path = prompt_dir / AUTOMATION_AGENT_PROMPTS[agent_type]
    with prompt_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_language_block(prompt_data: Dict[str, Any], field: str, language: str) -> str:
    value = prompt_data.get(field, "")
    if isinstance(value, dict):
        return value.get(language) or value.get("zh") or next(iter(value.values()))
    return str(value)


def _render_prompt_sections(
    prompt_data: Dict[str, Any], language: str, section_names: Iterable[str]
) -> str:
    sections = []
    for section_name in section_names:
        text = _resolve_language_block(prompt_data, section_name, language).strip()
        if text:
            sections.append(text)
    return "\n\n".join(sections).strip()


def generate_automation_system_prompt(agent_type: str, language: str) -> str:
    prompt_data = _load_prompt_file(agent_type)
    system_text = _render_prompt_sections(
        prompt_data,
        language,
        (
            "system",
            "design_principles",
            "multimodal_taxonomy",
            "complexity_policy",
            "citation_policy",
            "tool_workflow",
            "few_shot",
        ),
    )
    output_format = _resolve_language_block(prompt_data, "output_format", language)
    return (
        f"{system_text}\n\n"
        "## Output Contract\n"
        f"{output_format}\n\n"
        "Return valid JSON only. Do not wrap the result in Markdown."
    ).strip()


def generate_automation_user_guidance(agent_type: str, language: str) -> str:
    prompt_data = _load_prompt_file(agent_type)
    return _render_prompt_sections(
        prompt_data,
        language,
        (
            "user_prompt",
            "tool_workflow",
            "complexity_policy",
            "citation_policy",
        ),
    )


def generate_automation_summary_prompt(
    task_description: str, agent_type: str, language: str
) -> str:
    prompt_data = _load_prompt_file(agent_type)
    output_format = _resolve_language_block(prompt_data, "output_format", language)

    if language == "zh":
        return (
            "这是对你的直接指令，不是工具返回结果。\n\n"
            "当前阶段即将结束，你不能继续调用工具。\n"
            "请基于已有上下文，输出最终 JSON 结果。\n\n"
            "原始任务如下：\n"
            f"{task_description}\n\n"
            "输出约束：\n"
            f"{output_format}\n\n"
            "只输出合法 JSON，不要添加 Markdown 代码块或额外解释。"
        )

    return (
        "This is a direct instruction, not a tool result.\n\n"
        "The current stage is ending and you must not call any more tools.\n"
        "Please use the existing context to produce the final JSON result.\n\n"
        "Original task:\n"
        f"{task_description}\n\n"
        "Output constraints:\n"
        f"{output_format}\n\n"
        "Return valid JSON only, with no Markdown fences or extra explanation."
    )
