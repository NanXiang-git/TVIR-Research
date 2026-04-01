from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List


QUALITY_CHECK_FIELDS = (
    "role_clear",
    "sub_questions_coherent",
    "timely_sources",
    "image_accessible",
    "chart_reproducible",
)


def target_sub_question_count(complexity: str) -> int:
    return {"low": 3, "medium": 4, "high": 5}.get(complexity, 4)


def _ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_topic_result(raw: Any, domain: str, complexity: str) -> Dict[str, Any]:
    if isinstance(raw, list):
        raw = {"candidates": raw}
    if not isinstance(raw, dict):
        raw = {}

    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    normalized_candidates = []
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        topic = str(candidate.get("topic", "")).strip()
        if not topic:
            continue
        normalized_candidates.append(
            {
                "topic": topic,
                "rationale": str(candidate.get("rationale", "")).strip(),
                "relevance_score": candidate.get("relevance_score", 0.0),
                "source_links": _ensure_string_list(candidate.get("source_links")),
            }
        )

    if not normalized_candidates:
        normalized_candidates = [
            {
                "topic": f"{domain}领域近三年前沿趋势研究",
                "rationale": "Fallback topic generated because the topic phase output was incomplete.",
                "relevance_score": 0.5,
                "source_links": [],
            }
        ]

    while len(normalized_candidates) < 3:
        normalized_candidates.append(deepcopy(normalized_candidates[-1]))

    selected_topic = raw.get("selected_topic")
    if not isinstance(selected_topic, dict):
        selected_topic = normalized_candidates[0]

    selected_topic_value = str(
        selected_topic.get("topic", normalized_candidates[0]["topic"])
    ).strip() or normalized_candidates[0]["topic"]

    return {
        "domain": domain,
        "complexity": complexity,
        "candidates": normalized_candidates[:3],
        "selected_topic": {
            "topic": selected_topic_value,
            "rationale": str(
                selected_topic.get("rationale", normalized_candidates[0]["rationale"])
            ).strip(),
            "source_links": _ensure_string_list(
                selected_topic.get(
                    "source_links", normalized_candidates[0]["source_links"]
                )
            ),
            "freshness_window": str(
                selected_topic.get("freshness_window", "2021+")
            ).strip()
            or "2021+",
        },
    }


def normalize_query_result(
    raw: Any,
    domain: str,
    language: str,
    complexity: str,
    topic_result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    topic_text = topic_result["selected_topic"]["topic"]
    question_count = target_sub_question_count(complexity)

    user_role = str(raw.get("user_role", "")).strip()
    if not user_role:
        user_role = (
            f"{domain}领域机构负责人"
            if language == "zh"
            else f"Decision-maker in the {domain} domain"
        )

    main_task = str(raw.get("main_task", "")).strip()
    if not main_task:
        main_task = topic_text

    sub_questions = _ensure_string_list(raw.get("sub_questions"))[:question_count]
    while len(sub_questions) < question_count:
        index = len(sub_questions) + 1
        sub_questions.append(
            (
                f"补充分析与“{topic_text}”直接相关的关键问题 {index}"
                if language == "zh"
                else f"Add a directly related research sub-question for '{topic_text}' ({index})"
            )
        )

    multimodal_requirements = raw.get("multimodal_requirements", [])
    if not isinstance(multimodal_requirements, list):
        multimodal_requirements = []

    normalized_multimodal = []
    for item in multimodal_requirements:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        if item_type not in {"image", "chart"}:
            continue
        normalized_item = {
            "type": item_type,
            "description": str(item.get("description", "")).strip(),
        }
        for key in (
            "purpose",
            "source",
            "source_page",
            "verification",
            "data_source",
            "reproducibility_note",
            "query",
            "citation",
        ):
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                cleaned = _ensure_string_list(value)
                if cleaned:
                    normalized_item[key] = cleaned
            else:
                text = str(value).strip()
                if text:
                    normalized_item[key] = text
        normalized_multimodal.append(normalized_item)

    if not any(item["type"] == "image" for item in normalized_multimodal):
        normalized_multimodal.append(
            {
                "type": "image",
                "description": (
                    f"{topic_text}相关结构示意图"
                    if language == "zh"
                    else f"Illustrative architecture or scene image for {topic_text}"
                ),
                "verification": "",
                "source": "",
            }
        )

    if not any(item["type"] == "chart" for item in normalized_multimodal):
        normalized_multimodal.append(
            {
                "type": "chart",
                "description": (
                    f"{topic_text}关键指标对比图"
                    if language == "zh"
                    else f"Comparative chart of key metrics for {topic_text}"
                ),
                "data_source": "",
                "reproducibility_note": "",
            }
        )

    citations = raw.get("citations", [])
    if isinstance(citations, list):
        normalized_citations = []
        for item in citations:
            if isinstance(item, dict):
                url = str(item.get("url", "")).strip()
                if url:
                    normalized_citations.append(
                        {
                            "label": str(item.get("label", "reference")).strip()
                            or "reference",
                            "url": url,
                        }
                    )
            elif isinstance(item, str) and item.strip():
                normalized_citations.append({"label": "reference", "url": item.strip()})
    else:
        normalized_citations = []

    return {
        "user_role": user_role,
        "main_task": main_task,
        "sub_questions": sub_questions,
        "multimodal_requirements": normalized_multimodal,
        "citations": normalized_citations,
    }


def normalize_refine_review(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    quality_checks = raw.get("quality_checks", {})
    if not isinstance(quality_checks, dict):
        quality_checks = {}

    normalized_checks = {
        field: bool(quality_checks.get(field, False)) for field in QUALITY_CHECK_FIELDS
    }

    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = _ensure_string_list(issues)

    return {
        "mode": str(raw.get("mode", "inspect")).strip() or "inspect",
        "needs_repair": bool(raw.get("needs_repair", False)),
        "issues": [str(issue).strip() for issue in issues if str(issue).strip()],
        "quality_checks": normalized_checks,
    }


def normalize_refine_repair(raw: Any, fallback_query: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    repaired_query = raw.get("repaired_query")
    if not isinstance(repaired_query, dict):
        repaired_query = deepcopy(fallback_query)

    repair_notes = raw.get("repair_notes", [])
    if not isinstance(repair_notes, list):
        repair_notes = _ensure_string_list(repair_notes)

    return {
        "mode": str(raw.get("mode", "repair")).strip() or "repair",
        "repaired_query": repaired_query,
        "repair_notes": [str(note).strip() for note in repair_notes if str(note).strip()],
    }


def infer_quality_checks(task: Dict[str, Any]) -> Dict[str, bool]:
    role_clear = bool(str(task.get("user_role", "")).strip())
    sub_questions = _ensure_string_list(task.get("sub_questions"))
    sub_questions_coherent = len(sub_questions) >= 3
    citations = task.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    timely_sources = any(
        re.search(r"\b(202[1-9]|203\d)\b", " ".join(map(str, item.values())))
        for item in citations
        if isinstance(item, dict)
    )
    multimodal = task.get("multimodal_requirements", [])
    if not isinstance(multimodal, list):
        multimodal = []
    image_accessible = any(
        item.get("type") == "image" and bool(str(item.get("source", "")).strip())
        for item in multimodal
        if isinstance(item, dict)
    )
    chart_reproducible = any(
        item.get("type") == "chart"
        and bool(str(item.get("data_source", "")).strip())
        and bool(str(item.get("reproducibility_note", "")).strip())
        for item in multimodal
        if isinstance(item, dict)
    )

    return {
        "role_clear": role_clear,
        "sub_questions_coherent": sub_questions_coherent,
        "timely_sources": timely_sources,
        "image_accessible": image_accessible,
        "chart_reproducible": chart_reproducible,
    }


def synthesize_query_description(
    task: Dict[str, Any],
    language: str,
    freshness_window: str = "",
) -> str:
    user_role = str(task.get("user_role", "")).strip()
    main_task = str(task.get("main_task", "")).strip()
    sub_questions = _ensure_string_list(task.get("sub_questions"))
    multimodal_requirements = task.get("multimodal_requirements", [])
    if not isinstance(multimodal_requirements, list):
        multimodal_requirements = []

    if language == "zh":
        freshness_phrase = _format_freshness_phrase_zh(freshness_window)
        question_clause = "；".join(q.rstrip("？?。.") for q in sub_questions if q.strip())
        question_clause = f"{question_clause}。" if question_clause else ""
        multimodal_clause = _build_multimodal_clause_zh(multimodal_requirements)

        parts = [
            f"本人作为{user_role}，正在围绕“{main_task}”开展一项深度研究任务。",
            f"请聚焦{freshness_phrase}的最新技术、政策与产业进展，系统回答以下关键问题：{question_clause}"
            if question_clause
            else f"请聚焦{freshness_phrase}的最新技术、政策与产业进展，系统梳理该任务的关键事实、趋势与挑战。",
        ]
        if multimodal_clause:
            parts.append(f"研究过程中还需满足以下多模态要求：{multimodal_clause}。")
        parts.append("请基于可核实来源形成结构化分析，并给出可执行的判断、比较与建议。")
        return "".join(parts)

    freshness_phrase = _format_freshness_phrase_en(freshness_window)
    question_clause = "; ".join(
        q.rstrip("?.") + "?" for q in sub_questions if q.strip()
    )
    multimodal_clause = _build_multimodal_clause_en(multimodal_requirements)

    parts = [
        f"I am acting as {user_role} and need a deep research task focused on {main_task}. ",
        f"Please focus on the latest technology, policy, and market developments from {freshness_phrase}. ",
    ]
    if question_clause:
        parts.append(f"The task should systematically address the following key questions: {question_clause} ")
    if multimodal_clause:
        parts.append(f"The research must also satisfy these multimodal requirements: {multimodal_clause}. ")
    parts.append(
        "Ground the analysis in verifiable sources and end with actionable comparisons, judgments, and recommendations."
    )
    return "".join(parts).strip()


def _format_freshness_phrase_zh(freshness_window: str) -> str:
    text = str(freshness_window or "").strip()
    if not text:
        return "近三年"
    if re.search(r"20\d{2}\s*-\s*20\d{2}", text):
        return f"{text}年"
    if text.endswith("+"):
        return f"{text}以来"
    return text


def _format_freshness_phrase_en(freshness_window: str) -> str:
    text = str(freshness_window or "").strip()
    if not text:
        return "the past three years"
    if text.endswith("+"):
        return f"{text}"
    return text


def _build_multimodal_clause_zh(multimodal_requirements: List[Dict[str, Any]]) -> str:
    clauses = []
    for item in multimodal_requirements:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        description = str(item.get("description", "")).strip().rstrip("。")
        if not description:
            continue
        if item_type == "image":
            clauses.append(f"引用{description}")
        elif item_type == "chart":
            clauses.append(f"绘制{description}")
        else:
            clauses.append(description)
    return "，并".join(clauses)


def _build_multimodal_clause_en(multimodal_requirements: List[Dict[str, Any]]) -> str:
    clauses = []
    for item in multimodal_requirements:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        description = str(item.get("description", "")).strip().rstrip(".")
        if not description:
            continue
        if item_type == "image":
            clauses.append(f"include a referenced image for {description}")
        elif item_type == "chart":
            clauses.append(f"produce a chart for {description}")
        else:
            clauses.append(description)
    return "; ".join(clauses)
