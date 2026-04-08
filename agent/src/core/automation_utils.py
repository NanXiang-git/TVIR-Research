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


def target_analysis_point_count(complexity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(complexity, 2)


def _ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_citation_items(value: Any, default_label: str = "reference") -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, str) and item.strip():
            normalized.append({"label": default_label, "url": item.strip()})
            continue
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        if not url and not title:
            continue
        citation = {
            "label": str(item.get("label", default_label)).strip() or default_label,
        }
        if url:
            citation["url"] = url
        if title:
            citation["title"] = title
        for key in ("used_for", "key_finding", "summary", "entity"):
            text = str(item.get(key, "")).strip()
            if text:
                citation[key] = text
        normalized.append(citation)
    return normalized


def _default_analysis_points(title: str, complexity: str, language: str) -> List[str]:
    point_count = target_analysis_point_count(complexity)
    if language == "zh":
        defaults = [
            f"梳理“{title}”的关键事实、最新进展与核心驱动因素",
            f"比较“{title}”相关案例、指标差异与现实约束",
            f"评估“{title}”的风险边界、政策影响与后续行动建议",
        ]
    else:
        defaults = [
            f"Clarify the latest facts, developments, and drivers for '{title}'",
            f"Compare representative cases, metrics, and constraints related to '{title}'",
            f"Assess the risks, policy implications, and recommended next steps for '{title}'",
        ]
    return defaults[:point_count]


def _normalize_question_details(
    raw_details: Any,
    sub_questions: List[str],
    complexity: str,
    language: str,
) -> List[Dict[str, Any]]:
    if complexity == "low":
        return []

    point_count = target_analysis_point_count(complexity)
    normalized: List[Dict[str, Any]] = []
    details = raw_details if isinstance(raw_details, list) else []

    for index, title in enumerate(sub_questions):
        raw = details[index] if index < len(details) and isinstance(details[index], dict) else {}
        detail_title = str(raw.get("title", raw.get("question", title))).strip() or title
        analysis_points = _ensure_string_list(
            raw.get("analysis_points", raw.get("focus_points", raw.get("sub_questions", [])))
        )[:point_count]
        while len(analysis_points) < point_count:
            analysis_points = analysis_points + _default_analysis_points(
                detail_title, complexity, language
            )[len(analysis_points):point_count]

        detail: Dict[str, Any] = {
            "title": detail_title,
            "analysis_points": analysis_points,
            "citations": _normalize_citation_items(
                raw.get("citations", []), default_label=f"question_{index + 1}_reference"
            ),
        }
        objective = str(raw.get("objective", "")).strip()
        if objective:
            detail["objective"] = objective
        normalized.append(detail)

    return normalized


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
    if not sub_questions and isinstance(raw.get("question_details"), list):
        derived_titles = []
        for item in raw.get("question_details", []):
            if isinstance(item, dict):
                title = str(item.get("title", item.get("question", ""))).strip()
                if title:
                    derived_titles.append(title)
        sub_questions = derived_titles[:question_count]
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
        if item_type == "table":
            item_type = "chart"
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
    normalized_multimodal = _ensure_multimodal_coverage(
        multimodal_requirements=normalized_multimodal,
        sub_questions=sub_questions,
        topic_text=topic_text,
        language=language,
    )

    normalized_citations = _normalize_citation_items(raw.get("citations", []))
    question_details = _normalize_question_details(
        raw_details=raw.get("question_details", []),
        sub_questions=sub_questions,
        complexity=complexity,
        language=language,
    )

    return {
        "complexity": complexity,
        "user_role": user_role,
        "main_task": main_task,
        "sub_questions": sub_questions,
        "question_details": question_details,
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


def _collect_all_citations(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations = task.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    question_details = task.get("question_details", [])
    if isinstance(question_details, list):
        for item in question_details:
            if isinstance(item, dict) and isinstance(item.get("citations"), list):
                citations.extend(
                    citation
                    for citation in item["citations"]
                    if isinstance(citation, dict)
                )
    return citations


def infer_quality_checks(task: Dict[str, Any]) -> Dict[str, bool]:
    role_clear = bool(str(task.get("user_role", "")).strip())
    sub_questions = _ensure_string_list(task.get("sub_questions"))
    sub_questions_coherent = len(sub_questions) >= 3
    citations = _collect_all_citations(task)
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
    complexity = str(task.get("complexity", "medium")).strip() or "medium"
    sub_questions = _ensure_string_list(task.get("sub_questions"))
    question_details = task.get("question_details", [])
    if not isinstance(question_details, list):
        question_details = []
    multimodal_requirements = task.get("multimodal_requirements", [])
    if not isinstance(multimodal_requirements, list):
        multimodal_requirements = []

    if language == "zh":
        freshness_phrase = _format_freshness_phrase_zh(freshness_window)
        topic_phrase = _summarize_main_task_zh(main_task)
        if complexity == "low":
            dimension_clause = _build_compact_dimension_blocks_zh(
                sub_questions=sub_questions,
                multimodal_requirements=multimodal_requirements,
            )
            intro = (
                f"请聚焦{freshness_phrase}的最新数据、政策文本与权威文献，围绕{topic_phrase}展开研究，重点回答以下问题：{dimension_clause}。"
                if dimension_clause
                else f"请聚焦{freshness_phrase}的最新数据、政策文本与权威文献，围绕{topic_phrase}展开研究。"
            )
            closing = "请基于可核实来源形成简洁但专业的分析，并给出明确结论与建议。"
        else:
            dimension_clause = _build_detailed_dimension_blocks_zh(
                sub_questions=sub_questions,
                question_details=question_details,
                multimodal_requirements=multimodal_requirements,
                topic_phrase=topic_phrase,
                complexity=complexity,
            )
            intro = (
                f"请聚焦{freshness_phrase}的最新数据、政策文本与权威文献，围绕{topic_phrase}展开系统性研究，结合多源证据进行批判性分析，重点覆盖以下维度：{dimension_clause}。"
                if dimension_clause
                else f"请聚焦{freshness_phrase}的最新数据、政策文本与权威文献，围绕{topic_phrase}系统梳理关键事实、趋势、风险与实施路径。"
            )
            closing = (
                "请基于可核实来源形成结构化分析，明确关键证据、局限性、优先级排序与后续行动建议。"
                if complexity == "medium"
                else "请基于可核实来源形成结构化分析，分维度标注关键证据与对应引用，说明局限性、优先级排序、实施路径与后续行动建议。"
            )

        parts = [
            f"本人作为{user_role}，正在撰写一份深度研究报告。",
            intro,
        ]
        parts.append(closing)
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
            clauses.append(_render_chart_requirement_zh(description))
        else:
            clauses.append(description)
    return "，并".join(clauses)


def _ensure_multimodal_coverage(
    multimodal_requirements: List[Dict[str, Any]],
    sub_questions: List[str],
    topic_text: str,
    language: str,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    has_image = False
    for item in multimodal_requirements:
        if not isinstance(item, dict):
            continue
        cloned = deepcopy(item)
        description = str(cloned.get("description", "")).strip()
        if language == "zh" and cloned.get("type") == "chart":
            description = _normalize_chart_description_zh(description)
        cloned["description"] = description
        normalized.append(cloned)
        if cloned.get("type") == "image":
            has_image = True

    if not sub_questions:
        return normalized

    if language == "zh":
        chart_count = sum(1 for item in normalized if item.get("type") == "chart")
        for question in sub_questions[chart_count:]:
            normalized.append(
                _generate_default_multimodal_requirement(
                    question=question,
                    topic_text=topic_text,
                    language=language,
                    prefer_image=False,
                )
            )
        if not has_image:
            normalized.append(
                _generate_default_multimodal_requirement(
                    question=sub_questions[0],
                    topic_text=topic_text,
                    language=language,
                    prefer_image=True,
                )
            )
        return normalized

    target_count = max(2, len(sub_questions))
    question_index = 0
    while len(normalized) < target_count:
        default_item = _generate_default_multimodal_requirement(
            question=sub_questions[question_index % len(sub_questions)],
            topic_text=topic_text,
            language=language,
            prefer_image=not has_image,
        )
        normalized.append(default_item)
        if default_item.get("type") == "image":
            has_image = True
        question_index += 1

    return normalized


def _generate_default_multimodal_requirement(
    question: str,
    topic_text: str,
    language: str,
    prefer_image: bool = False,
) -> Dict[str, Any]:
    if language != "zh":
        if prefer_image:
            return {
                "type": "image",
                "description": f"Illustrative diagram for {topic_text}",
                "verification": "",
                "source": "",
            }
        return {
            "type": "chart",
            "description": f"Comparative chart of key metrics for {topic_text}",
            "data_source": "",
            "reproducibility_note": "",
        }

    if prefer_image or _question_prefers_image_zh(question):
        return {
            "type": "image",
            "description": _infer_image_description_zh(question, topic_text),
            "verification": "",
            "source": "",
        }
    return {
        "type": "chart",
        "description": _infer_chart_description_from_question_zh(question, topic_text),
        "data_source": "",
        "reproducibility_note": "",
    }


def _question_prefers_image_zh(question: str) -> bool:
    text = str(question or "")
    return any(
        keyword in text
        for keyword in ("框架", "架构", "机制", "流程", "路径", "系统", "技术演进", "技术进展")
    )


def _infer_image_description_zh(question: str, topic_text: str) -> str:
    text = str(question or "").strip().rstrip("？?。.；;")
    if any(keyword in text for keyword in ("流程", "路径")):
        return f"{text}流程图"
    if any(keyword in text for keyword in ("框架", "架构", "机制")):
        return f"{text}架构示意图"
    if any(keyword in text for keyword in ("技术演进", "演进")):
        return f"{topic_text}技术演进示意图"
    return f"{topic_text}关键机制示意图"


def _infer_chart_description_from_question_zh(question: str, topic_text: str) -> str:
    subject = _summarize_question_subject_zh(question) or topic_text
    chart_type = _infer_chart_type_zh(subject)
    if chart_type == "表格":
        return f"{subject}对比表格"
    return f"{subject}{chart_type}"


def _normalize_chart_description_zh(description: str) -> str:
    text = str(description or "").strip().rstrip("。")
    if not text:
        return "关键指标对比柱状图"
    if any(
        chart_type in text
        for chart_type in ("柱状图", "折线图", "雷达图", "饼图", "散点图", "热力图", "流程图", "桑基图", "箱线图", "表格")
    ):
        return text
    return _infer_chart_description_from_question_zh(text, text)


def _build_compact_dimension_blocks_zh(
    sub_questions: List[str],
    multimodal_requirements: List[Dict[str, Any]],
) -> str:
    chart_visuals = _assign_chart_visuals_to_dimensions_zh(
        sub_questions=sub_questions,
        multimodal_requirements=multimodal_requirements,
        topic_phrase="",
    )
    blocks = []
    for index, question in enumerate(sub_questions, start=1):
        text = str(question).strip().rstrip("。")
        if not text:
            continue
        chart_visual = chart_visuals[index - 1] if index - 1 < len(chart_visuals) else {}
        chart_clause = _render_visual_for_dimension_zh(chart_visual)
        if chart_clause:
            blocks.append(f"（{index}）{text}；{chart_clause}")
        else:
            blocks.append(f"（{index}）{text}")
    return " ".join(blocks)


def _build_detailed_dimension_blocks_zh(
    sub_questions: List[str],
    question_details: List[Dict[str, Any]],
    multimodal_requirements: List[Dict[str, Any]],
    topic_phrase: str,
    complexity: str,
) -> str:
    chart_visuals = _assign_chart_visuals_to_dimensions_zh(
        sub_questions=sub_questions,
        multimodal_requirements=multimodal_requirements,
        topic_phrase=topic_phrase,
    )
    image_assignments = _assign_images_to_dimensions_zh(
        sub_questions=sub_questions,
        multimodal_requirements=multimodal_requirements,
    )

    blocks = []
    for index, question in enumerate(sub_questions, start=1):
        detail = question_details[index - 1] if index - 1 < len(question_details) and isinstance(question_details[index - 1], dict) else {}
        detail_text = _expand_dimension_detail_zh(question, topic_phrase)
        analysis_points = []
        if isinstance(detail, dict):
            analysis_points = _ensure_string_list(detail.get("analysis_points", []))
        analysis_points = analysis_points[: target_analysis_point_count(complexity)]
        analysis_clause = _render_analysis_points_zh(analysis_points, complexity)

        visual_clauses = []
        chart_visual = chart_visuals[index - 1] if index - 1 < len(chart_visuals) else {}
        chart_clause = _render_visual_for_dimension_zh(chart_visual)
        if chart_clause:
            visual_clauses.append(chart_clause)
        image_visual = image_assignments.get(index - 1)
        image_clause = _render_visual_for_dimension_zh(image_visual)
        if image_clause:
            visual_clauses.append(image_clause)

        citation_clause = ""
        if complexity == "high" and isinstance(detail, dict):
            citations = _normalize_citation_items(detail.get("citations", []))
            if citations:
                citation_titles = [
                    str(item.get("title", "")).strip()
                    for item in citations
                    if isinstance(item, dict) and str(item.get("title", "")).strip()
                ]
                if citation_titles:
                    citation_clause = f"；并在该部分优先标注与解读《{citation_titles[0]}》等对应来源"

        block = f"（{index}）{detail_text}"
        if analysis_clause:
            block = f"{block}；{analysis_clause}"
        if visual_clauses:
            block = f"{block}；" + "；".join(visual_clauses)
        if citation_clause:
            block = f"{block}{citation_clause}"
        blocks.append(block)
    return " ".join(blocks)


def _render_analysis_points_zh(points: List[str], complexity: str) -> str:
    cleaned = [point.strip().rstrip("。") for point in points if point.strip()]
    if not cleaned:
        return ""
    if complexity == "medium":
        return "重点包括" + "、".join(cleaned)
    return "至少覆盖" + "；".join(cleaned)


def _build_dimension_blocks_zh(
    sub_questions: List[str],
    multimodal_requirements: List[Dict[str, Any]],
    topic_phrase: str,
) -> str:
    chart_visuals = _assign_chart_visuals_to_dimensions_zh(
        sub_questions=sub_questions,
        multimodal_requirements=multimodal_requirements,
        topic_phrase=topic_phrase,
    )
    image_assignments = _assign_images_to_dimensions_zh(
        sub_questions=sub_questions,
        multimodal_requirements=multimodal_requirements,
    )
    blocks = []
    for index, question in enumerate(sub_questions, start=1):
        detail = _expand_dimension_detail_zh(question, topic_phrase)
        visual_clauses = []
        chart_visual = chart_visuals[index - 1] if index - 1 < len(chart_visuals) else {}
        chart_clause = _render_visual_for_dimension_zh(chart_visual)
        if chart_clause:
            visual_clauses.append(chart_clause)
        image_visual = image_assignments.get(index - 1)
        image_clause = _render_visual_for_dimension_zh(image_visual)
        if image_clause:
            visual_clauses.append(image_clause)
        block = f"（{index}）{detail}"
        if visual_clauses:
            block = f"{block}；" + "；".join(visual_clauses)
        blocks.append(block)
    return " ".join(blocks)


def _assign_chart_visuals_to_dimensions_zh(
    sub_questions: List[str],
    multimodal_requirements: List[Dict[str, Any]],
    topic_phrase: str,
) -> List[Dict[str, Any]]:
    charts = [
        deepcopy(item)
        for item in multimodal_requirements
        if isinstance(item, dict) and item.get("type") == "chart"
    ]

    assigned: List[Dict[str, Any]] = []
    for question in sub_questions:
        if charts:
            assigned.append(charts.pop(0))
        else:
            assigned.append(
                _generate_default_multimodal_requirement(
                    question=question,
                    topic_text=topic_phrase,
                    language="zh",
                    prefer_image=False,
                )
            )
    return assigned


def _assign_images_to_dimensions_zh(
    sub_questions: List[str],
    multimodal_requirements: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    images = [
        deepcopy(item)
        for item in multimodal_requirements
        if isinstance(item, dict) and item.get("type") == "image"
    ]
    if not images:
        return {}

    assignments: Dict[int, Dict[str, Any]] = {}
    preferred_indices = [
        index
        for index, question in enumerate(sub_questions)
        if _question_prefers_image_zh(question)
    ]
    fallback_indices = [index for index in range(len(sub_questions)) if index not in preferred_indices]

    for index in preferred_indices + fallback_indices:
        if not images:
            break
        assignments[index] = images.pop(0)

    return assignments


def _expand_dimension_detail_zh(question: str, topic_phrase: str) -> str:
    title = str(question or "").strip().rstrip("？?。.；;")
    if not title:
        return f"{topic_phrase}关键议题：梳理核心事实、主要变化与现实约束，比较主要方案差异，并提出可执行判断"

    if any(keyword in title for keyword in ("未来发展趋势", "发展趋势", "趋势", "展望")):
        return f"{title}：总结当前应用基础与最新进展，分析未来3-5年的演进方向、关键不确定性与规模化前提，并判断潜在机会与约束"
    if any(keyword in title for keyword in ("框架", "架构", "机制", "技术演进", "技术进展", "底层技术", "模型", "数据分析")):
        return f"{title}：系统梳理核心技术路线、关键模型与数据流程，分析近三年升级方向、应用边界与部署门槛，并判断后续演进重点"
    if any(keyword in title for keyword in ("应用", "案例", "场景", "实践", "落地")):
        return f"{title}：总结典型应用场景与代表性案例，比较落地成效、患者获益、成本投入与可复制性，并识别推广障碍与成功条件"
    if any(keyword in title for keyword in ("性能", "指标", "效率", "准确率", "召回率", "对比", "成本", "效果", "满意度", "活跃度")):
        return f"{title}：对比关键性能指标、成本效率与实施效果，解释方案差异来源、适用边界与数据局限，并评估规模化可行性"
    if any(keyword in title for keyword in ("国际", "跨境", "全球", "海外", "竞争")):
        return f"{title}：比较主要国家或地区的推进路径、制度设计与竞争格局，评估对本地战略的机遇、挑战与可借鉴经验"
    if any(keyword in title for keyword in ("风险", "伦理", "隐私", "监管", "治理", "合规")):
        return f"{title}：识别主要风险、监管约束与治理缺口，评估现有政策工具的有效性、局限性与执行成本，并提出改进路径"
    if any(keyword in title for keyword in ("未来", "预测", "路径")):
        return f"{title}：预测未来3-5年的演进方向、关键不确定性与情景分化，比较不同路径成本收益，并提出可执行建议"
    return f"{title}：明确关键驱动因素、最新进展、现实约束与实施条件，比较主要方案或案例差异，并基于权威证据提出判断"


def _render_visual_for_dimension_zh(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    item_type = str(item.get("type", "")).strip().lower()
    description = str(item.get("description", "")).strip().rstrip("。")
    if not description:
        return ""
    if item_type == "image":
        return _render_image_requirement_zh(description)
    if item_type == "chart":
        return _render_chart_requirement_zh(description)
    return description


def _render_image_requirement_zh(description: str) -> str:
    subject = description
    if any(keyword in description for keyword in ("示意图", "架构图", "流程图")):
        return f"引用{description}，直观呈现{_strip_image_suffix_zh(subject)}"
    return f"引用{description}作为辅助说明"


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


def _summarize_main_task_zh(main_task: str) -> str:
    text = str(main_task or "").strip().rstrip("。")
    if not text:
        return "该研究主题"

    prefix_patterns = (
        r"^(设计和实施|设计与实施|设计|实施|分析|研究|评估|构建|搭建|探索|梳理|监测|预测|优化|制定|开发|推进|聚焦|围绕)",
        r"^(系统梳理|重点研究|深入研究)",
    )
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text).strip()

    text = re.sub(r"^并", "", text).strip()
    text = re.sub(r"^(实施|设计与实施|设计和实施)", "", text).strip()
    text = re.sub(r"[，,]\s*以(提高|提升|支持|促进|实现).*$", "", text).strip()
    text = re.sub(r"[，,]\s*并.*$", "", text).strip()
    text = re.sub(r"^(对|针对)", "", text).strip()
    return text or "该研究主题"


def _build_dimension_clause_zh(sub_questions: List[str]) -> str:
    clauses = []
    for index, question in enumerate(sub_questions, start=1):
        cleaned = str(question).strip().rstrip("？?。.；;")
        if cleaned:
            clauses.append(f"（{index}）{cleaned}")
    return "；".join(clauses) + "。" if clauses else ""


def _render_chart_requirement_zh(description: str) -> str:
    chart_type = _infer_chart_type_zh(description)
    subject = _strip_chart_type_suffix_zh(description)
    if chart_type == "表格":
        return f"以表格形式对比{subject}"
    if chart_type == "流程图":
        return f"绘制一张流程图，说明{subject}"
    return f"绘制一张{chart_type}，展示{subject}"


def _infer_chart_type_zh(description: str) -> str:
    text = str(description or "").strip()
    explicit_types = (
        "柱状图",
        "折线图",
        "雷达图",
        "饼图",
        "散点图",
        "热力图",
        "流程图",
        "桑基图",
        "箱线图",
        "表格",
    )
    for chart_type in explicit_types:
        if chart_type in text:
            return chart_type

    if any(keyword in text for keyword in ("效能维度", "能力维度", "多维", "综合表现")):
        return "雷达图"
    if any(keyword in text for keyword in ("趋势", "变化", "演进", "增长", "下降", "时间序列")):
        return "折线图"
    if any(keyword in text for keyword in ("对比", "比较", "差异", "统计数据", "排名")):
        return "柱状图"
    if any(keyword in text for keyword in ("占比", "比例", "构成", "份额")):
        return "饼图"
    if any(keyword in text for keyword in ("路径", "流程", "机制")):
        return "流程图"
    if any(keyword in text for keyword in ("指标", "矩阵", "清单")):
        return "表格"
    return "柱状图"


def _strip_chart_type_suffix_zh(description: str) -> str:
    text = str(description or "").strip().rstrip("。")
    text = re.sub(
        r"(的)?(柱状图|折线图|雷达图|饼图|散点图|热力图|流程图|桑基图|箱线图|表格)$",
        "",
        text,
    ).strip()
    if text.startswith("展示"):
        text = text[2:].strip()
    if text.startswith("显示"):
        text = text[2:].strip()
    if text.startswith("呈现"):
        text = text[2:].strip()
    return text or description.strip().rstrip("。")


def _summarize_question_subject_zh(question: str) -> str:
    text = str(question or "").strip().rstrip("？?。.；;")
    text = re.sub(r"^(系统梳理|重点分析|分析|评估|预测|比较|总结|识别|研究|说明|探讨|明确)", "", text).strip()
    text = re.sub(r"^(如何|哪些|什么|是否|为何)", "", text).strip()
    return text or str(question or "").strip().rstrip("？?。.；;")


def _strip_image_suffix_zh(description: str) -> str:
    text = str(description or "").strip().rstrip("。")
    text = re.sub(r"(示意图|架构图|流程图)$", "", text).strip()
    return text or description.strip().rstrip("。")
