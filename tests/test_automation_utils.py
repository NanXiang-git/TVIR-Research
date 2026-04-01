import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from src.core.automation_utils import (  # noqa: E402
    infer_quality_checks,
    normalize_query_result,
    normalize_topic_result,
    synthesize_query_description,
)


def test_normalize_topic_result_fills_missing_candidates():
    result = normalize_topic_result({}, domain="新能源", complexity="high")

    assert result["domain"] == "新能源"
    assert len(result["candidates"]) == 3
    assert result["selected_topic"]["topic"]


def test_normalize_query_result_enforces_required_shape():
    topic_result = normalize_topic_result({}, domain="environmental policy", complexity="medium")
    result = normalize_query_result(
        raw={"main_task": "Analyze the latest methane policy shifts"},
        domain="environmental policy",
        language="en",
        complexity="medium",
        topic_result=topic_result,
    )

    assert result["user_role"]
    assert len(result["sub_questions"]) == 4
    assert any(item["type"] == "image" for item in result["multimodal_requirements"])
    assert any(item["type"] == "chart" for item in result["multimodal_requirements"])


def test_infer_quality_checks_detects_complete_task():
    task = {
        "user_role": "Director of strategy at a clean-energy utility",
        "sub_questions": ["Q1", "Q2", "Q3"],
        "citations": [{"label": "paper", "url": "Nature Energy 2023 https://example.com"}],
        "multimodal_requirements": [
            {"type": "image", "source": "https://example.com/image.png"},
            {
                "type": "chart",
                "data_source": "IEA 2024 battery outlook",
                "reproducibility_note": "Rebuild from table 2 in the linked report.",
            },
        ],
    }

    checks = infer_quality_checks(task)

    assert checks == {
        "role_clear": True,
        "sub_questions_coherent": True,
        "timely_sources": True,
        "image_accessible": True,
        "chart_reproducible": True,
    }


def test_synthesize_query_description_includes_core_fields():
    task = {
        "user_role": "某三级甲等医院的AI医疗应用主管",
        "main_task": "聚焦AI医生在诊断领域的智能化进展",
        "sub_questions": [
            "系统梳理核心技术演进",
            "分析慢性病筛查与急诊分诊中的典型案例",
            "评估伦理隐私风险与监管要求",
        ],
        "multimodal_requirements": [
            {"type": "image", "description": "AI医生诊断流程图"},
            {"type": "chart", "description": "主流工具在效能维度上的雷达图"},
        ],
    }

    query = synthesize_query_description(task, language="zh", freshness_window="2023-2025")

    assert "某三级甲等医院的AI医疗应用主管" in query
    assert "2023-2025年" in query
    assert "系统梳理核心技术演进" in query
    assert "引用AI医生诊断流程图" in query
    assert "绘制主流工具在效能维度上的雷达图" in query
