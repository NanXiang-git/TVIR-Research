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
    assert len(result["multimodal_requirements"]) >= 4


def test_normalize_query_result_adds_explicit_chart_types_for_dimensions():
    topic_result = normalize_topic_result({}, domain="医疗健康", complexity="low")
    result = normalize_query_result(
        raw={
            "main_task": "分析AI医生在诊断场景中的应用演进",
            "sub_questions": [
                "系统梳理核心技术演进",
                "分析典型应用案例与落地场景",
                "评估性能指标对比与监管风险",
            ],
            "multimodal_requirements": [
                {"type": "image", "description": "AI医生诊断流程图"},
                {"type": "chart", "description": "AI医生应用效果趋势"},
            ],
        },
        domain="医疗健康",
        language="zh",
        complexity="low",
        topic_result=topic_result,
    )

    chart_descriptions = [
        item["description"]
        for item in result["multimodal_requirements"]
        if item["type"] == "chart"
    ]
    assert len(result["multimodal_requirements"]) >= 4
    assert len(chart_descriptions) >= 3
    assert any("折线图" in item or "柱状图" in item or "表格" in item or "雷达图" in item for item in chart_descriptions)


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
            {"type": "chart", "description": "主流工具在效能维度上的表现"},
        ],
    }

    query = synthesize_query_description(task, language="zh", freshness_window="2023-2025")

    assert "某三级甲等医院的AI医疗应用主管" in query
    assert "正在撰写一份深度研究报告" in query
    assert "2023-2025年" in query
    assert "请聚焦2023-2025年的最新数据、政策文本与权威文献" in query
    assert "（1）系统梳理核心技术演进：" in query
    assert "（2）分析慢性病筛查与急诊分诊中的典型案例：" in query
    assert "（3）评估伦理隐私风险与监管要求：" in query
    assert "引用AI医生诊断流程图" in query
    assert "绘制一张雷达图，展示主流工具在效能维度上的表现" in query
    assert "明确关键证据、局限性、优先级排序与后续行动建议" in query


def test_synthesize_query_description_inferrs_chart_type_from_trend_language():
    task = {
        "user_role": "某制造业协会政策顾问",
        "main_task": "分析后疫情时代全球供应链重组对中国制造业的影响",
        "sub_questions": ["评估电子与汽车行业的出口变化"],
        "multimodal_requirements": [
            {"type": "chart", "description": "中国制造业出口变化趋势"},
        ],
    }

    query = synthesize_query_description(task, language="zh", freshness_window="2023-2025")

    assert "绘制一张折线图，展示中国制造业出口变化趋势" in query
