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
    topic_result = normalize_topic_result(
        {}, domain="environmental policy", complexity="medium"
    )
    result = normalize_query_result(
        raw={"main_task": "Analyze the latest methane policy shifts"},
        domain="environmental policy",
        language="en",
        complexity="medium",
        topic_result=topic_result,
    )

    assert result["complexity"] == "medium"
    assert result["user_role"]
    assert len(result["sub_questions"]) == 4
    assert len(result["question_details"]) == 4
    assert any(item["type"] == "image" for item in result["multimodal_requirements"])
    assert any(item["type"] == "chart" for item in result["multimodal_requirements"])


def test_normalize_query_result_adds_question_level_details_and_chart_types():
    topic_result = normalize_topic_result({}, domain="医疗健康", complexity="high")
    result = normalize_query_result(
        raw={
            "main_task": "分析AI医生在诊断场景中的应用演进",
            "sub_questions": [
                "系统梳理核心技术演进",
                "分析典型应用案例与落地场景",
                "评估性能指标对比与监管风险",
            ],
            "question_details": [
                {
                    "title": "系统梳理核心技术演进",
                    "analysis_points": ["基础模型框架", "多模态诊断流程", "部署约束"],
                    "citations": [
                        {
                            "title": "Nature Medicine 2024",
                            "url": "https://example.com/nature-2024",
                            "used_for": "技术演进判断",
                            "key_finding": "模型在影像诊断中的表现提升",
                        }
                    ],
                }
            ],
            "multimodal_requirements": [
                {"type": "image", "description": "AI医生诊断流程图"},
                {"type": "chart", "description": "AI医生应用效果趋势"},
            ],
        },
        domain="医疗健康",
        language="zh",
        complexity="high",
        topic_result=topic_result,
    )

    chart_descriptions = [
        item["description"]
        for item in result["multimodal_requirements"]
        if item["type"] == "chart"
    ]

    assert len(result["sub_questions"]) == 5
    assert len(result["question_details"]) == 5
    assert len(result["question_details"][0]["analysis_points"]) == 3
    assert result["question_details"][0]["citations"][0]["title"] == "Nature Medicine 2024"
    assert len(chart_descriptions) >= 3
    assert any(
        keyword in item
        for item in chart_descriptions
        for keyword in ("折线图", "柱状图", "表格", "雷达图")
    )


def test_infer_quality_checks_detects_nested_citations():
    task = {
        "user_role": "Director of strategy at a clean-energy utility",
        "sub_questions": ["Q1", "Q2", "Q3"],
        "question_details": [
            {
                "title": "Q1",
                "analysis_points": ["A", "B"],
                "citations": [
                    {
                        "title": "Nature Energy 2024",
                        "url": "https://example.com/nature-energy-2024",
                        "key_finding": "Battery performance improved in 2024",
                    }
                ],
            }
        ],
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


def test_synthesize_query_description_respects_complexity():
    base_task = {
        "user_role": "某三级甲等医院的AI医疗应用主管",
        "main_task": "聚焦AI医生在诊断领域的智能化进展",
        "sub_questions": [
            "核心技术演进梳理",
            "典型应用案例分析",
            "监管风险与治理评估",
        ],
        "multimodal_requirements": [
            {"type": "image", "description": "AI医生诊断流程图"},
            {"type": "chart", "description": "主流工具在效能维度上的表现"},
            {"type": "chart", "description": "诊断准确率变化趋势"},
        ],
    }

    low_query = synthesize_query_description(
        {**base_task, "complexity": "low"},
        language="zh",
        freshness_window="2023-2025",
    )
    high_query = synthesize_query_description(
        {
            **base_task,
            "complexity": "high",
            "question_details": [
                {
                    "title": "核心技术演进梳理",
                    "analysis_points": ["基础框架", "多模态能力", "部署门槛"],
                    "citations": [
                        {
                            "title": "Nature Medicine 2024",
                            "url": "https://example.com/nm-2024",
                        }
                    ],
                },
                {
                    "title": "典型应用案例分析",
                    "analysis_points": ["慢病筛查", "急诊分诊", "患者获益"],
                    "citations": [],
                },
                {
                    "title": "监管风险与治理评估",
                    "analysis_points": ["隐私风险", "监管框架", "改进建议"],
                    "citations": [],
                },
            ],
        },
        language="zh",
        freshness_window="2023-2025",
    )

    assert "重点回答以下问题" in low_query
    assert "系统性研究" in high_query
    assert "至少覆盖" in high_query
    assert len(high_query) > len(low_query)


def test_synthesize_query_description_mentions_problem_level_citation_for_high():
    task = {
        "complexity": "high",
        "user_role": "某商业银行数字支付业务主管",
        "main_task": "评估数字人民币推广策略与场景落地路径",
        "sub_questions": [
            "技术框架与机制升级",
            "国内应用场景与交易活跃度变化",
            "跨境试点与国际竞争评估",
        ],
        "question_details": [
            {
                "title": "技术框架与机制升级",
                "analysis_points": ["双层运营架构", "钱包分级", "可控匿名机制"],
                "citations": [
                    {
                        "title": "中国数字人民币研发白皮书",
                        "url": "https://example.com/whitepaper",
                        "used_for": "技术框架",
                    }
                ],
            },
            {
                "title": "国内应用场景与交易活跃度变化",
                "analysis_points": ["零售消费", "公共缴费", "钱包活跃度"],
                "citations": [],
            },
            {
                "title": "跨境试点与国际竞争评估",
                "analysis_points": ["mBridge", "香港试点", "国际竞争"],
                "citations": [],
            },
        ],
        "multimodal_requirements": [
            {"type": "image", "description": "数字人民币双层运营架构示意图"},
            {"type": "chart", "description": "累计交易金额与笔数趋势折线图"},
            {"type": "chart", "description": "主要经济体CBDC试点阶段对比表格"},
        ],
    }

    query = synthesize_query_description(task, language="zh", freshness_window="2023-2025")

    assert "优先标注与解读《中国数字人民币研发白皮书》" in query
