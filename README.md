# TVIR Automation Task Generator

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package_manager-uv-6e56cf.svg)](https://github.com/astral-sh/uv)

</div>

TVIR Automation Task Generator 是一个基于 TVIR 多智能体框架重构而来的自动化任务生成系统。它不再以“生成图文研究报告”为主，而是以“生成高质量、可执行、可校验的研究任务描述”为核心目标。

系统输入为：

- `domain`：任务领域
- `language`：输出语言
- `complexity`：任务复杂度

系统输出为结构化任务 JSON，既包含适合程序消费的字段，也包含可直接交给研究员、分析师或下游 Agent 执行的完整 `query`。

## 项目定位

这个项目适合以下场景：

- 为深度研究 Agent 自动生成高质量研究任务
- 为政府、企业、高校、研究机构快速构建任务 brief
- 为 benchmark、评测集、任务池批量生成标准化 query
- 为下游报告生成器、检索 Agent、分析 Agent 提供稳定输入

相比简单的 prompt 拼接，本项目强调：

- 用户角色明确
- 任务分解清晰
- 研究维度有深度
- 引用来源可追溯
- 多模态要求真实可执行

## 本版已完成的改进

这一版在原有自动化任务生成基础上，完成了以下关键更新：

- 三阶段自动化 pipeline 已稳定落地：`Topic Generation -> Query Construction -> Refine + Stabilize`
- Prompt 系统已重构为分段注入形式，不再只依赖单条简单 system prompt
- 已增强中文支持，三阶段 user prompt 和 system prompt 都能更自然地输出中文任务
- 已补充 few-shot 风格约束，使最终 query 更接近专业研究 brief，而不是普通问答题
- 已引入多模态分类定义，区分 `image / chart / table`，并要求视觉元素服务于具体研究维度
- 已加强复杂度控制：
  - `low`：更短、更聚焦，偏一级研究维度
  - `medium`：标准长度，适合常规深度研究任务
  - `high`：更详细，维度展开更多，要求更专业
- 已支持生成完整任务描述 `query`，而不仅是 `sub_questions` 和 `multimodal_requirements`
- 已优化 query 写法，使其更接近真实项目需求文档，强调研究背景、用户角色、时间范围、数据来源、图表要求与结论目标
- 已增强 citation 结构，支持更好地表达“哪个问题使用了哪些引用”
- 已完善图片与图表稳定化逻辑，用于提升 `quality_checks` 通过率
- 已修复 GPT-4o 工具调用链路中的关键兼容问题：
  - 支持 `finish_reason=tool_calls`
  - 支持流式/非流式工具调用解析
  - 修复“无文本但有 tool calls 被误判为 LLM 调用失败”的循环问题
- README 与入口说明已同步到当前“任务生成器”版本，便于直接发布到 GitHub

## 核心能力

### 1. 三阶段自动化流程

系统按以下三阶段工作：

1. `Topic Generation`
   基于输入领域生成前沿、真实、有研究价值的主题候选。
2. `Query Construction`
   将主题转化为结构化研究任务，生成角色、主任务、研究维度、多模态要求与引用。
3. `Refine + Stabilize`
   对任务结果进行检查、修复和程序级稳定化，尽量保证图片链接、图表来源、时效性检查和最终 query 的可靠性。

### 2. 五项任务设计原则

生成结果围绕以下五项原则约束：

1. 用户特定
   必须明确指定用户角色，如“某省级生态环境厅负责人”“某商业银行数字支付业务主管”。
2. 需求明确
   任务需拆解为多个互相关联的研究维度，而不是单一笼统问题。
3. 深度研究
   每个维度应包含驱动因素、案例、指标、风险、政策、趋势等分析要求。
4. 前沿性
   优先关注近 2-3 年技术、政策、产业与案例进展，并优先使用 2021 年后的来源。
5. 多模态融合
   必须包含真实可获取的图片、图表或表格要求，且这些视觉元素服务于分析任务本身。

### 3. 双层输出

最终结果同时提供两类输出：

- 结构化 JSON 字段，便于程序处理
- 一段完整、专业、可直接交付的 `query`

## 输出示例

```json
{
  "user_role": "某商业银行数字支付业务主管",
  "main_task": "评估数字人民币推广策略与场景落地路径",
  "sub_questions": [
    "技术框架与机制升级",
    "国内应用场景与交易活跃度变化",
    "跨境试点、国际竞争与监管风险评估"
  ],
  "multimodal_requirements": [
    {
      "type": "image",
      "description": "数字人民币双层运营架构示意图",
      "source": "https://example.com/architecture.png"
    },
    {
      "type": "chart",
      "description": "累计交易金额与笔数趋势折线图",
      "data_source": "https://example.com/data-report"
    },
    {
      "type": "chart",
      "description": "中国、欧盟、美国 CBDC 试点阶段与交易规模对比表",
      "data_source": "https://example.com/cbdc-comparison"
    }
  ],
  "quality_checks": {
    "role_clear": true,
    "sub_questions_coherent": true,
    "timely_sources": true,
    "image_accessible": true,
    "chart_reproducible": true
  },
  "query": "本人作为某商业银行数字支付业务主管，正在撰写一份深度研究报告。请聚焦 2023-2025 年的最新数据、政策文本与权威文献，围绕数字人民币推广策略与场景落地路径展开系统性研究，结合多源证据进行批判性分析，重点覆盖以下维度：..."
}
```

## 项目结构

```text
TVIR/
├─ agent/
│  ├─ main.py
│  ├─ run_agent.sh
│  ├─ conf/
│  │  ├─ config.yaml
│  │  ├─ agent/
│  │  │  ├─ tvir_agent.yaml
│  │  │  └─ default.yaml
│  │  └─ llm/
│  │     ├─ default.yaml
│  │     ├─ claude-4-5.yaml
│  │     ├─ qwen-3.yaml
│  │     └─ glm-4-7.yaml
│  ├─ prompts/
│  │  ├─ topic_generation.json
│  │  ├─ query_construction.json
│  │  └─ refine_validation.json
│  └─ src/
│     ├─ core/
│     │  ├─ orchestrator.py
│     │  ├─ pipeline.py
│     │  └─ automation_utils.py
│     ├─ llm/
│     ├─ io/
│     └─ utils/
│        └─ automation_prompt_loader.py
├─ tests/
│  └─ test_automation_utils.py
├─ logs/
├─ benchmark/
├─ libs/
├─ .env.example
├─ pyproject.toml
└─ README.md
```

## 技术架构

### 工作流概览

```mermaid
flowchart TD
    A["Input: domain / language / complexity"] --> B["Topic Generation"]
    B --> C["Query Construction"]
    C --> D["Refine Inspection"]
    D --> E["Refine Repair"]
    E --> F["Deterministic Stabilization"]
    F --> G["Final JSON + query"]
```

### 工具职责

- `google_search`
  用于发现前沿主题、政策、论文、案例和图表来源。
- `scrape_website`
  用于验证页面内容、实体、数据来源和可追溯性。
- `google_image_search`
  用于检索公开可访问的图片候选。
- `visual_question_answering`
  用于验证图片是否可访问，以及是否与描述匹配。

## 环境要求

- Python `3.12+`
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖
- 需要可用的 LLM、搜索、图片检索和 VQA 相关 API

## 安装与部署

### 1. 克隆仓库

```bash
git clone https://github.com/<your-org-or-name>/TVIR.git
cd TVIR
```

### 2. 安装 uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 配置环境变量

复制模板文件：

```bash
cp .env.example .env
```

至少需要根据你的部署环境配置以下变量：

```bash
# Search
SERPER_API_KEY=your_serper_key
SERPER_BASE_URL=https://google.serper.dev

# LLM
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional model override
OPENAI_MODEL_NAME=gpt-4o

# VQA
VQA_MODEL_NAME=gpt-4o

# Optional scraping
JINA_API_KEY=your_jina_key
JINA_BASE_URL=https://r.jina.ai
```

如果你使用自定义模型配置，请同步检查：

- [agent/conf/llm](./agent/conf/llm)
- [agent/conf/agent](./agent/conf/agent)

## 快速开始

### 推荐运行方式

从仓库根目录运行：

```bash
uv run python agent/main.py \
  --domain 环境能源 \
  --language zh \
  --complexity high \
  --llm-config default \
  --agent-config tvir_agent
```

也可以先进入 `agent/` 目录运行：

```bash
cd agent
uv run python main.py \
  --domain 医疗健康 \
  --language zh \
  --complexity medium \
  --llm-config default \
  --agent-config tvir_agent
```

### 参数说明

- `--domain`
  任务领域，如 `环境能源`、`医疗健康`、`金融科技`
- `--language`
  输出语言，可选 `zh` 或 `en`
- `--complexity`
  复杂度，可选 `low / medium / high`
- `--llm-config`
  使用的 Hydra 模型配置名，默认 `default`
- `--agent-config`
  使用的 agent 编排配置名，默认 `tvir_agent`

### Shell 脚本运行

```bash
cd agent
bash run_agent.sh default 环境能源 high zh
```

## 结果输出

每次运行都会在 `agent/results/<model_name>/automation_<timestamp>/` 下生成结果目录，例如：

```text
agent/results/gpt-4o/automation_20260408_121203/
├─ 01_topic_candidates.json
├─ 02_query_draft.json
├─ 03_refine_review.json
├─ 04_query_repaired.json
├─ 05_refine_recheck.json
├─ 05b_stabilized_query.json
└─ 06_final_task.json
```

其中：

- `01_topic_candidates.json`
  主题候选及选中主题
- `02_query_draft.json`
  初始结构化任务
- `03_refine_review.json`
  质量审查结果
- `04_query_repaired.json`
  修复后的任务结果
- `05b_stabilized_query.json`
  程序级稳定化后的结果
- `06_final_task.json`
  最终输出，可直接用于下游执行

## 质量检查

最终结果中包含以下质量检查字段：

- `role_clear`
  用户角色是否明确
- `sub_questions_coherent`
  研究维度是否相互关联并支撑主任务
- `timely_sources`
  是否优先引用 2021 年后的来源
- `image_accessible`
  图片链接是否真实可访问且内容匹配
- `chart_reproducible`
  图表数据源是否可追溯、可复现

## 测试

运行与自动化任务生成相关的单测：

```bash
uv run python -m pytest tests/test_automation_utils.py
```

这些测试主要覆盖：

- topic 结果规范化
- query 结果规范化
- 质量检查推断
- 最终 query 合成逻辑
- 图表类型推断与多模态覆盖

## 适合二次开发的方向

你可以在此基础上继续扩展：

- 增加新的领域模板与角色模板
- 增加更多图表类型推断规则
- 引入更严格的来源评分机制
- 将最终输出接入报告生成器或评测系统
- 将 `query` 转成 dataset、benchmark 或任务池

## 与原 TVIR 的关系

本仓库保留了 TVIR 的部分历史结构与 benchmark 目录，但当前 README 描述的主能力已经切换为：

**面向自动化研究任务生成的多智能体 pipeline**

如果你只关心当前主功能，建议重点关注：

- [agent/main.py](./agent/main.py)
- [agent/src/core/orchestrator.py](./agent/src/core/orchestrator.py)
- [agent/src/core/automation_utils.py](./agent/src/core/automation_utils.py)
- [agent/prompts/query_construction.json](./agent/prompts/query_construction.json)

## License

This project is licensed under the [MIT License](./LICENSE).

## Acknowledgement

本项目基于 TVIR 原有多智能体框架重构，并围绕“自动化任务生成”场景进行了重新编排、提示词增强、质量校验与结果稳定化增强。
