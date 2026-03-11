# TVIR: Building Deep Research Agents Towards Text-Visual Interleaved Report Generation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 Overview

TVIR (Text-Visual Interleaved Report) is a deep research agent system powered by large language models, designed to automatically generate high-quality research reports with interleaved text and visual content. The system features:

- 🔍 **Intelligent Information Retrieval**: Multi-round deep information retrieval via Google Search API
- 📊 **Data Visualization**: Automatic generation of charts and data visualizations
- 🖼️ **Multimodal Fusion**: Integration of text, images, charts, and other media formats
- 📝 **Structured Reports**: Generation of professional research reports with clear logic and proper citations
- 🎯 **Task Alignment**: Precise understanding of user requirements to generate scenario-specific reports

## 📦 Dataset

The benchmark dataset is located at `benchmark/data/query.json`, containing 100 carefully curated deep research tasks (50 in Chinese, 50 in English) across 10 domains. Each task includes:

- **Query ID**: Unique identifier (e.g., `000001`, `000002`)
- **Task Description**: User-specific research requirements with explicit multimodal integration needs (text, images, charts)
- **Evaluation Checklist**: Structured verification criteria aligned with task requirements

Tasks are designed around real-world research scenarios, requiring multi-source information integration, critical analysis, and text-visual interleaved report generation.

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Analytical Depth & Breadth** |  | 
| **Citation Support** |  | 
| **Factual & Logical Consistency** |  | 
| **Instruction Alignment** |  | 
| **Writing Quality** |  | 
| **Chart-Source Consistency** |  |
| **Figure Caption Quality** | | 
| **Figure Context Integration** |  | 
| **Figure Quality** |  | 
| **Multimodal Composition** | | 

## 🏗️ Project Architecture

```
TVIR/
├── agent/                     # Agent core module
│   ├── main.py                # Main entry point
│   ├── run_agent.sh           # Run script
│   ├── conf/                  # Configuration files
│   │   ├── config.yaml        # Main configuration
│   │   ├── agent/             # Agent configurations
│   │   └── llm/               # LLM model configurations
│   └── src/                   # Source code
│       ├── core/              # Core logic (orchestrator, pipeline)
│       ├── io/                # Input/output handling
│       ├── llm/               # LLM client wrappers
│       ├── logging/           # Logging system
│       └── utils/             # Utility functions
├── benchmark/                 # Evaluation benchmark
│   ├── eval.py                # Evaluation script
│   ├── preprocess.py          # Preprocessing script
│   ├── data/                  # Benchmark dataset
│   ├── reports/               # Reports to be evaluated
│   └── scripts/               # Evaluation scripts
│       ├── preprocess/        # Preprocessing modules
│       └── evaluation/        # Evaluation metric modules
├── libs/                      # Third-party libraries
│   └── miroflow-tools/        # MCP tool integration
└── logs/                      # Log output directory
```

## 🚀 Quick Start

### Requirements

- Python 3.12+
- macOS / Linux / Windows
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

#### 1. Install uv Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clone the Repository

```bash
git clone https://github.com/NJU-LINK/TVIR.git
cd TVIR
```

#### 3. Configure Environment Variables

Copy the environment variable template and fill in the required API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and configure the following required API keys:

```bash
# ============================================
# Required Configuration for Agent
# ============================================

# API for Google Search
SERPER_API_KEY=your_serper_key
SERPER_BASE_URL=https://google.serper.dev

# API for Linux Sandbox
E2B_API_KEY=your_e2b_key

# Model for VQA
VQA_MODEL_NAME=gpt-41-0414-global

# API for OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1

# ============================================
# Required Configuration for Benchmark
# ============================================

# API for Google Search
SERPER_API_KEY=your_serper_key
SERPER_BASE_URL=https://google.serper.dev

# API for OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Evaluation Settings
EVAL_MODEL_NAME=gpt-5.2-1211-global
MAX_RETRIES=3
TEMPERATURE=0
STREAMING=true
```

#### 4. Install Dependencies

```bash
uv sync
```

## 💻 Usage

### Running the Agent to Generate Reports

#### Method 1: Using Benchmark Dataset

Load predefined query tasks from `benchmark/data/query.json`:

```bash
cd agent
bash run_agent.sh claude-4-5 000001
```

**Parameters:**
- `claude-4-5`: LLM model configuration to use (options: `claude-4-5`, `qwen-3`, `glm-4-7`, etc.)
- `000001`: Query ID corresponding to a task in `benchmark/data/query.json`

#### Method 2: Custom Task Description

Write your task description directly in `agent/main.py`:

```python
# Modify task description in main.py
task_description = """
Write a technical analysis report on memory mechanisms in large language model agents, including architecture diagrams, timelines, and other visualizations...
"""
```

Then run:

```bash
cd agent
bash run_agent.sh claude-4-5
```

### Running Benchmark Evaluation

#### Step 1: Prepare Report Files

Organize reports to be evaluated in the following directory structure:

```
benchmark/reports/{eval_system_name}/{query_id}/
├── report.md          # Main report file
├── images/            # Images retrieved from web
│   ├── image1.jpg
│   └── image2.png
└── charts/            # Charts generated by tools
    ├── chart1.png
    └── chart2.svg
```

**Notes:**
- Images in `report.md` can use local paths (e.g., `./images/pic.jpg`) or HTTP links
- `eval_system_name` is the evaluation system name (e.g., `claude`, `gpt4`, `qwen`)
- `query_id` must correspond to an ID in `benchmark/data/query.json`

#### Step 2: Preprocess Reports

Preprocess reports to extract citations, charts, and other information:

```bash
cd benchmark

# Preprocess a single report
uv run python preprocess.py \
  --report_root_dir reports \
  --eval_system_name claude \
  --query_id 000001

# Batch preprocessing
uv run python preprocess.py \
  --report_root_dir reports \
  --eval_system_name claude gpt4 \
  --query_id 000001 000002 000003
```

**Parameters:**
- `--report_root_dir`: Report root directory (default: `reports`)
- `--eval_system_name`: Evaluation system name(s), supports multiple
- `--query_id`: Query ID(s), supports multiple

#### Step 3: Run Evaluation

Execute automated evaluation to generate results:

```bash
cd benchmark

# Evaluate a single report
uv run python eval.py \
  --report_root_dir reports \
  --result_root_dir eval_results \
  --eval_system_name claude \
  --query_id 000001

# Batch evaluation
uv run python eval.py \
  --report_root_dir reports \
  --result_root_dir eval_results \
  --eval_system_name claude gpt4 \
  --query_id 000001 000002 000003
```

**Parameters:**
- `--report_root_dir`: Report root directory
- `--result_root_dir`: Evaluation result output directory
- `--eval_system_name`: Evaluation system name(s), supports multiple
- `--query_id`: Query ID(s), supports multiple

#### Step 4: Generate Summary Report

Run the summary script to aggregate evaluation results across all models and dimensions:

```bash
uv run python score_generate_result.py
```

This will generate `model_dimension_summary.xlsx` containing average scores for each model across text dimensions, visual dimensions, and overall performance.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🌟 Star History

If this project helps you, please give us a Star ⭐️

---
