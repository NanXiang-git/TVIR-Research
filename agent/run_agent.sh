#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: ./run_agent.sh <llm_model> <domain> <complexity> [language]"
    echo "Example: ./run_agent.sh default 环境能源 high zh"
    echo "Example: ./run_agent.sh claude-4-5 healthcare medium en"
    exit 1
fi

LLM_MODEL=$1
DOMAIN=$2
COMPLEXITY=$3
LANGUAGE=${4:-zh}

uv run python main.py \
  --llm-config "$LLM_MODEL" \
  --agent-config tvir_agent \
  --domain "$DOMAIN" \
  --complexity "$COMPLEXITY" \
  --language "$LANGUAGE"
