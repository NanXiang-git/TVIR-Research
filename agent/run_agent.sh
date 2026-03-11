#!/bin/bash

# Check if required arguments are provided
if [ -z "$1" ]; then
    echo "Usage: ./run_agent.sh <llm_model> [query_id]"
    echo "Example: ./run_agent.sh claude-4-5 016001"
    echo "Example: ./run_agent.sh claude-4-5"
    exit 1
fi

LLM_MODEL=$1
QUERY_ID=$2

# Set QUERY_ID as environment variable if provided
if [ -n "$QUERY_ID" ]; then
    export QUERY_ID
fi

# Run with clean config, query_id passed via environment
uv run python main.py --config-name config \
  llm=$LLM_MODEL \
  agent=tvir_agent