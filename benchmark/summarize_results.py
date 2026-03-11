import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv

load_dotenv()

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.2")


def safe_round(value, decimals=4):
    """Safely round a value, returning NaN if input is NaN."""
    return round(value, decimals) if not np.isnan(value) else np.nan


def calculate_mean_scores(dimension_scores, dimensions):
    """Calculate mean of scores for given dimensions, filtering out NaN values."""
    scores = [
        dimension_scores.get(d, np.nan) for d in dimensions if d in dimension_scores
    ]
    scores = [s for s in scores if not np.isnan(s)]
    return np.mean(scores) if scores else np.nan


def extract_scores(eval_result_dir):
    """
    Extract normalized scores from all evaluation dimensions.

    Returns:
        dict: {dimension: {model: {query_id: [scores]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Iterate through model directories
    for model_dir in eval_result_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Iterate through query_id directories
        for query_dir in model_dir.iterdir():
            if not query_dir.is_dir():
                continue
            query_id = query_dir.name

            # Iterate through evaluation model directories (e.g., gpt-5.2-*)
            for gpt_dir in query_dir.glob(f"{EVAL_MODEL_NAME}*"):
                if not gpt_dir.is_dir():
                    continue

                # Read all .json files (evaluation dimensions)
                for json_file in gpt_dir.glob("*.json"):
                    dimension = json_file.stem  # Use filename as dimension name

                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            content = json.load(f)
                            score = content.get("normalized_score")
                            if score is not None:
                                data[dimension][model_name][query_id].append(score)
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    return data


def create_model_dimension_summary(all_data, output_dir):
    """
    Create model-dimension summary table.
    Each row represents a model, each column represents a dimension.
    Values are the average scores across all queries for that model-dimension pair.

    Args:
        all_data: {dimension: {model: {query_id: [scores]}}}
        output_dir: Output directory path

    Returns:
        DataFrame: Summary table
    """
    # Collect all models
    all_models = set()
    for dimension_data in all_data.values():
        all_models.update(dimension_data.keys())

    models = sorted(all_models)

    # Define dimensions to exclude
    excluded_dimensions = {"image_quality", "chart_quality"}

    # Define text and visual dimensions (based on benchmark script names)
    text_dimensions = [
        "instruction_alignment",
        "citation_support",
        "analytical_depth_breadth",
        "factual_logical_consistency",
        "writing_quality",
    ]
    visual_dimensions = [
        "figure_quality",
        "chart_source_consistency",
        "figure_caption_quality",
        "figure_context_integration",
        "multimodal_composition",
    ]

    # Build summary data
    summary_data = []
    for model in models:
        row = {"model": model}

        # Calculate scores for each dimension
        dimension_scores = {}
        for dimension, dimension_data in all_data.items():
            if dimension in excluded_dimensions:
                continue

            model_data = dimension_data.get(model, {})
            # Collect all scores for this model in this dimension across all queries
            all_scores = [
                score for query_scores in model_data.values() for score in query_scores
            ]
            # Calculate mean score
            dimension_scores[dimension] = np.mean(all_scores) if all_scores else np.nan

        # Add text dimension scores
        for dimension in text_dimensions:
            if dimension in dimension_scores:
                row[dimension] = safe_round(dimension_scores[dimension])

        # Calculate text average
        row["text"] = safe_round(
            calculate_mean_scores(dimension_scores, text_dimensions)
        )

        # Add visual dimension scores
        for dimension in visual_dimensions:
            if dimension in dimension_scores:
                row[dimension] = safe_round(dimension_scores[dimension])

        # Calculate visual average
        row["visual"] = safe_round(
            calculate_mean_scores(dimension_scores, visual_dimensions)
        )

        # Calculate overall average (all ten dimensions)
        all_ten_dimensions = text_dimensions + visual_dimensions
        row["overall"] = safe_round(
            calculate_mean_scores(dimension_scores, all_ten_dimensions)
        )

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    # Ensure column order
    column_order = (
        ["model"]
        + text_dimensions
        + ["text"]
        + visual_dimensions
        + ["visual", "overall"]
    )
    # Keep only existing columns
    column_order = [col for col in column_order if col in df_summary.columns]
    df_summary = df_summary[column_order]

    # Save to Excel
    output_file = Path(output_dir) / "model_dimension_summary.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Created model-dimension summary file: {output_file}")

    return df_summary


def main():
    """
    Main function to extract scores and generate analysis reports.
    """
    base_dir = "./"
    eval_result_dir = Path(base_dir) / "eval_results"

    print("Extracting scores...")
    all_data = extract_scores(eval_result_dir)

    print("Creating model-dimension summary table...")
    create_model_dimension_summary(all_data, eval_result_dir)


if __name__ == "__main__":
    main()
