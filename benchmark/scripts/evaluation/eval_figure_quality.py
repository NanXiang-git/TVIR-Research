import json
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.2")


def eval_figure_quality(report_root_dir, query_id, eval_system_name, result_root_dir):
    EVAL_CHART_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/chart_quality.json"
    EVAL_IMAGE_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/image_quality.json"
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/figure_quality.json"

    try:
        with open(EVAL_CHART_PATH, "r", encoding="utf-8") as f:
            chart_result = json.load(f)
        with open(EVAL_IMAGE_PATH, "r", encoding="utf-8") as f:
            image_result = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    chart_score = chart_result.get("normalized_score")
    image_score = image_result.get("normalized_score")
    chart_count = chart_result.get("scored_item_count")
    image_count = image_result.get("scored_item_count")

    total_count = chart_count + image_count
    if total_count > 0 and chart_score is not None and image_score is not None:
        final_score = (
            chart_score * chart_count + image_score * image_count
        ) / total_count
    elif chart_score is not None:
        final_score = chart_score
    elif image_score is not None:
        final_score = image_score
    else:
        final_score = None

    result = {
        "normalized_score": (
            round(final_score, 4) if final_score is not None else None
        ),
        "scored_item_count": total_count,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {EVAL_RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root_dir",
        type=str,
        required=True,
        help="Root directory path for reports",
    )
    parser.add_argument(
        "--query_id", type=str, required=True, help="Evaluation report ID"
    )
    parser.add_argument(
        "--eval_system_name", type=str, required=True, help="Evaluation system name"
    )
    parser.add_argument(
        "--result_root_dir",
        type=str,
        required=True,
        help="Root directory path for evaluation results",
    )
    args = parser.parse_args()
    eval_figure_quality(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
