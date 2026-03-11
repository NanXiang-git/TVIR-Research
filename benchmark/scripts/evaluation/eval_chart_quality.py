import json
import os
from openai import OpenAI
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from json_repair import repair_json
import time

load_dotenv()

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
STREAMING = os.getenv("STREAMING", "true").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def _parse_json_result(result):
    """Parse agent-planner result using json-repair library."""

    # Remove BOM and invisible characters
    text = result.strip("\ufeff\u200b\u200c\u200d")

    # Use json-repair to fix and parse
    try:
        repaired = repair_json(text)
        output = json.loads(repaired)

        if not isinstance(output, dict):
            raise ValueError(f"Invalid type: {type(output)}")

        return output
    except Exception as e:
        raise ValueError(f"Could not parse JSON: {e}")


def eval_chart_quality(report_root_dir, query_id, eval_system_name, result_root_dir):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/chart_quality.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    charts = [fig for fig in figures if fig.get("type") == "chart"]

    if not charts:
        result = {
            "normalized_score": 0.0,
            "item_count": 0,
            "scored_item_count": 0,
            "evaluation_results": [],
        }
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No charts found, zero score saved to {EVAL_RESULT_PATH}")
        return

    evaluation_results = []
    total_scores = []

    # 2. Define System Prompt
    system_prompt = """
You are an expert Data Visualization Quality Assurance Specialist. Your task is to rigorously audit the visual quality of charts in a research report using a strict **Binary Checklist**.

GOAL:
Determine if the visual chart **fully, accurately, and clearly** meets the visual quality checklist items regarding layout, readability, and conciseness.

EVALUATION SCORING CRITERIA:
- **Score 1 (Pass)**: The chart is visually flawless regarding the specific criterion. No issues are visible.
- **Score 0 (Fail)**: The chart fails the criterion. Any visible defect (e.g., text overlap, unreadable font, broken encoding) results in a score of 0.

CHECKLIST ITEMS TO EVALUATE:

**A. Layout & Structure**
1. **No Overlap**
All text elements (titles, legends, axis labels, tick labels, annotations, data labels) and graphic elements do not overlap with each other or with the core data visualization components.
2. **No Misalignment**
Elements that should align (axis lines, tick marks with tick labels, legend markers with legend text, subplot panels in a grid) are consistently aligned, without clear, unintended offset or irregular positioning.
3. **Complete Visibility**
All chart content is fully visible within the canvas boundaries, with no elements cut off, clipped, or extending beyond the viewable area.
4. **Balanced Spatial Composition**
The chart's overall composition is balanced; the plot area, titles, legends, and annotations occupy appropriate proportions of the canvas; whitespace is distributed reasonably, appearing neither cramped nor empty.

**B. Readability**  
5. **Appropriate Text-to-Chart Proportion**
Text elements (titles, legends, axis labels, tick labels, data labels) are proportionate to the chart’s visual elements: text is neither dominating the plot area nor so small relative to marks/axes that it harms readability.
6. **Readable Contrast & Distinguishability**
Text is clearly readable against the background, and different series/categories are visually distinguishable. Fail if low contrast or overly similar colors make text or categories hard to read/identify.
7. **Correct Text Rendering**
All characters, symbols, and numbers are displayed correctly, without encoding errors, missing glyphs, or rendering issues.

**C. Conciseness**
8. **No Excessive Decoration**
The chart does not contain distracting or excessive decorative elements; all visual effects are restrained and purposeful, enhancing rather than obscuring data readability.
9. **No Information Overload**
The number of visual elements in the chart is not excessive; the chart avoids being too dense or overcrowded, which would cause difficulty in interpretation.
10. **No Redundant Labels**
Labels are used efficiently; the same information is not repeated or presented redundantly in ways that create visual clutter.

OUTPUT FORMAT:
Respond with a JSON object. Do not add markdown code blocks (like ```json) around the output.
{
  "scores": {
    "overlap": 0,
    "misalignment": 1,
    "visibility": 1,
    "spatial_composition": 1,
    "text_size": 0,
    "contrast": 1,
    "text_rendering": 1,
    "excessive_decoration": 1,
    "information_overload": 1,
    "redundant_labels": 1
  },
  "justification": {
    "overlap": "Specific description of overlap issue...",
    "text_size": "Specific description of font size issue..."
  },
  "suggestion": "Actionable advice to fix the issues..."
}
"""

    for fig in tqdm(charts, desc="🤖 Evaluating chart quality"):
        img_base64 = fig.get("img_base64", "")
        caption = fig.get("caption", "")

        if isinstance(img_base64, dict) and "error" in img_base64:
            evaluation_results.append(
                {"caption": caption, "error": img_base64["error"], "total_score": 0.0}
            )
            total_scores.append(0.0)
            continue

        user_message_content = [
            {
                "type": "text",
                "text": "Please evaluate the visual quality of the following chart based on the provided checklist. Remember to output ONLY the JSON object.",
            },
            {"type": "image_url", "image_url": {"url": img_base64}},
        ]

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=EVAL_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=TEMPERATURE,
                    stream=STREAMING,
                )

                if STREAMING:
                    result_str = ""
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                result_str += delta.content
                else:
                    result_str = response.choices[0].message.content

                if result_str.strip().startswith("```json"):
                    result_str = result_str.strip()[7:]
                if result_str.strip().endswith("```"):
                    result_str = result_str.strip()[:-3]
                evaluation_result = _parse_json_result(result_str)

                if "scores" not in evaluation_result:
                    raise ValueError(
                        f"Missing 'scores' field in response, result: {result_str}"
                    )

                last_err = None
                break

            except Exception as e:
                last_err = e
                time.sleep(4**attempt)

        if last_err is not None:
            print(f"❌ Error during evaluation: {last_err}")
            evaluation_results.append(
                {"caption": caption, "error": str(last_err), "total_score": None}
            )
            continue

        scores = evaluation_result.get("scores", {})
        if scores:
            evaluation_result["total_score"] = sum(scores.values())
            total_scores.append(evaluation_result["total_score"])
        else:
            evaluation_result["total_score"] = None
        evaluation_result["caption"] = caption
        evaluation_results.append(evaluation_result)

    if total_scores:
        max_score = 10
        normalized_score = sum(total_scores) / (len(total_scores) * max_score)
        normalized_score = round(normalized_score, 4)
    else:
        normalized_score = None

    result = {
        "normalized_score": normalized_score,
        "item_count": len(charts),
        "scored_item_count": len(total_scores),
        "evaluation_details": evaluation_results,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} | Normalized score: {normalized_score} | Scored items: {len(total_scores)}/{len(charts)}"
    )


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
    eval_chart_quality(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
