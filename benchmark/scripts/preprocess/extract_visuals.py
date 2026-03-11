import json
import os
from openai import OpenAI
import argparse
import time
from dotenv import load_dotenv

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


def extract_visuals(report_root_dir, query_id, eval_system_name):
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report.md"
    VISUAL_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/visuals.json"

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return {"error": "Report file not found"}

    visuals = []

    system_prompt = """
You are an expert academic information extractor. Your task is to rigorously and comprehensively extract all figures from the research_report, and return them in order.

EXTRACTION GUIDELINES:
- Only extract figures such as images, charts, graphs, diagrams and etc.
- For each figure, return:
  - caption: The complete figure caption/title exactly as it appears. (verbatim; do not rewrite or translate).
  - contents: The file path(s) or URL(s) of the figure.
  - context: If the figure is explicitly referenced in the main text (e.g., “see Figure 2”, “Figure 2 shows…”), extract the **sentence(s) or paragraph** that contains the reference and explains it. If there is no explicit reference, extract the **surrounding text near the figure**, and **insert the placeholder <figure> on its own line at the exact position where the figure appears**.
  - ref_idxs: Include **only** citation numbers that appear **within the figure's caption or associated legend** (typically indicating the data source). If none are directly mentioned, return an empty list.

OUTPUT FORMAT:
Respond with a JSON list. Each item:
{
  "caption": "Title or caption of the figure",
  "contents": ["Path(s) or URL(s) of the figure"],
  "context": "Contextual text from the report",
  "ref_idxs": ["1", "2"]
}
"""

    print("⏳ Extracting visual elements...")

    user_prompt = f"""
Please extract all figures from the following research_report according to the guidelines.

<research_report>
{report_content}
</research_report>

Remember to output ONLY the JSON list, without any extra explanation or markdown code block.
"""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                stream=STREAMING,
            )

            if STREAMING:
                result_json_str = ""
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            result_json_str += delta.content
            else:
                result_json_str = response.choices[0].message.content

            if isinstance(result_json_str, str):
                if result_json_str.strip().startswith("```json"):
                    result_json_str = result_json_str.strip()[7:]
                if result_json_str.strip().endswith("```"):
                    result_json_str = result_json_str.strip()[:-3]
            result_json = json.loads(result_json_str)
            if isinstance(result_json, dict) and "error" in result_json:
                raise ValueError(f"Response contains error: {result_json['error']}")
            if isinstance(result_json, dict) and len(result_json) == 1:
                result_json = next(iter(result_json.values()))

            for item in result_json:
                contents = item.get("contents", [])
                for content in contents:
                    visual = {
                        "caption": item.get("caption", ""),
                        "content": content,
                        "context": item.get("context", ""),
                        "ref_idxs": item.get("ref_idxs", []),
                    }
                    if (
                        content.startswith("images/")
                        or content.startswith("http")
                        or content.startswith("./images/")
                    ):
                        visual["type"] = "image"
                    elif content.startswith("charts/") or content.startswith(
                        "./charts/"
                    ):
                        visual["type"] = "chart"
                    else:
                        visual["type"] = "unknown"
                    visuals.append(visual)

            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(4**attempt)

    if last_error is not None:
        print(f"❌ Error during visual element extraction: {last_error}")
        visuals = {"error": str(last_error)}

    os.makedirs(os.path.dirname(VISUAL_PATH), exist_ok=True)
    with open(VISUAL_PATH, "w", encoding="utf-8") as f:
        json.dump(visuals, f, ensure_ascii=False, indent=2)
    print(
        f"✅ Visual element extraction completed, results saved to {VISUAL_PATH} | Number of visual elements: {len(visuals)}"
    )

    return visuals


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
    args = parser.parse_args()
    extract_visuals(args.report_root_dir, args.query_id, args.eval_system_name)
