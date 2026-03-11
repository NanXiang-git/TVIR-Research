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


def extract_refs(report_root_dir, query_id, eval_system_name):
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report.md"
    REF_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/refs.json"

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return {"error": "Report file not found"}

    system_prompt = """
You are an expert academic information extractor. Your task is to rigorously and comprehensively extract all references from the research_report, and return them in order.

EXTRACTION GUIDELINES:
- Only extract the references section, not citation facts from the body.
- For each reference, return:
  - ref_idx: The citation number N, in order of appearance.
  - ref_url: The URL of the reference (if available, otherwise empty string).

OUTPUT FORMAT: 
Respond with a JSON list. Each item:
{
  "ref_idx": "Citation number",
  "ref_url": "URL of the reference"
}
"""

    print("⏳ Extracting references...")

    user_prompt = f"""
Please extract all references from the following research_report according to the guidelines.

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
            refs = json.loads(result_json_str)
            if isinstance(refs, dict) and "error" in refs:
                raise ValueError(f"Response contains error: {refs['error']}")
            while isinstance(refs, dict) and len(refs) == 1:
                refs = next(iter(refs.values()))
                
            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(4**attempt)

    if last_error is not None:
        print(f"❌ Error during reference extraction: {last_error}")
        refs = {"error": str(last_error)}

    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    with open(REF_PATH, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Reference extraction completed, results saved to {REF_PATH} | Number of references: {len(refs)}"
    )

    return refs


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
    extract_refs(args.report_root_dir, args.query_id, args.eval_system_name)
