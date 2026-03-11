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


def extract_citations(report_root_dir, query_id, eval_system_name):
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report.md"
    CITATION_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/citations.json"

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return {"error": "Report file not found"}

    system_prompt = """
You are an expert academic information extractor. Your task is to rigorously and comprehensively extract all citation facts from the research_report body, in the form of (fact, ref_idx) pairs.

EXTRACTION GUIDELINES:
**1. Scope**  
- Only extract citation facts that appear in the **main text body** and have **explicit citation markers** (e.g. “[3]”, “[4,5]”).  
- Ignore:
  - Citations related only to **figures** (e.g. in figure captions or source notes around figures).
  - Any **unmarked** citations (mentions of prior work without explicit numbered markers).
**2. Fact**
- Must be complete, understandable sentences or paragraphs containing the citation markers. Expand context if needed for clarity.
- When one or more citations are associated with a table:
  - If the citations apply to the **entire table** (e.g., in the table caption or source notes around the table), the fact must contain the **entire table content**.
  - If the citations apply to **only part of the table** (e.g., a specific row, column, or cell), then:
    - Create a separate fact for **each such cited part**.
    - Each fact must describe **only that specific part** (the corresponding row/column/cell content), **not the whole table**.
**3. ref_idx**
- Use the citation numbers exactly as they appear in the text.
- If a fact cites multiple references, include all cited numbers in ref_idxs.

OUTPUT FORMAT: 
Respond with a JSON list. Each item:
{
  "fact": "Text fragment from the report body containing the citation marker(s)",
  "ref_idxs": ["1", "2"]
}
"""

    print("⏳ Extracting citations...")

    user_prompt = f"""
Please extract all citation facts from the following research_report according to the guidelines.

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
            citations = json.loads(result_json_str)
            if isinstance(citations, dict) and "error" in citations:
                raise ValueError(f"Response contains error: {citations['error']}")
            while isinstance(citations, dict) and len(citations) == 1:
                citations = next(iter(citations.values()))

            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(4**attempt)

    if last_error is not None:
        print(f"❌ Error during citation extraction: {last_error}")
        citations = {"error": str(last_error)}

    os.makedirs(os.path.dirname(CITATION_PATH), exist_ok=True)
    with open(CITATION_PATH, "w", encoding="utf-8") as f:
        json.dump(citations, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Citation extraction completed, results saved to {CITATION_PATH} | Number of citations: {len(citations)}"
    )

    return citations


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
    extract_citations(args.report_root_dir, args.query_id, args.eval_system_name)
