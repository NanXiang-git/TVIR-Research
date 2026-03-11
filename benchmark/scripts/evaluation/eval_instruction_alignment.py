import json
import os
from openai import OpenAI
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


def count_items_in_checklist(checklist):
    total = 0
    for v in checklist.values():
        if isinstance(v, list):
            total += len(v)
    return total


def get_query_and_checklist(query_list, target_id):
    for item in query_list:
        if item.get("id") == target_id:
            return item.get("query", ""), item.get("checklist", {})
    return "", {}


def eval_instruction_alignment(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/instruction_alignment.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"
    QUERY_PATH = "data/query.json"

    try:
        with open(QUERY_PATH, "r", encoding="utf-8") as f:
            query_list = json.load(f)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    original_query, checklist = get_query_and_checklist(query_list, query_id)
    if not original_query:
        print(f"❌ Error: In {QUERY_PATH} no entry found for query ID {query_id}")
        return
    if not checklist:
        print(f"❌ Error: In query ID {query_id} no checklist found")
        return

    system_prompt = """
You are an expert evaluator specializing in auditing multi-modal deep research reports. Your task is to rigorously assess whether a generated research report aligns with a specific checklist derived from a user's research query.

GOAL:
Determine if the report **fully, accurately, and deeply** addresses each item in the checklist.

EVALUATION SCORING CRITERIA:
- **Score 1 (Pass)**: The report provides a complete, specific, and substantial answer to the checklist item. All requested data, analysis, formats, and timeframes are met.
- **Score 0.5 (Partial)**: The report addresses the checklist item but is missing some required elements, is somewhat vague, only partially meets constraints, or provides incomplete/underspecified evidence. The item is *partially* satisfied.
- **Score 0 (Fail)**: The report fails to provide the required information, is largely vague, misses key constraints, or is missing entirely.

IMPORTANT GUIDELINES:
**1. Explicit Inclusion of Multimodal Elements**
- Valid multimodal elements include embedded or clearly shown **figures**, or a **concrete, non-empty file path or URL** that clearly points to such an element.  
- The following do **not** qualify as valid multimodal content:  
  - Placeholders (e.g., “Figure 1: [to be inserted]”, “image here”).  
  - Any references without an actual figure or a concrete locator (e.g., “see figure below” but no figure, URL, or file path is provided).  
  - Any figures shown only as plain text or ASCII-style diagrams. 
**2. Focus on Substantive Data Delivery**
- The evaluation must be based strictly on the actual, concrete content delivered in the report; scoring is only permitted when the report explicitly includes substantive content fulfilling the required information.
- The following do **not** qualify as satisfying an item:
  - Methodology-only text (plans, approaches, or descriptions of how data might be obtained) without delivered results.
  - Statements indicating the required information is missing, unavailable, not found, or not provided (e.g., “information not provided,” “no data found”).

INSTRUCTIONS:
- **Step 1**: For each checklist item, locate the directly relevant section in the "Research Report”.
- **Step 2**: Verify whether the identified content includes all the specific information required by that item.
- **Step 3**: Provide a score from {0, 0.5, 1}.
- **Step 4**: Write a clear justification that explicitly states which required elements have been provided, or specifically identifies what is missing or non-compliant, citing direct excerpts from the report as evidence.

OUTPUT FORMAT:
Respond with a JSON object mirroring the input structure. Do not add markdown code blocks (like ```json) around the output.
{
  "Section Name": {
    "Checklist Item Text": { "score": 0, "justification": "..." },
    ...
  }
}
"""

    print("🤖 Evaluating instruction alignment of the research report...")

    checklist_str = json.dumps(checklist, ensure_ascii=False, indent=2)

    user_prompt = f"""
Please evaluate the instruction alignment of the following research report based on the provided checklist.

<original_query>
{original_query}
</original_query>

<checklist>
{checklist_str}
</checklist>

<research_report>
{report_content}
</research_report>

Remember to output ONLY the JSON object.
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
                response_format={"type": "json_object"},
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

            if result_json_str.strip().startswith("```json"):
                result_json_str = result_json_str.strip()[7:]
            if result_json_str.strip().endswith("```"):
                result_json_str = result_json_str.strip()[:-3]
            evaluation_results = _parse_json_result(result_json_str)

            if isinstance(evaluation_results, dict) and "error" in evaluation_results:
                raise ValueError(
                    f"Response contains error: {evaluation_results['error']}"
                )

            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(4**attempt)

    if last_error is not None:
        print(f"❌ Error during evaluation: {last_error}")
        evaluation_results = {"error": str(last_error)}

    total_score = []
    if isinstance(evaluation_results, dict):
        for section in evaluation_results.values():
            if isinstance(section, dict):
                for item in section.values():
                    score = item.get("score")
                    if score is not None:
                        total_score.append(score)
    normalized_score = (
        round(sum(total_score) / len(total_score), 4) if total_score else None
    )

    result = {
        "normalized_score": normalized_score,
        "item_count": count_items_in_checklist(checklist),
        "scored_item_count": len(total_score),
        "evaluation_details": evaluation_results,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} | Normalized score: {normalized_score} | Scored items: {len(total_score)}/{count_items_in_checklist(checklist)}"
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
    eval_instruction_alignment(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
