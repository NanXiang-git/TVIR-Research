import json
import os
from openai import OpenAI
import argparse
from tqdm import tqdm
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


def eval_citation_support(report_root_dir, query_id, eval_system_name, result_root_dir):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/citation_support.json"
    REF_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/refs_with_web_content.json"
    )
    CITATION_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/citations_dedup.json"
    )

    try:
        with open(REF_PATH, "r", encoding="utf-8") as f:
            refs = json.load(f)
        with open(CITATION_PATH, "r", encoding="utf-8") as f:
            citations = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    ref_map = {item["ref_idx"]: item["web_content"] for item in refs}

    if not citations:
        result = {
            "normalized_score": 0.0,
            "item_count": 0,
            "scored_item_count": 0,
            "evaluation_results": [],
        }
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No citations found, zero score saved to {EVAL_RESULT_PATH}")
        return

    evaluation_results = []

    system_prompt = """
You are an expert evaluator specializing in fact verification against reference materials. Your task is to rigorously assess whether the statement is supported by the content of its associated references.

JUDGMENT CRITERIA:
- First, determine whether the references contain any valid information. If they are empty, show “page not found,” or otherwise lack substantive content, the statement should be marked as "unknown".
- If the references are valid:
    - If a statement's facts or data can be fully found in the references, mark it as "supported" (data can be rounded).
    - If only part of the facts or data in the statement can be found in the references, mark it as "partially_supported".
    - If none of the facts or data in the statement can be found in the references, mark it as "unsupported".
- Only judge based on the actual content provided.

OUTPUT FORMAT:
Respond with a JSON object. Do not add markdown code blocks (like ```json) around the output.
{
  "result": "supported",
  "justification": "Clearly explain which parts of the statement are supported, partially supported, or unsupported, and cite direct evidence from the references."
}
"""

    for item in tqdm(citations, desc="🤖 Evaluating citation support"):
        fact = item.get("fact", "")
        ref_idxs = item.get("ref_idxs", [])
        references_payload = []
        for rid in ref_idxs:
            references_payload.append(
                {
                    "ref_idx": rid,
                    "content": ref_map.get(rid, ""),
                }
            )

        user_prompt = f"""
Please judge whether the following statement is supported by the references according to the criteria.

<statement>
{fact}
</statement>

<references_json>
{json.dumps(references_payload, ensure_ascii=False)}
</references_json>

Remember to output ONLY the JSON object.
"""

        last_err = None
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
                evaluation_result = _parse_json_result(result_json_str)

                if "result" not in evaluation_result:
                    raise ValueError(
                        f"Missing 'result' field in response, result: {result_json_str}"
                    )

                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(4**attempt)

        if last_err is not None:
            print(f"❌ Error during evaluation: {last_err}")
            evaluation_results.append(
                {"ref_idxs": ref_idxs, "fact": fact, "error": str(last_err)}
            )
            continue

        evaluation_result["ref_idxs"] = ref_idxs
        evaluation_result["fact"] = fact
        evaluation_results.append(evaluation_result)

    total_score = []
    for item in evaluation_results:
        result = item.get("result", "")
        if result:
            if result == "supported":
                total_score.append(1)
            elif result == "partially_supported":
                total_score.append(0.5)
            else:
                total_score.append(0)

    normalized_score = (
        round(sum(total_score) / len(total_score), 4) if total_score else None
    )

    result = {
        "normalized_score": normalized_score,
        "item_count": len(citations),
        "scored_item_count": len(total_score),
        "evaluation_results": evaluation_results,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} ｜ Normalized Score: {normalized_score} ｜ Scored Items: {len(total_score)}/{len(citations)}"
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
    eval_citation_support(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
