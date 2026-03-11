import json
import os
from openai import OpenAI
import argparse
from dotenv import load_dotenv
import time
from tqdm import tqdm

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


def dedup_citations(report_root_dir, query_id, eval_system_name):
    CITATION_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/citations.json"
    OUTPUT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/citations_dedup.json"
    )
    try:
        with open(CITATION_PATH, "r", encoding="utf-8") as f:
            citations = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return {"error": "Citation file not found"}

    if not citations:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"✅ No citations found, empty file saved to {OUTPUT_PATH}")
        return []

    system_prompt = """
You are an expert academic information processor. Your task is to rigorously determine whether two statements, both citing the same references, are semantically redundant (i.e., express the same fact in different words or with minor differences).

DEDUPLICATION GUIDELINES:
- If two statements convey the same core fact, even with different wording, they are considered duplicates.
- If there are substantive differences in meaning, data, scope, or context, they are not duplicates.
- Focus strictly on the factual content, not on stylistic or minor wording differences.

OUTPUT FORMAT: 
Respond ONLY with a JSON object: {"dedup": true} if they are duplicates, {"dedup": false} otherwise.
"""

    dedup_results = []
    n = len(citations)

    total_comparisons = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if citations[i]["ref_idxs"] == citations[j]["ref_idxs"]
    )

    with tqdm(total=total_comparisons, desc="⏳ Deduplicating citations") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                s1 = citations[i]["fact"]
                s2 = citations[j]["fact"]
                ref_idxs = citations[i]["ref_idxs"]
                if ref_idxs != citations[j]["ref_idxs"]:
                    continue
                user_prompt = f"""
        Compare the following two statements, both citing reference [{ref_idxs}]. Are they semantically redundant according to the guidelines?
        Statement 1: {s1}
        Statement 2: {s2}

        Remember to output ONLY the JSON object, without any extra explanation or markdown code block.
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
                        result = json.loads(result_json_str)

                        if "dedup" not in result:
                            raise ValueError(
                                f"Missing 'dedup' field in response, result: {result_json_str}"
                            )

                        if result.get("dedup") is True:
                            dedup_results.append((i, j))

                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        time.sleep(4**attempt)

                if last_error is not None:
                    print(f"❌ Error during citation deduplication: {last_error}")

                pbar.update(1)

    keep = [True] * n
    for i, j in dedup_results:
        keep[j] = False

    deduped = [citations[idx] for idx, flag in enumerate(keep) if flag]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"✅ Citation deduplication completed, results saved to {OUTPUT_PATH}")

    return deduped


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
    dedup_citations(args.report_root_dir, args.query_id, args.eval_system_name)
