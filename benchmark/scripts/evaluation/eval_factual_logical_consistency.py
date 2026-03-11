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


def eval_factual_logical_consistency(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/factual_logical_consistency.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    system_prompt = """
You are an expert academic editor. Your task is to evaluate the **Factual & Logical Consistency** of a model-generated **research_report**.

GOAL:  
Determine whether the report **contains factual or logical contradictions**, and count the total number of distinct contradictions.

OUT OF SCOPE:
1. **Citation and reference issues**: Do NOT evaluate any problems related to citations or references. This includes inconsistencies in attribution, source identification, or provenance of methods/concepts/data that arise from how citation markers are used or organized. **If a contradiction exists only because of citation or reference-list issues, it is OUT OF SCOPE.** Only identify contradictions that would still exist if all citation markers and reference entries were removed from the report.
2. **External factual accuracy**: Do NOT evaluate whether the report’s statements are true in the real world; only check whether the report contradicts itself internally.  
3. **Terminology consistency**: Do NOT evaluate whether key terms, labels, or names for the same concept, variable, or method are used consistently throughout the report; only when such terminological differences lead to a direct contradiction about the same underlying object or result should it be counted as an issue.

CONTRADICTION DEFINITION:  

An **contradiction (issue)** exists when two or more statements in the report, taken at face value within the report’s own context, **cannot all be true at the same time**, and the report does not provide a clear explanation, condition, or update that resolves this conflict.

You must consider both:

1. **Factual contradictions**
Examples include, but are not limited to:
- The same quantity (e.g., sample size, number of companies, growth rate, percentage, etc.) is given conflicting values with no explanation (e.g., no mention of different subsamples, filtering, or time points).  
- The same time period, region/market, or population/scope is described differently in different places for what is clearly the same analysis or dataset.  
- The same dataset’s source or status (complete vs missing/bias, etc.) is described in mutually conflicting ways for the same stage of analysis.  
- The same result (same metric, same time period) is described with opposite trends or directions in different parts of the text (e.g., “increasing” vs “decreasing”) without clarification.

2. **Logical contradictions**
Examples include, but are not limited to:
- The report states an explicit premise, assumption, or applicability condition, and later reasoning or conclusions clearly violate that premise under the same context.  
- The report gives a certain judgment/evaluation of a model, strategy, or view in one place and later uses an opposite characterization of the same item in the same context, without new evidence or conditions.  
- The conclusions or recommendations directly state the opposite of the report’s own presented results (e.g., results show “unlikely to succeed,” conclusion claims “almost certain to succeed”).  
- The report explicitly labels its results as non-generalizable or restricted, but later presents them as broadly applicable without any additional justification.

COUNTING RULES:
- Each distinct contradiction (factual or logical) counts as **one** issue.  
- Repeated restatements of the same underlying contradiction count as **one** issue.  
- If multiple statements all conflict on the same fact or conclusion, treat that as **one** issue.  

INSTRUCTIONS:  
- **Step 1**: Read the entire **Research_Report**.  
- **Step 2**: Identify all contradictions where the report’s own statements cannot all be true at the same time and no clear explanation is given.  
- **Step 3**: For each identified issue:  
  - Provide a brief description of the contradiction.  
  - Provide textual evidence: quote or closely paraphrase the relevant conflicting statements with location references (section/paragraph labels if available, or your own brief locator such as “Introduction, paragraph 2”).  

REMEMBER: 
Be as strict as possible in identifying contradictions. Scrutinize the report thoroughly to find any potential logical or factual inconsistencies.

OUTPUT FORMAT:  
Please respond with a JSON object. Do not add markdown code blocks (like ```json) around the output.
{
  "total_issues": 4,
  "issues": [
    {  
      "id": 1,
      "description": "brief summary of the contradiction",
      "evidence": {
        "section/paragraph reference": "excerpt or close paraphrase",
        ...
      }
    },
    ...
  ]
}
"""

    print("🤖 Evaluating factual and logical consistency of the research report...")

    user_prompt = f"""
Please evaluate the **Factual & Logical Consistency** of the following research report based on the provided rubric.

<research_report>
{report_content}
</research_report>

Remember to output ONLY the JSON object.
"""

    last_err = None
    for attempt in range(1):
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

            if "total_issues" not in evaluation_results:
                raise ValueError(
                    f"Missing 'total_issues' field in response, result: {result_json_str}"
                )

            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(4**attempt)

    if last_err is not None:
        print(f"❌ Error during evaluation: {last_err}")
        evaluation_results = {"error": str(last_err)}

    N_issues = evaluation_results.get("total_issues", None)
    if N_issues == 0:
        score = 10
    elif 1 <= N_issues <= 2:
        score = 9
    elif 3 <= N_issues <= 4:
        score = 8
    elif 5 <= N_issues <= 6:
        score = 7
    elif 7 <= N_issues <= 8:
        score = 6
    elif 9 <= N_issues <= 10:
        score = 5
    elif 11 <= N_issues <= 12:
        score = 4
    elif 13 <= N_issues <= 14:
        score = 3
    elif 15 <= N_issues <= 17:
        score = 2
    elif N_issues >= 18:
        score = 1
    else:
        score = None

    evaluation_results["normalized_score"] = score / 10 if score is not None else None

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation complete, results saved to {EVAL_RESULT_PATH} ｜ Normalized Score: {evaluation_results['normalized_score']}"
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
    eval_factual_logical_consistency(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
