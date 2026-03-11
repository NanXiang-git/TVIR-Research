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


def get_query_and_checklist(query_list, target_id):
    for item in query_list:
        if item.get("id") == target_id:
            return item.get("query", ""), item.get("checklist", {})
    return "", {}


def eval_analytical_depth_breadth(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/analytical_depth_breadth.json"
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

    system_prompt = """
You are an expert academic editor. Your task is to evaluate the **Analytical Depth & Breadth** of a model-generated **research_report**.

You will be given:  
1. **Original_Query**: The user’s original request or research question.  
2. **Research_Report**: The model-generated report responding to that query.

GOAL:  
Assess how deeply and broadly the **research_report** thinks about and analyzes the topic, beyond merely restating facts or copying surface-level content.

SCORING DIMENSIONS (Each 1–10 points)

**1. Causal Explanatory Reasoning**

*To what extent does the report go beyond description and provide clear, causal or mechanism-oriented explanations for its main claims?*

Focus:  
- Does the report move beyond simply stating “what happened” to explain “why” and “how it happens”?  
- Are there clear causal or mechanism links (e.g., linking causes, conditions, and outcomes), rather than a loose list of factors?  
- Does it specify key causal pathways and the conditions under which they hold, rather than implying causality?

Scoring:  
- **9-10 – Excellent causal explanatory reasoning**: The report consistently explains why and how key phenomena occur via explicit causal pathways and mechanisms, clarifying conditions/boundaries for major claims with minimal unexplained jumps.  
- **7-8 – Strong but slightly uneven reasoning**: Most important points include plausible causal/mechanism explanations, though some claims lack clear pathways or boundary conditions.  
- **5-6 – Moderate explanatory reasoning**: Some causal/mechanism explanation is present, but several key claims rely on generic “drivers” language without specifying how the mechanism operates or when it applies.  
- **3-4 – Limited or shallow reasoning**: Explanations are mostly vague (“due to”, “driven by”) with little mechanism detail or conditionality; causal links are often implied rather than articulated.  
- **1-2 – Very poor or purely descriptive**: The report is largely descriptive, with almost no attempts to explain why or how; causal/mechanism thinking is essentially absent.

**2. Analytical Depth & Development**

*To what extent does the report develop its key points into sustained analysis rather than brief, list-like statements?*

Focus:
- Are key claims unpacked with multi-step reasoning rather than asserted in a single step?
- Are the most important points developed into coherent, connected paragraphs instead of fragmented one-line bullets?
- Does the report prioritize depth on central points, with added length being substantive rather than repetitive?

Scoring:
- **9-10 – Excellent development**: Central claims are unpacked with clear multi-step reasoning; key sections show sustained, connected analysis with strong internal structure. Depth is concentrated on the most important points; added length is substantively analytical with minimal filler or repetition.
- **7-8 – Strong development with minor thin spots**: Most key points are developed beyond surface summary and include some multi-step reasoning. Structure is generally coherent, but a few sections remain somewhat compressed, list-like, or stop short of fully unpacking implications/logic.
- **5-6 – Mixed/uneven development**: The report alternates between some developed passages and many thin statements. Several important claims are only partially unpacked, with reasoning that is short, generic, or stops after one step. List-like presentation is common for core points.
- **3-4 – Shallow/list-like development**: Predominantly brief bullets or high-level summaries. Claims are often asserted with little to no reasoning, and important points lack sustained development or connected explanation.
- **1-2 – Very poor development**: Fragmentary and minimally elaborated. The report mostly consists of headings or one-liners with little analytical content and no meaningful development of key points.

**3. Critical Evaluation & Assumptions**

*To what extent does the report critically examine its assumptions, limitations, and alternative viewpoints instead of treating its own perspective as unquestioned?*

Focus:  
- Does the report identify and discuss key assumptions underlying its analysis, models, or narratives?  
- Does it acknowledge limitations in data, methods, or perspective, and consider how these might affect the reliability or scope of its conclusions?  
- Does it engage with relevant alternative explanations, interpretations, or strategies, rather than presenting a single viewpoint as obviously correct?

Scoring:
- **9-10 – Excellent critical evaluation**: The report explicitly surfaces major assumptions and examines their plausibility. It discusses important limitations of data and methods and thoughtfully considers how these constrain the conclusions. Alternative viewpoints or explanations are actively engaged and weighed, showing strong self-critical awareness.  
- **7-8 – Solid but incomplete critical reflection**: The report shows clear critical thinking: it notes some key assumptions, limitations, and alternative perspectives. However, not all important issues are explored in depth, and parts of the analysis still rely on relatively unexamined premises.  
- **5-6 – Moderate critical evaluation**: There is some acknowledgement of assumptions or limitations, and the report may briefly mention alternative views. However, this is sporadic and not integrated into the core reasoning; most arguments still proceed as if their premises are mostly unquestioned.  
- **3-4 – Limited critical reflection**: Assumptions and limitations are largely implicit or mentioned only superficially. Alternative explanations or viewpoints are barely considered, and the report mostly treats its own framing and data as straightforward and unproblematic.  
- **1-2 – Very poor or absent critical evaluation**: The report presents its analysis as if it is fully self-evident, with virtually no discussion of assumptions, limitations, or alternatives. It shows little awareness of potential biases, blind spots, or uncertainties.

**4. Actionable Forward-Looking Insight**

*Does the report translate its analysis into meaningful, forward-looking implications or recommendations that help the reader understand what to do or watch for next?*

Focus:
- Does the report derive concrete implications for action, decision-making, strategy, policy, or further work from its analysis?  
- Are recommendations or takeaways specific, directional, and conditional (where appropriate), rather than vague or purely formulaic statements?  
- Does the report consider future scenarios, risks, opportunities, or monitoring points, instead of stopping at a description of the current state?

Scoring:
- **9-10 – Excellent actionable and forward-looking insight**: The report offers clear, specific, and well-justified implications or recommendations that follow from the analysis. It highlights relevant future scenarios, risks, or opportunities and gives the reader a strong sense of “what to do or pay attention to next.”  
- **7-8 – Strong but somewhat limited insight**: There are meaningful, reasonably specific implications or recommendations, and some forward-looking reflection. However, coverage is not fully systematic: certain conclusions are left without actionable follow-through or future-oriented consideration.  
- **5-6 – Moderate actionable insight**: The report contains some useful takeaways, but they are either unevenly developed, somewhat generic, or only loosely tied to the preceding analysis. Forward-looking elements are present but not deeply integrated.  
- **3-4 – Limited or generic implications**: Implications and recommendations, if present, are mostly high-level, vague, or boilerplate with little specificity or prioritization. There is little sense of future scenarios or concrete next steps.  
- **1-2 – Very poor or absent forward-looking content**: The report mostly describes or analyzes the current situation with almost no attempt to derive practical implications or future-oriented insights.

**5. Thematic Breadth & Coverage**

*How comprehensively does the report cover the key aspects of its topic, showing a sufficiently broad and well-chosen thematic scope without obvious blind spots?*

Focus:  
- Does the report address the main dimensions that are clearly relevant to the core topic, rather than focusing narrowly on a single aspect?  
- Does it reasonably extend beyond only the explicitly mentioned points in the original_query to include other clearly important, closely related angles?  
- While remaining focused, does the report avoid obvious gaps where a major, relevant facet of the topic is missing or severely underdeveloped?

Scoring:  
- **9-10 – Excellent thematic breadth and coverage**: The report covers the topic in a well-balanced, comprehensive way, addressing the major relevant dimensions with appropriate depth. It not only responds to the explicitly stated points in the original_query but also proactively includes other clearly important, closely related angles. This extension beyond the original_query feels natural and necessary rather than off-topic. There are no clear blind spots.  
- **7-8 – Strong but slightly uneven coverage**: Most important aspects of the topic are covered, and the scope feels suitably broad. The report does move beyond the exact wording of the original_query to touch on some additional, relevant facets, but one or two significant dimensions are treated briefly or somewhat underdeveloped. The overall breadth is good, with only minor gaps.  
- **5-6 – Moderate thematic breadth**: The report addresses several key facets of the topic, but coverage is selective or uneven. It may mostly stick to the points explicitly mentioned in the original_query, with only limited extension to other relevant angles, or it may treat some dimensions in depth while leaving others only hinted at. Overall breadth is acceptable but leaves noticeable room for a more complete picture.  
- **3-4 – Narrow or imbalanced coverage**: The report focuses heavily on a limited subset of relevant aspects and largely stays within the most obvious or explicitly stated parts of the original_query. It rarely brings in additional, closely related angles that would help complete the thematic picture. As a result, the scope feels constrained or skewed, and important dimensions of the topic are missing or severely underdeveloped.  
- **1-2 – Very poor or severely narrow coverage**: The report’s treatment of the topic is extremely narrow, fragmentary, or misaligned. It may restrict itself almost entirely to a small portion of what the original_query mentions, ignoring other clearly relevant aspects, and it does not attempt to go beyond the original_query’s explicit wording in any meaningful way. The reader is left without a coherent overall understanding of the subject.

INSTRUCTIONS:
- **Step 1**: Review the provided **Original_Query** and **Research_Report**.  
- **Step 2**: Evaluate the **Analytical Depth & Breadth** of the Research_Report across the five dimensions: **Causal Explanatory Reasoning**, **Analytical Depth & Development**, **Critical Evaluation & Assumptions**, **Actionable Forward-Looking Insight**, and **Thematic Breadth & Coverage**.  
- **Step 3**: For each dimension, provide a **detailed justification** that examines concrete evidence from the report and explains the reasoning leading to the score.
- **Step 4**: Based on the justification, assign a **1–10 score** for each dimension according to the detailed scoring criteria.
- **Step 5**: If any dimension scores **6 or lower**, use the **suggestion** field to propose concrete improvements. If **all** dimensions score **7 or higher**, set the **suggestion** field as an empty string "".

OUTPUT FORMAT:  
Respond with a JSON object. Do not add markdown code blocks (like ```json) around the output.
{
  "justification": {
    "Scoring Dimension": "specific reasons for the score...",
    ...
  },
  "scores": {
    "Scoring Dimension": 8,
    ...
  },
  "suggestion": "..."
}
"""

    print("🤖 Evaluating analytical depth and breadth of the research report...")

    user_prompt = f"""
Please evaluate the Analytical Depth & Breadth of the following research report based on the provided rubric.

<original_query>
{original_query}
</original_query>

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

            if "scores" not in evaluation_results:
                raise ValueError(
                    f"Missing 'scores' field in response, result: {result_json_str}"
                )

            last_error = None
            break
        except Exception as e:
            last_error = e
            time.sleep(4**attempt)

    if last_error is not None:
        print(f"❌ Error during evaluation: {last_error}")
        evaluation_results = {"error": str(last_error)}

    scores = evaluation_results.get("scores", {})
    if scores:
        score_list = list(scores.values())
        normalized_score = sum(score_list) / (len(score_list) * 10)
        evaluation_results["normalized_score"] = round(normalized_score, 4)
    else:
        evaluation_results["normalized_score"] = None

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} | Normalized score: {evaluation_results['normalized_score']}"
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
    eval_analytical_depth_breadth(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
