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


def eval_multimodal_composition(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/multimodal_composition.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    figures = [fig for fig in figures if fig.get("type") != "unknown"]
    if not figures:
        result = {"normalized_score": 0.0}
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No figures found, zero score saved to {EVAL_RESULT_PATH}")
        return

    system_prompt = """
You are an expert evaluator specializing in auditing multi-modal research reports. Your task is to assess the **Multimodal Composition** of a model-generated **research_report**.

GOAL:  
Evaluate how effectively the **research_report** organizes and incorporates multimodal elements in its overall document design, with particular attention to the **layout, quantity, variety, and richness** of figures. Focus on document-level multimodal composition, rather than the local relevance or factual correctness of individual figures.

SCORING DIMENSIONS (Each 1–10 points)

**1. Multimodal Layout & Quantity Appropriateness**

*Are multimodal elements placed, distributed, and used in an appropriate amount across the report’s stages, supporting a smooth reading flow without obvious gaps or overload?*

Focus:  
- Figures are placed near the sections where they are discussed, not arbitrarily clustered or detached.  
- The overall visual distribution supports a natural reading rhythm, avoiding both long stretches with no visuals despite complex content and sudden dense clusters that disrupt the flow.  
- Important complex content (e.g., structures, workflows, data comparisons) has at least some visual support, rather than relying solely on dense text.  
- The total number of figures feels proportionate to the length and complexity of the report (neither too few nor excessive).  
- Multimodal elements are reasonably spread across key parts, rather than confined to only one section while other equally complex sections have none.

Scoring:
- **9-10 – Excellent layout and quantity appropriateness**: Multimodal elements are consistently placed near relevant text and are well distributed across the report’s stages. Coverage is well targeted (no clear gaps), and the quantity feels proportionate with no obvious redundancy. Overall, the layout and amount clearly support a smooth, coherent reading experience.
- **7-8 – Good layout and quantity with minor issues**: Placement and distribution are mostly sensible with occasional issues (a figure could be closer to its discussion, slight clustering, or a small coverage gap). Quantity is generally appropriate, with only minor under/over-use that does not strongly affect readability.
- **5-6 – Acceptable but uneven**: The report remains readable, but layout and/or quantity feel insufficiently planned. There may be noticeable stretches lacking visuals despite complexity, awkward groupings, uneven coverage across sections, or a quantity that is somewhat misaligned (a bit sparse or somewhat cluttered).
- **3-4 – Weak layout/quantity choices**: Figures are frequently detached from relevant text or awkwardly grouped; some key sections lack needed visuals while others feel cluttered. Coverage and quantity choices noticeably hinder reading flow and understanding.
- **1-2 – Very poor layout and quantity appropriateness**: Figures (if any) are placed with little regard to structure and discussion points, and the report is either severely under-illustrated despite complexity or overloaded with visuals. The multimodal design seriously disrupts or confuses the reading experience.

**2. Multimodal Variety & Richness**

*Does the report demonstrate appropriate variety and richness in its multimodal elements based on the report's topic and nature?*

Focus:
- **Source-type variety**:
  - **Retrieved external images**: e.g., architecture diagrams, system schematics, flowcharts, sequence diagrams, environment photos, illustrations, etc.
  - **Code-generated charts**: e.g., line charts, bar charts, scatter plots, heatmaps, radar charts, box plots, etc.
- **Intra-type variety**: Within each source type present, the report uses multiple appropriate sub-types.
  - **For charts**: Different chart types count as different sub-types. (e.g., line chart, bar chart, scatter plot are 3 distinct sub-types).
  - **For images**: Different purposes count as different sub-types (e.g., offshore wind power architecture diagram, reinforcement learning algorithm architecture diagram are 2 distinct sub-types).
- **Topic-appropriate multimodal strategy**: The choice and balance of multimodal elements should align with the report's subject matter.
  - **Data/analysis-heavy topics** (e.g., financial analysis, statistical studies, performance benchmarks): Naturally emphasize code-generated charts with rich variety.
  - **Technical/architectural topics** (e.g., system design, software architecture, infrastructure): Naturally emphasize retrieved images with rich variety.
  - **Balanced topics** (e.g., comprehensive surveys, product analyses): Should include both types with reasonable variety in each.

Hard Constraints:
1) You MUST compute these intermediate variables:
   - topic_category: string in {"data_heavy", "technical_architectural", "balanced"}
   - source_types_present: integer in {0,1,2}
   - image_subtypes_count: integer (0 if no valid images)
   - chart_subtypes_count: integer (0 if no valid charts)
   - total_subtypes: integer (image_subtypes_count + chart_subtypes_count)
   - dominant_subtypes_count: integer (chart_subtypes_count for data_heavy, image_subtypes_count for technical_architectural)

2) ABSOLUTE SCORING RULES - NO EXCEPTIONS:
   a) Once the numerical thresholds are met, the score range is LOCKED and cannot be changed.
   b) You MUST NOT apply any additional subjective criteria beyond the defined rules, including but not limited to:
      - Subjective quality judgments (e.g., "not especially diverse", "basic/common types")
      - Suggestions for additional content (e.g., "could include more advanced visualizations", "lacks X type")
      - Personal preferences about specific chart or image types
   c) The score range is determined EXCLUSIVELY by these objective factors:
      - The COUNT of subtypes (as computed above)
      - Whether both source types are present (for balanced topics only)
      - The topic category classification
   d) Within the determined score range (e.g., 9-10 or 7-8), you may select the specific score based on the minor execution quality differences. BUT you CANNOT move to a different score range under any circumstances.

3) You MUST determine the score by STRICTLY following the mapping below. The score ranges are **hard boundaries** that you CANNOT violate.

Score Mapping (EVALUATE IN ORDER, STOP AT FIRST MATCH):
- If source_types_present == 0:
    → score MUST be 1 or 2. STOP.

- If topic_category == "balanced":
    - If source_types_present == 2:
        - If image_subtypes_count >= 3 AND chart_subtypes_count >= 3:
            → score MUST be 9 or 10. STOP.
        - Else if image_subtypes_count >= 2 AND chart_subtypes_count >= 2:
            → score MUST be 7 or 8. STOP.
        - Else:
            → score MUST be 5 or 6. STOP.
    - If source_types_present == 1:
        - If max(image_subtypes_count, chart_subtypes_count) >= 3:
            → score MUST be 5 or 6. STOP.
        - Else:
            → score MUST be 3 or 4. STOP.

- If topic_category in {"data_heavy", "technical_architectural"}:
    - If dominant_subtypes_count >= 4:
        → score MUST be 9 or 10. STOP.
    - Else if dominant_subtypes_count >= 3:
        → score MUST be 7 or 8. STOP.
    - Else if total_subtypes >= 2:
        → score MUST be 5 or 6. STOP.
    - Else:
        → score MUST be 3 or 4. STOP.

IMPORTANT GUIDELINES:
- Only **actual, present multimodal elements** are considered valid for this evaluation.  
- Valid elements include embedded or clearly shown **figures**, or a **concrete, non-empty file path or URL** that clearly points to such an element.  
- The following do **not** count as valid multimodal content:  
  - Placeholders (e.g., “Figure 1: [to be inserted]”, “image here”).  
  - Any references without an actual figure or a concrete locator (e.g., “see figure below” but no figure, URL, or file path is provided).  
  - Any figures shown only as plain text or ASCII-style diagrams. 
- If the report contains **no valid multimodal elements** under these rules, reflect this in your scores (e.g., very weak layout, quantity and variety) and explicitly mention the absence of real multimodal content in your justification and suggestions.

INSTRUCTIONS:
- **Step 1**: Review the provided **Research_Report**.  
- **Step 2**: Evaluate the **Multimodal Composition** of the Research_Report across the two dimensions: **Multimodal Layout & Quantity Appropriateness**, and **Multimodal Variety & Richness**.  
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

    print("🤖 Evaluating multimodal composition of the research report...")

    user_prompt = f"""
Please evaluate the Multimodal Composition of the following research report based on the provided rubric.

<research_report>
{report_content}
</research_report>

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
            evaluation_results = _parse_json_result(result_json_str)

            if "scores" not in evaluation_results:
                raise ValueError(
                    f"Missing 'scores' field in response, result: {result_json_str}"
                )

            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(4**attempt)

    if last_err is not None:
        print(f"❌ Error during evaluation: {last_err}")
        evaluation_results = {"error": str(last_err)}

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
    eval_multimodal_composition(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
