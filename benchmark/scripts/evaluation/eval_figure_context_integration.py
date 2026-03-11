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


def eval_figure_context_integration(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/figure_context_integration.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    if not figures:
        result = {
            "normalized_score": 0.0,
            "item_count": 0,
            "scored_item_count": 0,
            "evaluation_results": [],
        }
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No figures found, zero score saved to {EVAL_RESULT_PATH}")
        return

    total_scores = []
    evaluation_results = []

    system_prompt = """
You are an expert academic editor and visual analyst. Your task is to evaluate how well a **Figure** is integrated with its surrounding **Context** in a research report.

You will be given:
1. **Figure**: The visual content (such as a chart, graph, photo, diagram, etc.).
2. **Context**: The paragraph(s) in the report that reference or surround this figure. The placeholder <figure> indicates where the figure appears in the text.

GOAL:
Judge the **relationship between the Figure and the Context**.

SCORING DIMENSIONS (Each 1–10 points):

**1. Contextual Relevance**

*How well does the figure's content match the specific topic and claims in the context?*

Focus:
- **Semantic alignment** between what the figure shows and what the context discusses.
- Whether the figure depicts the same topic, variables, phenomena, or cases mentioned in the surrounding text.

Scoring:
- **9-10 – Highly Relevant**: The figure directly depicts or represents the specific subject matter discussed in the immediate context. A reader can clearly see why this particular figure was chosen for this passage.
- **7-8 – Relevant**: The figure clearly depicts the specific subtopic or subject in this passage, though it may not capture every detail mentioned in the text. The figure choice is appropriate for this context.
- **5-6 – Moderately Relevant**: The figure belongs to the same broader theme or research area and is not out of place, but the connection between what the figure shows and what this passage specifically discusses is loose.
- **3-4 – Weak Relevance**: The connection to the context is vague or forced. It only shares a very general domain, without depicting what this passage is specifically about.
- **1-2 – Irrelevant**: The figure does not depict anything meaningfully related to the local context; a typical reader would be confused why this particular figure is placed here.

**2. Narrative Coherence**

*How smoothly is the figure integrated into the flow of the text?*

Focus:
- Whether the context **explicitly or clearly implicitly** points the reader to this figure.
- How natural the transition is between text and the figure, and whether the discussion around it is sufficient.  

Scoring:
- **9-10 – Seamless Integration**: The text explicitly references the figure (e.g., “As shown in Figure 3…”), and it meaningfully discusses key elements of the figure, so the reader is guided to look at and interpret it.
- **7-8 – Good Integration**: There is a clear reference to the figure, but the transition is slightly abrupt, or the discussion is a bit brief or incomplete.
- **5-6 – Weak Link**: The figure is mentioned (e.g., only “(see Figure 2)” at the end of a sentence), but the surrounding text does not really explain or interpret it.
- **3-4 – Disconnected**: No explicit reference; the figure appears between paragraphs or at the side, and the relation has to be guessed by the reader.
- **1-2 – Disruptive**: The placement of the figure breaks the reading flow, feels random, or even contradicts the narrative sequence.

**3. Visual Information Value**

*Does the figure provide visual information that text alone cannot effectively convey?*

Focus:
- Whether the figure offers **unique visual value** that would be difficult, inefficient, or impossible to communicate through text alone.
- Consider how much understanding would be lost if the figure were removed? Would readers struggle to grasp the information from the text alone?

Scoring:
- **9-10 – Visually Essential**: The figure conveys information that is extremely difficult or impractical to describe in text. Without the visual, readers would struggle to grasp the information even with lengthy description.
- **7-8 – High Visual Value**: The figure significantly enhances understanding by providing visual information that would require extensive text to approximate. Text alone would be notably less efficient or clear.
- **5-6 – Moderate Visual Value**: The figure provides some visual convenience, but most information could be reasonably conveyed in a few sentences. The visual format is helpful but not particularly necessary.
- **3-4 – Low Visual Value**: The figure adds minimal visual information beyond what text easily provides. The visual medium is barely justified.
- **1-2 – No Visual Necessity**: The figure provides no meaningful visual information that text cannot easily convey. Using a visual medium here is unnecessary regardless of content relevance.

INSTRUCTIONS:
- **Step 1**: Review the provided **Figure** and **Context**.
- **Step 2**: Evaluate the **Integration between the Figure and the Context** across the three dimensions: **Contextual Relevance**, **Narrative Coherence**, and **Visual Information Value**.
- **Step 3**: For each dimension, provide a **detailed justification** that examines concrete evidence from the figure and context, **then** explains the reasoning leading to the score.
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

    for fig in tqdm(figures, desc="🤖 Evaluating figure-context integration"):
        caption = fig.get("caption", "")
        img_base64 = fig.get("img_base64", "")
        context = fig.get("context", "")

        if isinstance(img_base64, dict) and "error" in img_base64:
            evaluation_results.append(
                {"caption": caption, "error": img_base64["error"], "total_score": 0.0}
            )
            total_scores.append(0.0)
            continue

        if not context:
            evaluation_results.append(
                {"caption": caption, "error": "context missing", "total_score": 0.0}
            )
            total_scores.append(0.0)
            continue

        user_prompt = f"""
Please evaluate the integration between the figure and the context based on the provided rubric.

<context>
{context}
</context>

Remember to output ONLY the JSON object.
"""

        user_message_content = [
            {
                "type": "text",
                "text": user_prompt,
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

                if "scores" not in evaluation_result:
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
        normalized_score = sum(total_scores) / (len(total_scores) * 10 * 3)
        normalized_score = round(normalized_score, 4)
    else:
        normalized_score = None

    result = {
        "normalized_score": normalized_score,
        "item_count": len(figures),
        "scored_item_count": len(total_scores),
        "evaluation_details": evaluation_results,
    }

    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} | Normalized score: {normalized_score} | Scored items: {len(total_scores)}/{len(figures)}"
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
    eval_figure_context_integration(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
