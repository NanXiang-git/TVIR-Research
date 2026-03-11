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


def eval_figure_caption_quality(
    report_root_dir, query_id, eval_system_name, result_root_dir
):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/figure_caption_quality.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )
    UPDATE_REPORT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
        with open(UPDATE_REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
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
        with open(UPDATE_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"✅ No figures found, zero score saved to {EVAL_RESULT_PATH}")
        return

    evaluation_results = []
    total_scores = []

    system_prompt = """
You are an expert academic editor and visual analyst. Your task is to evaluate the **Quality of the Figure Caption** in a research report.

You may be provided with:
1. **Figure**: The visual content (such as a chart, graph, photo, diagram, etc.).
2. **Caption**: The full text associated with the figure.

SCORING DIMENSIONS (Each 1–10 points):

**1. Visual Accuracy**
*Does the caption correctly identify and describe what is actually shown?*

Focus:
- Alignment between caption and figure.
- Correct identification of subject, variables, and visible relationships.

Scoring:
- **9-10** – Perfect match. The caption correctly identifies what the figure is (e.g., chart/diagram/photo), names the key entities/components/variables, and accurately describes the visible content. All claims are consistent with the figure.
- **7-8** – High accuracy. The main identification and description are correct; minor visible elements are omitted or slightly under-specified, but nothing is wrong or misleading.
- **5-6** – Partial match. The caption is too generic or incomplete: it roughly matches the topic, but misses key visible elements needed to understand what is shown.
- **3-4** – Low accuracy. The caption refers to related but materially different content (e.g., wrong type of figure, wrong components/variables, or mischaracterized relationship/comparison).
- **1-2** – Contradiction. The caption describes content not present in the figure, or asserts relationships/structures/process steps that are clearly contradicted by what is shown.

**2. Minimum Necessary Information**
*Using only the figure and this caption, can a reader correctly interpret what is shown? Does the caption provide the **minimum necessary** information given the figure’s complexity (no less, no more)?*

Focus:
- If a reader can understand the figure’s main point using **the figure itself + this caption** (without the main text), the caption is sufficient.
- Provide only the **minimal guidance needed to read this figure**. **Do NOT require** the caption to restate or list details already visible in the figure.
- A caption can be short and still score high if the figure is self-explanatory. A longer caption scores high only if the added details are necessary to remove ambiguity for this specific figure.

Scoring:
- **9–10** – Minimally sufficient and perfectly calibrated. The caption provides just enough information for correct interpretation (given the figure), with little or no ambiguity, and without redundant listing or filler content.
- **7–8** – Sufficient with minor inefficiency. Interpretation is possible without the main text; one small clarification is missing **or** there is a small amount of unnecessary detail, but overall still clear.
- **5–6** – Borderline. The caption misses 1–2 key clarifications needed to interpret the figure **or** includes noticeable non-essential detail that does not improve interpretability; a reader may need the main text.
- **3–4** – Insufficient or bloated. Multiple necessary clarifications are missing **and/or** the caption is mostly verbose filler, leading to confusion or ambiguity.
- **1–2** – Not usable. The caption is too vague/empty to interpret the figure, or is largely irrelevant to what is shown, making independent interpretation impossible.

**3. Clarity & Readability**
*Is the caption written in clear and accessible language that is easy to understand?*

Focus:
- Clear sentence structure and word choice.
- Avoids unnecessary jargon or defines technical terms when necessary.
- Easy to parse and understand on first reading.

Scoring:
- **9-10** – Highly clear. The caption uses straightforward, direct language that is immediately understandable. Technical terms are either avoided or clearly defined. No ambiguous phrasing.
- **7-8** – Clear. Generally easy to understand with minor issues (e.g., one slightly complex sentence or undefined term that doesn't significantly hinder comprehension).
- **5-6** – Moderately clear. Contains some unclear phrasing, unnecessary complexity, or undefined jargon that requires re-reading or may confuse readers.
- **3-4** – Unclear. Multiple confusing sentences, heavy jargon without explanation, or overly complex language that obscures the main point.
- **1-2** – Incomprehensible. The caption is so poorly written, convoluted, or jargon-heavy that readers would struggle to understand what is being described.

IMPORTANT GUIDELINES:
- You must primarily compare the **Caption** against the **Visual Evidence** in the **Figure**.  
- Do **not** invent or assume facts that are not supported by the figure.

INSTRUCTIONS:
- **Step 1**: Review the provided **Figure** and **Caption**.
- **Step 2**: Evaluate the **Quality of the Figure Caption** across the three dimensions: **Visual Accuracy**, **Minimum Necessary Information**, and **Clarity & Readability**.
- **Step 3**: For each dimension, provide a **detailed justification** that examines concrete evidence from the figure and explains the reasoning leading to the score.
- **Step 4**: Based on the justification, assign a **1–10 score** for each dimension according to the detailed scoring criteria.
- **Step 5**: If any dimension scores **6 or lower**, provide a **revised caption** in the **suggestion** field, using only information supported by the figure. If **all** dimensions score **7 or higher**, set the **suggestion** field as an empty string "".

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

    for fig in tqdm(figures, desc="🤖 Evaluating figure caption quality"):
        caption = fig.get("caption", "")
        img_base64 = fig.get("img_base64", "")
        content = fig.get("content", "")

        if isinstance(img_base64, dict) and "error" in img_base64:
            evaluation_results.append(
                {"caption": caption, "error": img_base64["error"], "total_score": 0.0}
            )
            total_scores.append(0.0)
            continue

        user_prompt = f"""
Please evaluate the Quality of the Figure Caption based on the provided rubric.

<caption>
{caption}
</caption>

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

        revised_caption = evaluation_result.get("suggestion", "")
        if caption:
            scores = evaluation_result.get("scores", {})
            if scores:
                evaluation_result["total_score"] = sum(scores.values())
                total_scores.append(evaluation_result["total_score"])
                if any(score <= 6 for score in scores.values()):
                    report_content = report_content.replace(caption, revised_caption)
            else:
                evaluation_result["total_score"] = None
            evaluation_result["caption"] = caption
            evaluation_results.append(evaluation_result)
        else:
            if content:
                lines = report_content.split("\n")
                for i, line in enumerate(lines):
                    if content in line:
                        lines.insert(i + 1, f"**{revised_caption}**")
                        report_content = "\n".join(lines)
                        break
            evaluation_results.append(
                {"caption": caption, "error": "caption missing", "total_score": 0.0}
            )
            total_scores.append(0.0)

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
    with open(UPDATE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

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
    eval_figure_caption_quality(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
