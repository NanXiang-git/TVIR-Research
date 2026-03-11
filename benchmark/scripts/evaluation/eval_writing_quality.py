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


def eval_writing_quality(report_root_dir, query_id, eval_system_name, result_root_dir):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/writing_quality.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    system_prompt = """
You are an expert academic editor. Your task is to evaluate the **Writing Quality** of a model-generated **research_report**.

GOAL:  
Evaluate how well the **research_report** functions as academic/professional prose in the style of a research report, focusing on writing quality rather than factual correctness.

SCORING DIMENSIONS (Each 1–10 points)

**1. Coherence & Organization**

*Does the research_report have a logical, coherent flow at the document and paragraph level, with well-structured sections and smooth transitions?*

Focus: 
- Overall structure: clear sections and a reasonable order.  
- Paragraph-level coherence: each paragraph has a clear main idea; sentences within a paragraph follow a logical progression.  
- Transitions between sections/paragraphs: the reader is guided smoothly from one topic to the next, without abrupt or unexplained jumps.

Scoring:
- **9-10 – Excellent coherence and organization**: The report’s structure is clear and well-designed; sections and paragraphs are logically ordered. Transitions are smooth, the progression of ideas is easy to follow, and the overall reading experience feels cohesive from start to finish.
- **7-8 – Good coherence**: The overall structure is clear and sensible. There may be minor rough transitions or slightly uneven sections, but the progression of topics or ideas is consistently understandable and largely well ordered.
- **5-6 – Acceptable but uneven**: A recognizable structure exists, but some paragraphs or sections feel disjointed, misplaced, or weakly connected. Transitions are occasionally abrupt, and the reader sometimes has to work to reconstruct how one part leads to the next.
- **3-4 – Weak coherence**: Organization is confusing or inconsistent. Sections or paragraphs appear in a suboptimal order; topic shifts are abrupt; transitions are often missing. The overall flow is unclear and hard to follow.
- **1-2 – Poorly organized**: The report is largely incoherent in its sequencing. Topics are mixed together without clear structure, transitions are absent, and the reader struggles to understand how the text is organized or how different parts relate to each other.

**2. Clarity & Readability**

*Is the research_report written in clear, readable language, avoiding unnecessary complexity and ambiguity?*

Focus:  
- Sentence-level clarity: proper grammar, clear syntax, and appropriate vocabulary.  
- Avoidance of overly convoluted, ambiguous, or unnecessarily complex sentences.  
- Technical terms and specialized concepts are used appropriately and explained or contextualized when necessary.

Scoring:
- **9-10 – Very clear and readable**: Sentences are well-constructed and easy to understand. Technical terms are used appropriately and explained when necessary. The text is pleasant to read and accessible to the intended audience.
- **7-8 – Generally clear**: The text is understandable with only minor awkward phrasing or occasional dense passages. Overall readability is good.
- **5-6 – Mixed clarity**: The main ideas can be understood, but there are noticeable awkward sentences, occasional ambiguities, or sections that are harder to parse. The reader needs extra effort at times.
- **3-4 – Hard to read**: Frequent grammar or syntax issues, long and tangled sentences, or confusing wording. Clarity problems significantly hinder comprehension.
- **1-2 – Very unclear**: Serious language problems or extremely convoluted writing make the report difficult to understand or even follow.

**3. Conciseness & Redundancy**

*Does the research_report avoid unnecessary repetition and filler, expressing ideas as succinctly as is reasonable for the task defined by the query?*

Focus:  
- Presence of repeated arguments, sentences, or paragraphs without added value.  
- Overuse of generic phrases, boilerplate, or “throat-clearing” that does not advance the content.  
- Whether section length is justified by information content; no obvious padding or “water”.  

Scoring:  
- **9-10 – Highly concise**: The report is tight and efficient: almost no redundancy or filler. Each paragraph contributes new information or perspective relevant to the query.
- **7-8 – Mostly concise**: Some minor repetition or slightly wordy passages, but overall the report is reasonably succinct and does not feel bloated.
- **5-6 – Moderate redundancy**: The report contains noticeable repeated ideas or verbose phrasing. While still usable, it could be significantly tightened without loss of content.
- **3-4 – Substantial redundancy**: Many points are repeated, or large portions add little beyond what has already been stated. The report feels padded or overly long relative to its substantive content.
- **1-2 – Highly verbose and repetitive**: Extensive redundancy and filler severely dilute the message. The key ideas are buried under unnecessary text.

**4. Stylistic & Referencing Consistency**  

*Is the writing style internally consistent in terms of formatting, including its use of tone, terminology, citations, references, and figures?*

Focus:
- Tone: the level of formality is stable; the text does not jump needlessly between very informal and highly formal styles.  
- Terminology: key terms, labels, and names for the same concept, variable, or method are used consistently.  
- Citations: one coherent citation format is followed uniformly.
- References: the format of reference list entries is consistent.  
- Figures: 
  - Numbering format is consistent.  
  - Caption format is consistent.  
  - Source attribution format is consistent across all figures.
  - In-text citation format for figures is consistent.

**Important**: Evaluate only whether these elements are **internally consistent in formatting** throughout the report. Do NOT evaluate whether they conform to academic standards, whether content quality is appropriate, whether sources are credible, whether information is accurate, whether the number of sources is uniform, or whether source types are uniform. Focus solely on consistency of formatting: Are the same things formatted the same way throughout?

Scoring:
- **9-10 – Highly consistent formatting**: Tone level, terminology spelling/capitalization, citation format, reference list format, and figure formatting are uniform throughout. All formatting choices follow a single consistent pattern.
- **7-8 – Minor formatting inconsistencies**: Overall formatting is consistent, with only occasional small deviations that do not significantly affect the professional appearance.
- **5-6 – Noticeable formatting inconsistencies**: There are several visible inconsistencies in formatting that reduce visual polish but do not prevent understanding.
- **3-4 – Frequent formatting inconsistencies**: The report often switches between different formatting styles for the same elements. Citation formats are mixed, terminology formatting is unstable, and figure formatting varies significantly. This is visually distracting and appears unprofessional.
- **1-2 – Very inconsistent formatting**: Formatting is chaotic with no clear standard. The same terms are formatted differently throughout, citation styles change arbitrarily, reference entries have no consistent format, and figure numbering/captions follow no pattern. The report appears visually disorganized and unprofessional.

INSTRUCTIONS:
- **Step 1**: Review the provided **Research_Report**.  
- **Step 2**: Evaluate the **Writing Quality** of the Research_Report across the four dimensions: **Coherence & Organization**, **Clarity & Readability**, **Conciseness & Redundancy**, and **Stylistic & Referencing Consistency**.  
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

    print("🤖 Evaluating writing quality of the research report...")

    user_prompt = f"""
Please evaluate the Writing Quality of the following research report based on the provided rubric.

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
            time.sleep(5**attempt)

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
    eval_writing_quality(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
