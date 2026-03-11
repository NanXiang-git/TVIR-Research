import json
import os
from tqdm import tqdm
import argparse
import requests
import time
from dotenv import load_dotenv

load_dotenv()

SERPER_BASE_URL = os.getenv("SERPER_BASE_URL")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


def extract_web_content(report_root_dir, query_id, eval_system_name):
    REF_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/refs.json"
    OUTPUT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/refs_with_web_content.json"
    )
    CITATION_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/citations.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(REF_PATH, "r", encoding="utf-8") as f:
            refs = json.load(f)
        with open(CITATION_PATH, "r", encoding="utf-8") as f:
            citations = json.load(f)
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            visuals = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    ref_idxs_set = set()
    for item in citations:
        ref_idxs_set.update(item.get("ref_idxs", []))
    for item in visuals:
        ref_idxs_set.update(item.get("ref_idxs", []))

    for item in tqdm(refs, desc="⏳ Extracting web content"):
        ref_url = item.get("ref_url")
        ref_idx = item.get("ref_idx")
        if str(ref_idx) not in ref_idxs_set:
            item["web_content"] = ""
            continue
        if not ref_url:
            item["web_content"] = ""
            continue

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    SERPER_BASE_URL,
                    headers={
                        "X-API-KEY": SERPER_API_KEY,
                        "Content-Type": "application/json",
                    },
                    data=json.dumps({"url": ref_url}),
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                item["web_content"] = result.get("text", "")

                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(4**attempt)

        if last_error is not None:
            print(f"❌ Error extracting web content: {last_error}")
            item["web_content"] = f"Error extracting web content: {last_error}"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    print(f"✅ Web content extraction completed, results saved to {OUTPUT_PATH}")


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
    extract_web_content(args.report_root_dir, args.query_id, args.eval_system_name)
