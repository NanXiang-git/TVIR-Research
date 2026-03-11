from scripts.preprocess.extract_refs import extract_refs
from scripts.preprocess.extract_citations import extract_citations
from scripts.preprocess.extract_visuals import extract_visuals
from scripts.preprocess.extract_visuals_base64 import extract_visuals_base64
from scripts.preprocess.extract_web_content import extract_web_content
from scripts.preprocess.dedup_citations import dedup_citations
import argparse
import os
import sys
from datetime import datetime


def _banner(text: str, ch: str = "=", width: int = 90) -> str:
    text = f" {text} "
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    right = pad - left
    return (ch * left) + text + (ch * right)


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m"


class _Tee:
    """Write print output to both terminal and log file (with real-time flush)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root_dir", type=str, required=True, help="Root directory for reports"
    )
    parser.add_argument(
        "--eval_system_name",
        type=str,
        nargs="+",
        required=True,
        help="Evaluation system name",
    )
    parser.add_argument(
        "--query_id", type=str, nargs="+", required=True, help="Evaluation report ID"
    )
    args = parser.parse_args()

    for eval_system_name in args.eval_system_name:
        print(
            _color(_banner(f"Evaluation System: {eval_system_name}", "=", 90), "1;36")
        )

        log_file_path = f"../logs/preprocess/{eval_system_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        orig_out, orig_err = sys.stdout, sys.stderr

        with open(log_file_path, "a", encoding="utf-8") as log_fp:
            sys.stdout = _Tee(orig_out, log_fp)
            sys.stderr = _Tee(orig_err, log_fp)
            try:
                for query_id in args.query_id:
                    print(
                        _color(
                            _banner(f"Processing Query ID: {query_id}", "-", 90), "1;33"
                        )
                    )

                    refs = extract_refs(
                        args.report_root_dir, query_id, eval_system_name
                    )
                    citations = extract_citations(
                        args.report_root_dir, query_id, eval_system_name
                    )
                    if "error" not in citations:
                        deduped_citations = dedup_citations(
                            args.report_root_dir, query_id, eval_system_name
                        )
                    visuals = extract_visuals(
                        args.report_root_dir, query_id, eval_system_name
                    )
                    if "error" not in visuals:
                        visuals_base64 = extract_visuals_base64(
                            args.report_root_dir, query_id, eval_system_name
                        )
                    if (
                        "error" not in refs
                        and "error" not in citations
                        and "error" not in visuals
                    ):
                        extract_web_content(
                            args.report_root_dir, query_id, eval_system_name
                        )

            finally:
                sys.stdout = orig_out
                sys.stderr = orig_err
