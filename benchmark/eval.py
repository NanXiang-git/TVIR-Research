from scripts.evaluation.eval_figure_context_integration import (
    eval_figure_context_integration,
)
from scripts.evaluation.eval_analytical_depth_breadth import (
    eval_analytical_depth_breadth,
)
from scripts.evaluation.eval_chart_quality import eval_chart_quality
from scripts.evaluation.eval_chart_source_consistency import (
    eval_chart_source_consistency,
)
from scripts.evaluation.eval_citation_support import eval_citation_support
from scripts.evaluation.eval_factual_logical_consistency import (
    eval_factual_logical_consistency,
)
from scripts.evaluation.eval_instruction_alignment import eval_instruction_alignment
from scripts.evaluation.eval_multimodal_composition import eval_multimodal_composition
from scripts.evaluation.eval_figure_quality import eval_figure_quality
from scripts.evaluation.eval_figure_caption_quality import eval_figure_caption_quality
from scripts.evaluation.eval_image_quality import eval_image_quality
from scripts.evaluation.eval_writing_quality import eval_writing_quality
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
        "--result_root_dir",
        type=str,
        required=True,
        help="Root directory for evaluation results",
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

        log_file_path = f"../logs/eval/{eval_system_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

                    eval_chart_quality(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_image_quality(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_figure_quality(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_figure_caption_quality(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_figure_context_integration(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_chart_source_consistency(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_multimodal_composition(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_analytical_depth_breadth(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_writing_quality(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_factual_logical_consistency(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_instruction_alignment(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )
                    eval_citation_support(
                        args.report_root_dir,
                        query_id,
                        eval_system_name,
                        args.result_root_dir,
                    )

            finally:
                sys.stdout = orig_out
                sys.stderr = orig_err
