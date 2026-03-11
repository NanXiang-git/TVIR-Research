import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import base64
from io import BytesIO
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5.2")

# ==============================
# 🔧 Configuration Parameters (Adjust as needed)
# ==============================

HASH_THRESHOLD = 10  # Hamming distance threshold (for pHash, 64-bit)
HASH_SIZE = 8  # pHash hash_size (8 → 64 bits)
MIN_RESOLUTION = 300  # Minimum width/height (pixels)
LOW_QUALITY_THRESHOLD = (
    0.3  # Image quality below this value is considered "low quality"
)


def compute_edge_density(gray, low_thresh=50, high_thresh=150):
    """Compute edge density"""
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return np.count_nonzero(edges) / edges.size


def assess_single_image(img_base64):
    """Assess the quality score of a single image [0,1]"""
    try:
        img_data = base64.b64decode(img_base64.split(",")[-1])
        pil_img = Image.open(BytesIO(img_data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None, None, None, None, None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Resolution score
    min_dim = min(h, w)
    R = min(1.0, min_dim / 400.0)

    # 2. Aspect ratio score
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio <= 2.0:
        A = 1.0
    elif aspect_ratio <= 4.0:
        A = 1.0 - (aspect_ratio - 2.0) / 4.0
    else:
        A = max(0.0, 0.5 - (aspect_ratio - 4.0) / 8.0)

    # 3. Clarity score
    lap_fine = cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var()  # Most detailed
    lap_medium = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()  # Medium detail
    lap_coarse = cv2.Laplacian(gray, cv2.CV_64F, ksize=5).var()  # Coarse outline

    # Adaptively adjust threshold based on image brightness
    brightness = gray.mean()
    if brightness > 120:  # Bright image
        threshold = 90
    elif brightness > 80:  # Normal
        threshold = 80
    else:  # Dark image
        threshold = 50

    S = np.clip(
        (0.5 * lap_fine + 0.3 * lap_medium + 0.2 * lap_coarse) / threshold, 0, 1
    )

    # 4. Contrast score (global + simplified local)
    global_std = gray.std()

    # Adaptive block size: ensure at least 4 blocks
    block_size = max(min(h, w) // 4, 64)
    local_stds = []

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i : i + block_size, j : j + block_size]
            if block.size >= 4096:  # At least 64x64
                local_stds.append(block.std())

    # Combine global and local
    if local_stds:
        local_contrast = np.median(local_stds)
        contrast = 0.7 * global_std + 0.3 * local_contrast
    else:
        contrast = global_std

    if brightness > 120:  # Bright image
        norm_threshold = 45.0
    elif brightness > 80:  # Normal
        norm_threshold = 55.0
    else:  # Dark image
        norm_threshold = 50.0

    C = np.clip(contrast / norm_threshold, 0, 1)

    # Overall score
    Q = 0.25 * R + 0.15 * A + 0.35 * S + 0.25 * C

    # pHash
    try:
        phash_val = imagehash.phash(pil_img, hash_size=HASH_SIZE)
        phash_str = str(phash_val)
    except Exception:
        phash_str = None

    return R, A, S, C, Q, phash_str


def eval_image_quality(report_root_dir, query_id, eval_system_name, result_root_dir):
    EVAL_RESULT_PATH = f"{result_root_dir}/{eval_system_name}/{query_id}/{EVAL_MODEL_NAME}/image_quality.json"
    VISUAL_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            figures = json.load(f)
    except Exception as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    images = [fig for fig in figures if fig.get("type") == "image"]

    if not images:
        result = {
            "normalized_score": 0.0,
            "item_count": 0,
            "scored_item_count": 0,
            "evaluation_results": [],
        }
        os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
        with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ No images found, zero score saved to {EVAL_RESULT_PATH}")
        return

    qualities, hashes = [], []
    evaluation_results = []
    valid_img_indices = []

    # Image analysis
    for idx, img in enumerate(images):
        img_base64 = img.get("img_base64", "")
        caption = img.get("caption", "")

        if isinstance(img_base64, dict) and "error" in img_base64:
            evaluation_results.append(
                {"caption": caption, "error": img_base64["error"], "quality_score": 0.0}
            )
            qualities.append(0.0)
            continue

        R, A, S, C, Q, ph = assess_single_image(img_base64)
        if Q is None or ph is None:
            evaluation_results.append(
                {
                    "caption": caption,
                    "error": "image processing failed",
                    "quality_score": 0.0,
                }
            )
            qualities.append(0.0)
            continue

        valid_img_indices.append(len(evaluation_results))
        qualities.append(Q)
        hashes.append(ph)
        evaluation_results.append(
            {
                "caption": img.get("caption", ""),
                "resolution_score": round(R, 4),
                "aspect_ratio_score": round(A, 4),
                "sharpness_score": round(S, 4),
                "contrast_score": round(C, 4),
                "weighted_quality_score": round(Q, 4),
                "duplicate": False,
                "duplicate_with": [],
            }
        )

    valid_N = len(hashes)

    base_score = float(np.mean(qualities))

    # Duplicate detection
    duplicate_groups = []
    used = [False] * valid_N
    for i in range(valid_N):
        if used[i]:
            continue
        group = [i]
        hi = imagehash.hex_to_hash(hashes[i])
        for j in range(i + 1, valid_N):
            if used[j]:
                continue
            hj = imagehash.hex_to_hash(hashes[j])
            if hi - hj <= HASH_THRESHOLD:
                group.append(j)
                used[j] = True
        if len(group) > 1:
            for idx in group:
                used[idx] = True
            duplicate_groups.append(group)

    # Mark duplicate information
    for group in duplicate_groups:
        group_global = [valid_img_indices[i] for i in group]
        for global_idx in group_global:
            evaluation_results[global_idx]["duplicate"] = True
            evaluation_results[global_idx]["duplicate_with"] = [
                x for x in group_global if x != global_idx
            ]

    total_duplicate_count = sum(len(g) - 1 for g in duplicate_groups)
    duplicate_ratio = total_duplicate_count / valid_N if valid_N > 0 else 0
    penalty = min(0.5, duplicate_ratio * 0.8)
    normalized_score = base_score * (1 - penalty)

    result = {
        "normalized_score": normalized_score,
        "item_count": len(images),
        "scored_item_count": len(images),
        "evaluation_results": evaluation_results,
    }

    # Save results
    os.makedirs(os.path.dirname(EVAL_RESULT_PATH), exist_ok=True)
    with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"✅ Evaluation completed, results saved to {EVAL_RESULT_PATH} | Normalized score: {normalized_score} | Scored items: {len(images)}/{len(images)}"
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
    eval_image_quality(
        args.report_root_dir, args.query_id, args.eval_system_name, args.result_root_dir
    )
