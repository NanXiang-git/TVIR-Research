import json
import requests
import base64
import os
import argparse
from PIL import Image
import cv2
import numpy as np
from io import BytesIO


def get_mime_type(img_url):
    ext = os.path.splitext(img_url)[-1].lower()
    if ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    else:
        return {"error": f"unsupported image type: {ext}"}


def detect_image_format(img_data):
    """Detect the actual format of the image"""
    signatures = {
        b"\x89PNG\r\n\x1a\n": "png",
        b"\xff\xd8\xff": "jpeg",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
        b"RIFF": "webp",
        b"\x00\x00\x00\x0cjP  ": "jp2",
        b"\x00\x00\x00 ftyp": "avif",
    }

    for sig, fmt in signatures.items():
        if img_data.startswith(sig):
            return fmt
    return "unknown"


def normalize_image_format(img_base64, target_format="png"):
    """
    Detect and convert image format

    Args:
        img_base64: base64 encoded image string (may have data:image/xxx;base64, prefix)
        target_format: target format (png, jpeg, webp, etc.)

    Returns:
        converted base64 string
    """
    try:
        # Parse base64 string
        if "," in img_base64:
            header, encoded_data = img_base64.split(",", 1)
        else:
            encoded_data = img_base64

        # Decode image data
        img_data = base64.b64decode(encoded_data)

        # Detect actual format
        actual_format = detect_image_format(img_data)

        # Convert if format mismatch or unknown
        if actual_format != target_format or actual_format == "unknown":
            print(
                f"🔄 Format mismatch detected: actual={actual_format}, target={target_format}"
            )

            # Decode using OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"⚠️ OpenCV cannot decode image, keeping original format")
                return img_base64

            # Encode to target format
            ext = f".{target_format}"
            success, buffer = cv2.imencode(ext, img)

            if not success:
                print(f"⚠️ Cannot encode to {target_format}, keeping original format")
                return img_base64

            # Convert to base64
            converted_data = base64.b64encode(buffer).decode("utf-8")
            result = f"data:image/{target_format};base64,{converted_data}"
            print(f"✅ Converted to {target_format} format")
            return result
        else:
            return f"data:image/{target_format};base64,{encoded_data}"

    except Exception as e:
        print(f"❌ Image format conversion failed: {e}")
        return img_base64


def flatten_alpha(img_bytes: bytes, mime_type: str):
    try:
        im = Image.open(BytesIO(img_bytes))
        has_alpha = im.mode in ("RGBA", "LA") or (
            im.mode == "P" and "transparency" in im.info
        )
        if not has_alpha:
            return img_bytes, mime_type

        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        out = Image.alpha_composite(bg, rgba).convert("RGB")

        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception:
        return img_bytes, mime_type


def get_base64_from_url(url, mime_type: str):
    try:
        resp = requests.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        img_bytes, final_mime = flatten_alpha(resp.content, mime_type)
        return base64.b64encode(img_bytes).decode("utf-8"), final_mime
    except Exception as e:
        return {"error": str(e)}, mime_type


def get_base64_from_path(path, mime_type: str):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        img_bytes, final_mime = flatten_alpha(raw, mime_type)
        return base64.b64encode(img_bytes).decode("utf-8"), final_mime
    except Exception as e:
        return {"error": str(e)}, mime_type


def extract_visuals_base64(report_root_dir, query_id, eval_system_name):
    VISUAL_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/visuals.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report.md"
    UPDATE_REPORT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"
    )
    OUTPUT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            visuals = json.load(f)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    for item in visuals:
        content = item.get("content", "")
        if not content:
            item["img_base64"] = {"error": "no content found"}
            continue

        mime_type = get_mime_type(content)
        if "error" in mime_type:
            item["img_base64"] = mime_type
            print(f"❌ Cannot recognize image type: {mime_type['error']}")
            continue

        if content.startswith("http"):
            img_base64, mime_type = get_base64_from_url(content, mime_type)
            if isinstance(img_base64, dict) and "error" in img_base64:
                print(f"❌ Failed to get image from URL: {img_base64['error']}")
                item["img_base64"] = img_base64
                report_content = report_content.replace(content, "")
            else:
                img_base64 = normalize_image_format(
                    img_base64, target_format=mime_type.split("/")[-1]
                )
                item["img_base64"] = img_base64
        else:
            img_path = os.path.join(os.path.dirname(VISUAL_PATH), content)
            img_base64, mime_type = get_base64_from_path(img_path, mime_type)
            if isinstance(img_base64, dict) and "error" in img_base64:
                print(f"❌ Failed to get image from local path: {img_base64['error']}")
                item["img_base64"] = img_base64
                report_content = report_content.replace(content, "")
            else:
                img_base64 = normalize_image_format(
                    img_base64, target_format=mime_type.split("/")[-1]
                )
                item["img_base64"] = img_base64

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(visuals, f, ensure_ascii=False, indent=2)
    with open(UPDATE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(
        f"✅ Visual elements Base64 encoding completed, results saved to {OUTPUT_PATH}"
    )

    return visuals


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
    extract_visuals_base64(args.report_root_dir, args.query_id, args.eval_system_name)
