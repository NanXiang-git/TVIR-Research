# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio
import calendar
import datetime
import json
import os
import sys

import requests

from urllib.parse import urlparse
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
from mcp.client.stdio import stdio_client

# Load environment variables
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")


# Initialize FastMCP server
mcp = FastMCP("searching-image-mcp-server")

# Excluded keywords for filtering low-quality images
EXCLUDED_KEYWORDS = {
    # UI elements
    "icon",
    "logo",
    "favicon",
    "avatar",
    "thumbnail",
    "button",
    "badge",
    "banner",
    # Social media and profile images
    "profile",
    "headshot",
    # Decorative elements
    "emoji",
    "sticker",
    "decoration",
    # Ads and promotional
    "advertisement",
    "sponsor",
    # Small/low quality indicators
    "placeholder",
    "loading",
}

# Excluded file formats
EXCLUDED_FORMATS = {".svg", ".gif", ".ico"}


def verify_image_accessibility(image_url: str, timeout: int = 5) -> bool:
    """
    Verify if an image URL is accessible and returns valid image content.

    Args:
        image_url: The image URL to verify
        timeout: Request timeout in seconds (default: 3)

    Returns:
        True if image is accessible and valid, False otherwise
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        response = requests.head(
            image_url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,  # Let requests handle redirects automatically
            verify=True,  # Enable SSL verification for secure response
        )

        # Check if successful and is an image
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            print(f"Content-Type for {image_url}: {content_type}")
            return "image/" in content_type

        return False

    except Exception:
        return False


def is_low_quality_image(image_url: str, title: str) -> bool:
    """
    Determine if an image should be filtered out as low quality.

    Filtering criteria:
    1. URL or title contains excluded keywords (icon, logo, favicon, etc.)
    2. File format is in excluded formats (.svg, .gif, .ico)
    3. Image URL is not accessible or does not return valid image content

    Args:
        image_url: The image URL to check
        title: The image title or alt text

    Returns:
        True if the image is low quality (should be filtered), False if high quality (keep)
    """
    url_lower = image_url.lower()
    title_lower = title.lower()

    # 1. Check for excluded keywords in URL and title
    if any(keyword in url_lower for keyword in EXCLUDED_KEYWORDS):
        return True
    if any(keyword in title_lower for keyword in EXCLUDED_KEYWORDS):
        return True

    # 2. Check file extension
    if any(url_lower.endswith(ext) for ext in EXCLUDED_FORMATS):
        return True

    # 3. Verify image accessibility
    # if not verify_image_accessibility(image_url):
    #     return True

    return False


def filter_image_search_results(result_content: str) -> str:
    """
    Filter image search results to remove low-quality images.

    Args:
        result_content: JSON string result from Google image search

    Returns:
        Filtered JSON string with low-quality images removed
    """
    try:
        data = json.loads(result_content)

        if "images" not in data:
            return result_content

        # Filter out low-quality images
        filtered_images = []
        for item in data["images"]:
            image_url = item.get("imageUrl", "")
            title = item.get("title", "")

            # Normalize image URL
            normalize_url = normalize_image_url(image_url)

            # Keep high-quality images only
            if not is_low_quality_image(normalize_url, title):
                item["imageUrl"] = normalize_url
                filtered_images.append(item)

        # Re-assign position numbers after filtering
        for idx, item in enumerate(filtered_images, start=1):
            item["position"] = idx

        data["images"] = filtered_images
        return json.dumps(data, ensure_ascii=False, indent=None)

    except (json.JSONDecodeError, Exception):
        # Return original content if filtering fails
        return result_content


def normalize_image_url(image_url: str) -> str:
    """
    Normalize image URL by removing query parameters and fragments.
    This helps identify duplicate images with different URL parameters.

    Args:
        image_url: Original image URL

    Returns:
        Normalized URL without query parameters and fragments
    """
    try:
        parsed = urlparse(image_url)
        # Reconstruct URL without query and fragment
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return normalized_url
    except Exception:
        # If parsing fails, return original URL
        return image_url


@mcp.tool()
async def google_image_search(
    q: str,
    gl: str,
    hl: str,
    location: str = None,
    num: int = 5,
    tbs: str = None,
    page: int = 1,
) -> str:
    """Perform google image searches via Serper API and retrieve image results.

    Args:
        q: Image search query string.
        gl: Country context for search. Influences regional results priority:
            - 'us': United States - Use for most international topics and global content
            - 'cn': China - Use for China-specific content
        hl: Google interface language. Affects result metadata language:
            - 'en': English - Use for international topics
            - 'zh-cn': Simplified Chinese - Use when query is in Chinese
        location: City-level location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of image results to return (default: 5).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).

    Returns:
        The image search results in JSON format. Each image result contains:
        - title: 图片来源页面的标题
        - imageUrl: 图片的URL
        - imageWidth/imageHeight: 图片尺寸（像素）
        - source: 图片来源网站的名称
        - domain: 图片来源网站的域名
        - link: 图片来源页面的URL
        - position: 图片在搜索结果中的位置
    """
    if SERPER_API_KEY == "":
        return (
            "[ERROR]: SERPER_API_KEY is not set, google_search tool is not available."
        )

    tool_name = "google_image_search"

    if gl is None:
        gl = "us"
    if hl is None:
        hl = "en"

    arguments = {
        "q": q,
        "gl": gl,
        "hl": hl,
        "num": num,
        "page": page,
        "autocorrect": False,
    }
    if location:
        arguments["location"] = location
    if tbs:
        arguments["tbs"] = tbs
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.mcp_servers.serper_mcp_server"],
        env={"SERPER_API_KEY": SERPER_API_KEY, "SERPER_BASE_URL": SERPER_BASE_URL},
    )
    result_content = ""

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(
                    read, write, sampling_callback=None
                ) as session:
                    await session.initialize()
                    tool_result = await session.call_tool(
                        tool_name, arguments=arguments
                    )
                    result_content = (
                        tool_result.content[-1].text if tool_result.content else ""
                    )
                    assert (
                        result_content is not None and result_content.strip() != ""
                    ), "Empty result from google_image_search tool, please try again."
                    # Apply filtering based on environment variables
                    filtered_result = filter_image_search_results(result_content)
                    return filtered_result  # Success, exit retry loop
        except Exception as error:
            retry_count += 1
            if retry_count >= max_retries:
                return f"[ERROR]: google_image_search tool execution failed after {max_retries} attempts: {str(error)}"
            # Wait before retrying
            await asyncio.sleep(min(2**retry_count, 60))

    return (
        "[ERROR]: Unknown error occurred in google_image_search tool, please try again."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
