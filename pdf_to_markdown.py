#!/usr/bin/env python3
"""
Convert a PDF to Markdown using text extraction with optional OCR fallback.

Usage:
  python pdf_to_markdown.py input.pdf
  python pdf_to_markdown.py input.pdf -o output.md --mode ocr

Environment:
  SILICONFLOW_API_KEY=...  (required for OCR)

Dependencies:
  pip install pdfplumber pdf2image pillow requests

System dependency (pdf2image):
  - macOS: brew install poppler
  - Ubuntu: sudo apt-get install poppler-utils
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber
import requests
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/deepseek-vl2"
DEFAULT_SEPARATOR = "\n\n---\n\n"
DEFAULT_CACHE_DIRNAME = ".pdf_to_markdown_cache"
SCRIPT_VERSION = "0.4.0"

END_PUNCT = ".?!;:" + "\u3002\uff01\uff1f\uff1b\uff1a"

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_logging(quiet: bool, log_level: str) -> None:
    logger = logging.getLogger("pdf_to_markdown")
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    if quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))


@dataclass
class PageStat:
    page: int
    method: str
    seconds: float
    chars: int
    cached: bool = False
    error: Optional[str] = None
    model: Optional[str] = None
    api_url: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class LayoutDecision:
    mode: str
    split_ratio: Optional[float] = None


@dataclass
class OCRProvider:
    url: str
    api_key: str


class RateLimiter:
    def __init__(self, rps: float, max_interval: float = 10.0) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.max_interval = max_interval
        self.lock = Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.monotonic()
            self.next_time = now + self.min_interval

    def backoff(self, factor: float = 2.0, min_interval: float = 0.5) -> None:
        if factor <= 1.0:
            factor = 2.0
        with self.lock:
            if self.min_interval <= 0:
                self.min_interval = min_interval
            else:
                self.min_interval = min(self.min_interval * factor, self.max_interval)


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class TableExtractor:
    def __init__(self, backend: str, flavor: str, quiet: bool) -> None:
        self.backend = backend
        self.flavor = flavor
        self.quiet = quiet
        self.camelot = None
        self.tabula = None
        self.backends: List[str] = []
        self.errors: Dict[str, str] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        if self.backend in ("camelot", "auto"):
            try:
                import camelot  # type: ignore

                self.camelot = camelot
                self.backends.append("camelot")
            except Exception as exc:  # pragma: no cover - optional dependency
                self.errors["camelot"] = str(exc)
        if self.backend in ("tabula", "auto"):
            try:
                import tabula  # type: ignore

                self.tabula = tabula
                self.backends.append("tabula")
            except Exception as exc:  # pragma: no cover - optional dependency
                self.errors["tabula"] = str(exc)

    def available(self) -> bool:
        return bool(self.backends)

    def extract_tables(
        self, pdf_path: Path, page_number: int
    ) -> Tuple[List[List[List[str]]], Optional[str]]:
        for backend in self.backends:
            try:
                if backend == "camelot" and self.camelot:
                    tables = self.camelot.read_pdf(
                        str(pdf_path),
                        pages=str(page_number),
                        flavor=self.flavor,
                    )
                    return [table.df.values.tolist() for table in tables], "camelot"
                if backend == "tabula" and self.tabula:
                    dfs = self.tabula.read_pdf(
                        str(pdf_path),
                        pages=page_number,
                        multiple_tables=True,
                        lattice=self.flavor == "lattice",
                    )
                    return [df.values.tolist() for df in dfs], "tabula"
            except Exception as exc:  # pragma: no cover - optional dependency
                if not self.quiet:
                    print(f"Table extraction failed with {backend}: {exc}")
        return [], None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_escaped(value: str) -> str:
    if not value:
        return value
    return bytes(value, "utf-8").decode("unicode_escape")


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def unique_list(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_models(primary: str, fallback: Optional[str]) -> List[str]:
    models = split_csv(primary) if primary else []
    if not models:
        models = [DEFAULT_MODEL]
    models.extend(split_csv(fallback))
    return unique_list(models)


def parse_api_urls(primary: str, fallback: Optional[str]) -> List[str]:
    urls = [primary] if primary else []
    urls.extend(split_csv(fallback))
    if not urls:
        urls = [API_URL]
    return unique_list(urls)


def parse_api_keys(
    keys_arg: Optional[str], env_key: str, url_count: int
) -> List[Optional[str]]:
    keys = split_csv(keys_arg)
    if not keys and env_key:
        keys = [env_key]
    if not keys:
        return [None] * url_count
    if len(keys) == 1 and url_count > 1:
        keys = keys * url_count
    if len(keys) != url_count:
        raise ValueError("Number of API keys must match the number of API URLs.")
    return [key if key else None for key in keys]


def build_providers(
    api_urls: List[str], api_keys: List[Optional[str]]
) -> List[OCRProvider]:
    providers: List[OCRProvider] = []
    for url, key in zip(api_urls, api_keys):
        if key:
            providers.append(OCRProvider(url=url, api_key=key))
    return providers


def parse_pages_arg(pages_arg: str) -> List[int]:
    """
    Parse page list like: "20" or "2,20-21,34".
    Returns 1-based page numbers.
    """
    pages: List[int] = []
    if not pages_arg:
        return pages
    for part in re.split(r"[,\s]+", pages_arg.strip()):
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(p for p in pages if p > 0))


def resolve_pages(total_pages: int, include_pages: str, skip_pages: str) -> List[int]:
    include = parse_pages_arg(include_pages)
    skip = set(parse_pages_arg(skip_pages))

    if include:
        selected = [p for p in include if 1 <= p <= total_pages and p not in skip]
    else:
        selected = [p for p in range(1, total_pages + 1) if p not in skip]

    return sorted(set(selected))


def page_filename(page_number: int, width: int) -> str:
    return f"page_{page_number:0{width}d}.md"


def page_meta_filename(page_number: int, width: int) -> str:
    return f"page_{page_number:0{width}d}.json"


def image_cache_filename(page_number: int, width: int, image_format: str) -> str:
    ext = image_format.lower()
    if ext == "jpeg":
        ext = "jpg"
    return f"page_{page_number:0{width}d}.{ext}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def text_metrics(text: str) -> Dict[str, float | int]:
    normalized = normalize_text(text)
    total_chars = len(normalized)
    lines = normalized.splitlines()
    line_count = max(len(lines), 1)
    alnum_count = sum(ch.isalnum() for ch in normalized)
    alnum_ratio = alnum_count / max(total_chars, 1)
    avg_line_len = sum(len(line) for line in lines) / line_count
    blank_ratio = sum(1 for line in lines if not line.strip()) / line_count
    return {
        "total_chars": total_chars,
        "alnum_ratio": alnum_ratio,
        "avg_line_len": avg_line_len,
        "blank_ratio": blank_ratio,
    }


def looks_like_meaningful_text(
    text: str,
    min_chars: int,
    min_alnum_ratio: float,
    min_avg_line_len: float,
    max_blank_ratio: float,
) -> bool:
    if not text:
        return False
    metrics = text_metrics(text)
    if metrics["total_chars"] < min_chars:
        return False
    if metrics["alnum_ratio"] < min_alnum_ratio:
        return False
    if (
        metrics["avg_line_len"] < min_avg_line_len
        and metrics["total_chars"] < min_chars * 2
    ):
        return False
    if metrics["blank_ratio"] > max_blank_ratio:
        return False
    return True


def detect_language_hint(text: str) -> str:
    if not text:
        return ""
    sample = text[:1000]
    cjk = 0
    cyrillic = 0
    latin = 0
    for ch in sample:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            cjk += 1
        elif 0x0400 <= code <= 0x04FF:
            cyrillic += 1
        elif "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin += 1
    total = cjk + cyrillic + latin
    if total == 0:
        return ""
    if cjk / total >= 0.4:
        return "zh"
    if cyrillic / total >= 0.4:
        return "ru"
    if latin / total >= 0.4:
        return "en"
    return ""


def ocr_quality_metrics(text: str) -> Dict[str, float | int]:
    normalized = normalize_text(text)
    total_chars = len(normalized)
    alnum_count = sum(ch.isalnum() for ch in normalized)
    symbol_count = sum(1 for ch in normalized if not ch.isalnum() and not ch.isspace())
    replacement_count = (
        normalized.count("�") + normalized.count("□") + normalized.count("[??]")
    )
    alnum_ratio = alnum_count / max(total_chars, 1)
    symbol_ratio = symbol_count / max(total_chars, 1)
    replacement_ratio = replacement_count / max(total_chars, 1)
    return {
        "total_chars": total_chars,
        "alnum_ratio": alnum_ratio,
        "symbol_ratio": symbol_ratio,
        "replacement_ratio": replacement_ratio,
    }


def ocr_quality_ok(
    text: str,
    min_chars: int,
    min_alnum_ratio: float,
    max_symbol_ratio: float,
    max_replacement_ratio: float,
) -> Tuple[bool, str]:
    metrics = ocr_quality_metrics(text)
    if metrics["total_chars"] < min_chars:
        return False, f"total_chars<{min_chars}"
    if metrics["alnum_ratio"] < min_alnum_ratio:
        return False, f"alnum_ratio<{min_alnum_ratio}"
    if metrics["symbol_ratio"] > max_symbol_ratio:
        return False, f"symbol_ratio>{max_symbol_ratio}"
    if metrics["replacement_ratio"] > max_replacement_ratio:
        return False, f"replacement_ratio>{max_replacement_ratio}"
    return True, ""


def otsu_threshold(gray: Image.Image) -> int:
    histogram = gray.histogram()
    total = sum(histogram)
    sum_total = sum(i * count for i, count in enumerate(histogram))
    sum_b = 0
    w_b = 0
    max_var = 0.0
    threshold = 127
    for i, count in enumerate(histogram):
        w_b += count
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * count
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def binarize_image(image: Image.Image, threshold: int) -> Image.Image:
    gray = image.convert("L")
    th = threshold if threshold > 0 else otsu_threshold(gray)
    return gray.point(lambda p: 255 if p >= th else 0, mode="L")


def enhance_contrast(image: Image.Image, factor: float) -> Image.Image:
    if factor <= 0 or abs(factor - 1.0) < 1e-3:
        return image
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def denoise_image(image: Image.Image, size: int) -> Image.Image:
    if size <= 1:
        return image
    return image.filter(ImageFilter.MedianFilter(size=size))


def estimate_skew_angle(gray: Image.Image, max_angle: float, step: float) -> float:
    if max_angle <= 0 or step <= 0:
        return 0.0
    width, height = gray.size
    scale = min(600 / max(width, height), 1.0)
    if scale < 1.0:
        gray = gray.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    binary = binarize_image(gray, threshold=0)

    def score_angle(angle: float) -> float:
        rotated = binary.rotate(angle, expand=True, fillcolor=255)
        w, h = rotated.size
        pixels = rotated.load()
        row_sums = [0] * h
        for y in range(h):
            count = 0
            for x in range(w):
                if pixels[x, y] < 128:
                    count += 1
            row_sums[y] = count
        mean = sum(row_sums) / max(h, 1)
        variance = sum((val - mean) ** 2 for val in row_sums) / max(h, 1)
        return variance

    best_angle = 0.0
    best_score = -1.0
    angle = -max_angle
    while angle <= max_angle + 1e-6:
        score = score_angle(angle)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += step
    return best_angle


def deskew_image(image: Image.Image, max_angle: float, step: float) -> Image.Image:
    gray = image.convert("L")
    angle = estimate_skew_angle(gray, max_angle=max_angle, step=step)
    if abs(angle) < 1e-3:
        return image
    return image.rotate(angle, expand=True, fillcolor=255)


def preprocess_image(
    image: Image.Image,
    enabled: bool,
    denoise: bool,
    denoise_size: int,
    contrast: float,
    binarize: bool,
    binarize_threshold: int,
    deskew: bool,
    deskew_max_angle: float,
    deskew_step: float,
) -> Image.Image:
    if not enabled:
        return image
    img = image
    if deskew:
        img = deskew_image(img, max_angle=deskew_max_angle, step=deskew_step)
    if denoise:
        img = denoise_image(img, denoise_size)
    img = enhance_contrast(img, contrast)
    if binarize:
        img = binarize_image(img, binarize_threshold)
    return img


def adjust_max_side(
    image: Image.Image, max_side: int, max_pixels: int, min_side: int
) -> int:
    if max_pixels <= 0 or max_side <= 0:
        return max_side
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return max_side
    scale = math.sqrt(max_pixels / max(pixels, 1))
    adjusted = int(max_side * scale)
    return max(min_side, adjusted)


def split_image_blocks(
    image: Image.Image,
    gap_ratio: float,
    min_block_ratio: float,
    ink_ratio: float,
    white_threshold: int,
    max_blocks: int,
) -> List[Tuple[int, int]]:
    width, height = image.size
    scale = min(800 / max(width, height), 1.0)
    gray = image.convert("L")
    if scale < 1.0:
        gray = gray.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    w, h = gray.size
    pixels = gray.load()
    min_ink = max(1, int(w * ink_ratio))
    gap_rows = [
        sum(1 for x in range(w) if pixels[x, y] < white_threshold) <= min_ink
        for y in range(h)
    ]

    min_gap_rows = max(1, int(h * gap_ratio))
    min_block_rows = max(1, int(h * min_block_ratio))

    gaps: List[Tuple[int, int]] = []
    start = None
    for y, is_gap in enumerate(gap_rows):
        if is_gap and start is None:
            start = y
        if not is_gap and start is not None:
            if y - start >= min_gap_rows:
                gaps.append((start, y))
            start = None
    if start is not None and h - start >= min_gap_rows:
        gaps.append((start, h))

    blocks: List[Tuple[int, int]] = []
    prev = 0
    for gap_start, gap_end in gaps:
        if gap_start - prev >= min_block_rows:
            blocks.append((prev, gap_start))
        prev = gap_end
    if h - prev >= min_block_rows:
        blocks.append((prev, h))

    if len(blocks) <= 1 or len(blocks) > max_blocks:
        return [(0, height)]

    scale_back = 1.0 / scale if scale < 1.0 else 1.0
    mapped = [
        (int(top * scale_back), int(bottom * scale_back)) for top, bottom in blocks
    ]
    return mapped


def merge_hyphenated_lines(text: str) -> str:
    return re.sub(
        r"([A-Za-z])-\n(\s*[a-z])", lambda m: m.group(1) + m.group(2).lstrip(), text
    )


def normalize_hf_line(line: str, strip_digits: bool) -> str:
    normalized = re.sub(r"\s+", " ", line.strip())
    if strip_digits:
        normalized = re.sub(r"\d+", "#", normalized)
    return normalized.lower()


def is_page_number_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    lower = text.lower()
    patterns = [
        r"^(page\s*)?\d+(\s*/\s*\d+)?$",
        r"^\d+\s*(of|/|\\)\s*\d+$",
        r"^[ivxlcdm]+$",
        r"^第\s*\d+\s*页$",
        r"^第\s*\d+\s*/\s*\d+\s*页$",
    ]
    return any(re.fullmatch(pattern, lower) for pattern in patterns)


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 1:
        return values_sorted[mid]
    return (values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def extract_line_blocks(
    page: pdfplumber.page.Page, y_tolerance: float
) -> List[Dict[str, float | str]]:
    words = page.extract_words(use_text_flow=True) or []
    if not words:
        return []

    words.sort(key=lambda w: (w.get("top", 0), w.get("x0", 0)))
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_top: Optional[float] = None

    for word in words:
        top = float(word.get("top", 0))
        if current_top is None or abs(top - current_top) <= y_tolerance:
            current.append(word)
            if current_top is None:
                current_top = top
        else:
            lines.append(current)
            current = [word]
            current_top = top
    if current:
        lines.append(current)

    blocks: List[Dict[str, float | str]] = []
    chars = page.chars or []
    page_char_sizes = [float(ch.get("size", 0)) for ch in chars if ch.get("size")]
    page_median_size = median(page_char_sizes)

    for line_words in lines:
        line_words.sort(key=lambda w: w.get("x0", 0))
        text = " ".join(w.get("text", "") for w in line_words).strip()
        top = min(w.get("top", 0) for w in line_words)
        bottom = max(w.get("bottom", 0) for w in line_words)
        line_sizes = [
            float(ch.get("size", 0))
            for ch in chars
            if ch.get("size")
            and (bottom - y_tolerance) <= ch.get("top", 0) <= (bottom + y_tolerance)
        ]
        line_size = median(line_sizes) if line_sizes else page_median_size
        blocks.append({"text": text, "top": top, "bottom": bottom, "size": line_size})
    return blocks


def detect_headers_footers_with_layout(
    pdf: pdfplumber.PDF,
    selected_pages: List[int],
    top_margin: float,
    bottom_margin: float,
    min_ratio: float,
    min_pages: int,
    strip_digits: bool,
    font_max_ratio: float,
    line_tolerance: float,
) -> Tuple[set, set]:
    counts_top: Counter[str] = Counter()
    counts_bottom: Counter[str] = Counter()

    for page_number in selected_pages:
        page = pdf.pages[page_number - 1]
        height = float(page.height) if page.height else 0.0
        if height <= 0:
            continue
        top_limit = height * top_margin
        bottom_limit = height * (1.0 - bottom_margin)
        blocks = extract_line_blocks(page, y_tolerance=line_tolerance)
        if not blocks:
            continue
        page_sizes = [float(b.get("size", 0)) for b in blocks if b.get("size")]
        page_median = median(page_sizes)
        for block in blocks:
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            top = float(block.get("top", 0))
            bottom = float(block.get("bottom", 0))
            line_size = float(block.get("size", 0))
            if (
                page_median
                and font_max_ratio > 0
                and line_size > page_median * font_max_ratio
            ):
                continue
            signature = normalize_hf_line(text, strip_digits)
            if bottom <= top_limit:
                counts_top[signature] += 1
            if top >= bottom_limit:
                counts_bottom[signature] += 1

    total_pages = len(selected_pages)
    threshold = max(min_pages, int(total_pages * min_ratio + 0.999))
    header = {sig for sig, count in counts_top.items() if count >= threshold and sig}
    footer = {sig for sig, count in counts_bottom.items() if count >= threshold and sig}
    return header, footer


def detect_headers_footers(
    pages: Dict[int, str],
    top_n: int,
    bottom_n: int,
    min_ratio: float,
    min_pages: int,
    strip_digits: bool,
) -> Tuple[set, set]:
    if not pages:
        return set(), set()
    counts_top: Counter[str] = Counter()
    counts_bottom: Counter[str] = Counter()
    for content in pages.values():
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines[:top_n]:
            counts_top[normalize_hf_line(line, strip_digits)] += 1
        for line in lines[-bottom_n:] if bottom_n > 0 else []:
            counts_bottom[normalize_hf_line(line, strip_digits)] += 1

    total_pages = len(pages)
    threshold = max(min_pages, int(total_pages * min_ratio + 0.999))
    header = {sig for sig, count in counts_top.items() if count >= threshold and sig}
    footer = {sig for sig, count in counts_bottom.items() if count >= threshold and sig}
    return header, footer


def remove_headers_footers(
    text: str,
    header_sigs: set,
    footer_sigs: set,
    top_n: int,
    bottom_n: int,
    strip_digits: bool,
    remove_page_numbers: bool,
) -> str:
    if not header_sigs and not footer_sigs and not remove_page_numbers:
        return text
    lines = text.splitlines()
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    remove_indices = set()

    for pos, idx in enumerate(non_empty_indices):
        line = lines[idx]
        sig = normalize_hf_line(line, strip_digits)
        is_top = pos < top_n
        is_bottom = pos >= len(non_empty_indices) - bottom_n if bottom_n > 0 else False

        if is_top and header_sigs and sig in header_sigs:
            remove_indices.add(idx)
        if is_bottom and footer_sigs and sig in footer_sigs:
            remove_indices.add(idx)
        if remove_page_numbers and (is_top or is_bottom) and is_page_number_line(line):
            remove_indices.add(idx)

    cleaned_lines = [line for i, line in enumerate(lines) if i not in remove_indices]
    return "\n".join(cleaned_lines).strip()


def is_block_line(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith(">"):
        return True
    if stripped.startswith("|"):
        return True
    if re.match(r"^\s*[-*_]{3,}\s*$", stripped):
        return True
    if re.match(r"^\s*([-*+]|\d+[.)])\s+", stripped):
        return True
    if re.match(r"^\s{4,}\S", line):
        return True
    if stripped.startswith("```"):
        return True
    if stripped.count("|") >= 2 and stripped.startswith("|"):
        return True
    return False


def should_join(prev_line: str, next_line: str) -> bool:
    if is_block_line(prev_line) or is_block_line(next_line):
        return False
    prev = prev_line.rstrip()
    if prev.endswith(tuple(END_PUNCT)):
        return False
    return True


def reflow_paragraphs(text: str) -> str:
    lines = text.splitlines()
    out_lines: List[str] = []
    buffer = ""
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            if buffer:
                out_lines.append(buffer)
                buffer = ""
            out_lines.append(line.rstrip())
            in_code_block = not in_code_block
            continue

        if in_code_block:
            out_lines.append(line.rstrip())
            continue

        if not line.strip():
            if buffer:
                out_lines.append(buffer)
                buffer = ""
            out_lines.append("")
            continue

        if is_block_line(line):
            if buffer:
                out_lines.append(buffer)
                buffer = ""
            out_lines.append(line.rstrip())
            continue

        if buffer:
            if should_join(buffer, line.strip()):
                buffer = buffer.rstrip() + " " + line.strip()
            else:
                out_lines.append(buffer)
                buffer = line.strip()
        else:
            buffer = line.strip()

    if buffer:
        out_lines.append(buffer)

    return "\n".join(out_lines).strip()


def apply_short_warning(
    content: str,
    min_chars: int,
    warn_prefix: str,
    empty_placeholder: str,
) -> Tuple[str, Optional[str]]:
    if min_chars <= 0:
        return content, None
    trimmed = content.strip()
    if trimmed and len(trimmed) >= min_chars:
        return content, None
    warning = f"{warn_prefix} content length below {min_chars} chars"
    if not trimmed:
        trimmed = empty_placeholder
    decorated = f"> {warning}\n\n{trimmed}".strip()
    return decorated, warning


def normalize_paragraph(text: str, strip_digits: bool) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if strip_digits:
        normalized = re.sub(r"\d+", "#", normalized)
    return normalized.lower()


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def dedupe_pages(
    pages: Dict[int, str],
    min_ratio: float,
    min_pages: int,
    min_chars: int,
    strip_digits: bool,
) -> Tuple[Dict[int, str], int]:
    if not pages:
        return pages, 0

    counts: Counter[str] = Counter()
    page_paragraphs: Dict[int, List[str]] = {}
    page_signatures: Dict[int, List[str]] = {}

    for page_number, content in pages.items():
        paragraphs = split_paragraphs(content)
        page_paragraphs[page_number] = paragraphs
        sigs: List[str] = []
        for para in paragraphs:
            if len(para) < min_chars:
                continue
            sig = normalize_paragraph(para, strip_digits)
            if sig:
                sigs.append(sig)
        page_signatures[page_number] = sigs
        counts.update(set(sigs))

    total_pages = len(pages)
    threshold = max(min_pages, int(total_pages * min_ratio + 0.999))
    remove_sigs = {sig for sig, count in counts.items() if count >= threshold}

    removed = 0
    cleaned: Dict[int, str] = {}
    for page_number, paragraphs in page_paragraphs.items():
        new_paragraphs: List[str] = []
        for para in paragraphs:
            if len(para) >= min_chars:
                sig = normalize_paragraph(para, strip_digits)
                if sig in remove_sigs:
                    removed += 1
                    continue
            new_paragraphs.append(para)
        cleaned[page_number] = "\n\n".join(new_paragraphs).strip()

    return cleaned, removed


def load_replacements(path: Optional[str]) -> List[Tuple[str, str]]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    replacements: List[Tuple[str, str]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=>" in stripped:
            src, dst = stripped.split("=>", 1)
        elif "->" in stripped:
            src, dst = stripped.split("->", 1)
        elif "\t" in stripped:
            src, dst = stripped.split("\t", 1)
        else:
            continue
        replacements.append((src.strip(), dst.strip()))
    return replacements


def apply_replacements(
    text: str,
    replacements: List[Tuple[str, str]],
    mode: str,
    ignore_case: bool,
) -> str:
    if not replacements:
        return text
    flags = re.IGNORECASE if ignore_case else 0
    result = text
    for src, dst in replacements:
        if not src:
            continue
        if mode == "word":
            pattern = re.compile(r"\b" + re.escape(src) + r"\b", flags)
            result = pattern.sub(dst, result)
        else:
            if ignore_case:
                pattern = re.compile(re.escape(src), flags)
                result = pattern.sub(dst, result)
            else:
                result = result.replace(src, dst)
    return result


def detect_heading_level(line: str, base_level: int, max_len: int) -> Optional[int]:
    text = line.strip()
    if not text or len(text) > max_len:
        return None
    if text.startswith("#"):
        return None
    if re.match(r"^\s*([-*+]|\d+[.)])\s+", text):
        return None
    cn_num = r"[一二三四五六七八九十百千零〇两0-9]+"
    if re.match(rf"^第\s*{cn_num}\s*[章节篇]\b", text):
        return base_level
    if re.match(rf"^第\s*{cn_num}\s*节\b", text):
        return min(base_level + 1, 6)
    if re.match(r"^(chapter|chap\.?|section)\s+\d+", text, re.IGNORECASE):
        return base_level
    if re.match(r"^\d+(?:\.\d+){1,4}\s+\S", text):
        level = base_level + text.count(".")
        return min(level, 6)
    return None


def auto_headings(text: str, base_level: int, max_len: int) -> str:
    lines = text.splitlines()
    out_lines: List[str] = []
    for line in lines:
        level = detect_heading_level(line, base_level=base_level, max_len=max_len)
        if level:
            out_lines.append("#" * level + " " + line.strip())
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def run_hook(
    command: Optional[str],
    content: str,
    page_number: int,
    method: str,
    pdf_path: Path,
    stage: str,
    continue_on_error: bool,
    quiet: bool,
) -> str:
    if not command:
        return content
    args = shlex.split(command)
    env = os.environ.copy()
    env.update(
        {
            "PDF_PATH": str(pdf_path),
            "PAGE_NUMBER": str(page_number),
            "METHOD": method,
            "STAGE": stage,
        }
    )
    result = subprocess.run(
        args,
        input=content,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "Hook failed"
        if continue_on_error:
            if not quiet:
                print(f"Hook error on page {page_number} ({stage}): {message}")
            return content
        raise RuntimeError(f"Hook error on page {page_number} ({stage}): {message}")
    return result.stdout.rstrip("\n")


def postprocess_pages(
    raw_pages: Dict[int, str],
    methods_by_page: Dict[int, str],
    pdf_path: Path,
    postprocess: bool,
    dehyphenate: bool,
    reflow: bool,
    remove_hf: bool,
    hf_top: int,
    hf_bottom: int,
    hf_min_ratio: float,
    hf_min_pages: int,
    hf_strip_digits: bool,
    remove_page_numbers: bool,
    warn_short: int,
    warn_prefix: str,
    empty_placeholder: str,
    pre_hook: Optional[str],
    post_hook: Optional[str],
    continue_on_error: bool,
    quiet: bool,
    header_sigs: Optional[set],
    footer_sigs: Optional[set],
    dedupe: bool,
    dedupe_min_ratio: float,
    dedupe_min_pages: int,
    dedupe_min_chars: int,
    dedupe_strip_digits: bool,
    replacements: List[Tuple[str, str]],
    replace_mode: str,
    replace_ignore_case: bool,
    auto_heading: bool,
    heading_level: int,
    heading_max_len: int,
) -> Tuple[Dict[int, str], Dict[int, Optional[str]], int]:
    cleaned: Dict[int, str] = {}
    warnings: Dict[int, Optional[str]] = {}

    normalized_pages = {
        page: normalize_text(content) for page, content in raw_pages.items()
    }

    if postprocess and remove_hf and header_sigs is None and footer_sigs is None:
        header_sigs, footer_sigs = detect_headers_footers(
            normalized_pages,
            top_n=hf_top,
            bottom_n=hf_bottom,
            min_ratio=hf_min_ratio,
            min_pages=hf_min_pages,
            strip_digits=hf_strip_digits,
        )

    header_sigs = header_sigs or set()
    footer_sigs = footer_sigs or set()

    for page_number in sorted(normalized_pages):
        content = normalized_pages[page_number]
        method = methods_by_page.get(page_number, "unknown")

        content = run_hook(
            pre_hook,
            content,
            page_number,
            method,
            pdf_path,
            "pre",
            continue_on_error,
            quiet,
        )

        if postprocess and remove_hf:
            content = remove_headers_footers(
                content,
                header_sigs,
                footer_sigs,
                top_n=hf_top,
                bottom_n=hf_bottom,
                strip_digits=hf_strip_digits,
                remove_page_numbers=remove_page_numbers,
            )

        if postprocess and dehyphenate:
            content = merge_hyphenated_lines(content)

        if postprocess and reflow:
            content = reflow_paragraphs(content)

        if postprocess and replacements:
            content = apply_replacements(
                content, replacements, replace_mode, replace_ignore_case
            )

        if postprocess and auto_heading:
            content = auto_headings(content, heading_level, heading_max_len)

        content = normalize_text(content)

        content = run_hook(
            post_hook,
            content,
            page_number,
            method,
            pdf_path,
            "post",
            continue_on_error,
            quiet,
        )

        cleaned[page_number] = content

    dedupe_removed = 0
    if postprocess and dedupe:
        cleaned, dedupe_removed = dedupe_pages(
            cleaned,
            min_ratio=dedupe_min_ratio,
            min_pages=dedupe_min_pages,
            min_chars=dedupe_min_chars,
            strip_digits=dedupe_strip_digits,
        )

    for page_number in sorted(cleaned):
        content, warning = apply_short_warning(
            cleaned[page_number], warn_short, warn_prefix, empty_placeholder
        )
        cleaned[page_number] = content
        warnings[page_number] = warning

    return cleaned, warnings, dedupe_removed


def pil_to_data_url(img: Image.Image, fmt: str = "PNG", max_side: int = 2000) -> str:
    img = img.convert("RGB") if fmt.upper() in ("JPG", "JPEG") else img.convert("RGBA")

    width, height = img.size
    scale = min(max_side / max(width, height), 1.0)
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    from io import BytesIO

    buf = BytesIO()
    save_fmt = "PNG" if fmt.upper() == "PNG" else "JPEG"
    if save_fmt == "JPEG":
        img = img.convert("RGB")
        img.save(buf, format=save_fmt, quality=90, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format=save_fmt, optimize=True)
        mime = "image/png"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_default_prompt(page_index: int, language_hint: str) -> str:
    lines = [
        "You are a high-accuracy OCR engine. Extract all text from the image and output Markdown.",
    ]
    if language_hint:
        lines.append(f"Language hint: {language_hint}")
    lines.extend(
        [
            "Requirements:",
            "1) Preserve reading order (top to bottom, left to right; multi-column left then right).",
            "2) Preserve paragraphs and line breaks as much as possible; use #/##/### for headings.",
            "3) Reconstruct tables as Markdown tables when possible; otherwise use aligned plain text.",
            "4) Do not invent content; use [??] for unreadable characters.",
            "5) Output only the final Markdown, no explanations or code fences.",
            f"(This is page {page_index}.)",
        ]
    )
    return "\n".join(lines)


def build_prompt(page_index: int, template: Optional[str], language_hint: str) -> str:
    if template:
        return template.format_map(
            SafeDict(page=page_index, language=language_hint or "")
        ).strip()
    return build_default_prompt(page_index, language_hint)


def call_vlm_ocr_markdown(
    providers: List[OCRProvider],
    models: List[str],
    image_data_url: str,
    page_index: int,
    detail: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    max_continuations: int,
    max_retries: int,
    sleep_base: float,
    rate_limiter: Optional[RateLimiter],
    prompt_template: Optional[str],
    language_hint: str,
    quality_check: bool,
    quality_min_chars: int,
    quality_min_alnum: float,
    quality_max_symbol: float,
    quality_max_replace: float,
) -> Tuple[str, str, str]:
    prompt = build_prompt(page_index, prompt_template, language_hint)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": detail},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    def post_with_retries(payload: dict, api_url: str, api_key: str) -> dict:
        last_err: Optional[Exception] = None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, max_retries + 1):
            try:
                if rate_limiter:
                    rate_limiter.wait()
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=timeout
                )
                if response.status_code in (429, 503):
                    if rate_limiter:
                        rate_limiter.backoff()
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(float(retry_after))
                        except ValueError:
                            time.sleep(sleep_base * (2 ** (attempt - 1)))
                    else:
                        time.sleep(sleep_base * (2 ** (attempt - 1)))
                    last_err = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:800]}"
                    )
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:800]}"
                    )
                return response.json()
            except Exception as exc:
                if rate_limiter and "HTTP 429" in str(exc):
                    rate_limiter.backoff()
                last_err = exc
                time.sleep(sleep_base * (2 ** (attempt - 1)))
        raise RuntimeError(
            f"VLM OCR failed after {max_retries} retries. Last error: {last_err}"
        )

    last_error: Optional[Exception] = None
    for provider in providers:
        for model in models:
            try:
                parts: List[str] = []
                local_messages = list(messages)
                for _ in range(max_continuations + 1):
                    payload = {
                        "model": model,
                        "messages": local_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }

                    data = post_with_retries(payload, provider.url, provider.api_key)
                    choice = data["choices"][0]
                    content = normalize_text(choice["message"]["content"])
                    parts.append(content)

                    finish_reason = choice.get("finish_reason")
                    if finish_reason != "length":
                        break

                    local_messages.append({"role": "assistant", "content": content})
                    local_messages.append(
                        {
                            "role": "user",
                            "content": "Continue from the last line. Do not repeat. Output only Markdown.",
                        }
                    )

                combined = normalize_text("\n".join(parts))
                if quality_check:
                    ok, reason = ocr_quality_ok(
                        combined,
                        min_chars=quality_min_chars,
                        min_alnum_ratio=quality_min_alnum,
                        max_symbol_ratio=quality_max_symbol,
                        max_replacement_ratio=quality_max_replace,
                    )
                    if not ok:
                        last_error = RuntimeError(f"OCR quality check failed: {reason}")
                        continue
                return combined, model, provider.url
            except Exception as exc:
                last_error = exc
                continue

    raise RuntimeError(
        f"VLM OCR failed for all models/providers. Last error: {last_error}"
    )


def extract_text_page(pdf: pdfplumber.PDF, page_index: int) -> str:
    page = pdf.pages[page_index]
    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    return normalize_text(text)


def render_page_to_file_task(args: Tuple[str, int, int, str, str]) -> str:
    pdf_path, page_number, dpi, image_format, output_path = args
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt=image_format.lower(),
        first_page=page_number,
        last_page=page_number,
    )
    if not images:
        raise RuntimeError(f"Failed to render page {page_number} for OCR.")
    image = images[0]
    save_format = "JPEG" if image_format.upper() in ("JPG", "JPEG") else "PNG"
    save_kwargs = {"optimize": True}
    if save_format == "JPEG":
        save_kwargs.update({"quality": 90})
    image.save(output_path, format=save_format, **save_kwargs)
    return output_path


def render_image_to_path(
    pdf_path: Path,
    page_number: int,
    dpi: int,
    image_format: str,
    cache_dir: Path,
    width: int,
    cache_images: bool,
    render_pool: Optional[ProcessPoolExecutor],
) -> Tuple[Path, bool]:
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if cache_images:
        image_path = image_dir / image_cache_filename(page_number, width, image_format)
        if image_path.exists():
            return image_path, False
    else:
        suffix = uuid.uuid4().hex[:8]
        ext = image_format.lower()
        if ext == "jpeg":
            ext = "jpg"
        image_path = image_dir / f"page_{page_number:0{width}d}_tmp_{suffix}.{ext}"

    task = (str(pdf_path), page_number, dpi, image_format, str(image_path))
    if render_pool:
        future = render_pool.submit(render_page_to_file_task, task)
        future.result()
    else:
        render_page_to_file_task(task)

    return image_path, not cache_images


def detect_two_columns(
    page: pdfplumber.page.Page,
    min_words: int,
    gap_ratio: float,
    left_ratio: float,
    right_ratio: float,
) -> Optional[float]:
    words = page.extract_words() or []
    if len(words) < min_words:
        return None
    width = float(page.width) if page.width else 0.0
    if width <= 0:
        return None

    left_words = [w for w in words if w.get("x1", 0) <= width * left_ratio]
    right_words = [w for w in words if w.get("x0", width) >= width * right_ratio]

    if not left_words or not right_words:
        return None

    left_max = max(w.get("x1", 0) for w in left_words)
    right_min = min(w.get("x0", width) for w in right_words)

    if right_min - left_max < width * gap_ratio:
        return None

    split_x = (left_max + right_min) / 2.0
    return split_x / width


def decide_layout(
    page: pdfplumber.page.Page,
    layout_mode: str,
    gap_ratio: float,
    min_words: int,
    left_ratio: float,
    right_ratio: float,
    default_split: float,
) -> LayoutDecision:
    if layout_mode == "single":
        return LayoutDecision(mode="single")
    if layout_mode == "columns":
        return LayoutDecision(mode="columns", split_ratio=default_split)
    split_ratio = detect_two_columns(
        page,
        min_words=min_words,
        gap_ratio=gap_ratio,
        left_ratio=left_ratio,
        right_ratio=right_ratio,
    )
    if split_ratio is None:
        return LayoutDecision(mode="single")
    return LayoutDecision(mode="columns", split_ratio=split_ratio)


def ocr_image_by_layout(
    image: Image.Image,
    layout: LayoutDecision,
    providers: List[OCRProvider],
    models: List[str],
    page_number: int,
    detail: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    max_continuations: int,
    max_retries: int,
    sleep_base: float,
    rate_limiter: Optional[RateLimiter],
    prompt_template: Optional[str],
    language_hint: str,
    image_format: str,
    max_side: int,
    column_overlap: float,
    quality_check: bool,
    quality_min_chars: int,
    quality_min_alnum: float,
    quality_max_symbol: float,
    quality_max_replace: float,
    preprocess_enabled: bool,
    denoise: bool,
    denoise_size: int,
    contrast: float,
    binarize: bool,
    binarize_threshold: int,
    deskew: bool,
    deskew_max_angle: float,
    deskew_step: float,
    split_blocks: bool,
    block_gap_ratio: float,
    block_min_ratio: float,
    block_ink_ratio: float,
    block_white_threshold: int,
    block_max_blocks: int,
    ocr_max_pixels: int,
    ocr_skip_large: bool,
    ocr_min_side: int,
) -> Tuple[str, str, str]:
    def run_single(img: Image.Image) -> Tuple[str, str, str]:
        processed = preprocess_image(
            img,
            enabled=preprocess_enabled,
            denoise=denoise,
            denoise_size=denoise_size,
            contrast=contrast,
            binarize=binarize,
            binarize_threshold=binarize_threshold,
            deskew=deskew,
            deskew_max_angle=deskew_max_angle,
            deskew_step=deskew_step,
        )
        if ocr_max_pixels > 0:
            pixels = processed.size[0] * processed.size[1]
            if pixels > ocr_max_pixels and ocr_skip_large:
                raise RuntimeError("OCR skipped due to image size")
        effective_max_side = adjust_max_side(
            processed, max_side, ocr_max_pixels, ocr_min_side
        )
        data_url = pil_to_data_url(
            processed, fmt=image_format, max_side=effective_max_side
        )
        return call_vlm_ocr_markdown(
            providers=providers,
            models=models,
            image_data_url=data_url,
            page_index=page_number,
            detail=detail,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            max_continuations=max_continuations,
            max_retries=max_retries,
            sleep_base=sleep_base,
            rate_limiter=rate_limiter,
            prompt_template=prompt_template,
            language_hint=language_hint,
            quality_check=quality_check,
            quality_min_chars=quality_min_chars,
            quality_min_alnum=quality_min_alnum,
            quality_max_symbol=quality_max_symbol,
            quality_max_replace=quality_max_replace,
        )

    if layout.mode == "columns" and layout.split_ratio:
        width, height = image.size
        split_px = int(width * layout.split_ratio)
        overlap_px = int(width * column_overlap)
        left_box = (0, 0, min(width, split_px + overlap_px), height)
        right_box = (max(0, split_px - overlap_px), 0, width, height)

        left_img = image.crop(left_box)
        right_img = image.crop(right_box)

        left_text, left_model, left_api = run_single(left_img)
        right_text, right_model, right_api = run_single(right_img)

        combined_text = normalize_text(f"{left_text}\n\n{right_text}")
        models_used = unique_list([left_model, right_model])
        apis_used = unique_list([left_api, right_api])
        return combined_text, ",".join(models_used), ",".join(apis_used)

    if split_blocks:
        blocks = split_image_blocks(
            image,
            gap_ratio=block_gap_ratio,
            min_block_ratio=block_min_ratio,
            ink_ratio=block_ink_ratio,
            white_threshold=block_white_threshold,
            max_blocks=block_max_blocks,
        )
        if len(blocks) > 1:
            parts: List[str] = []
            models_used: List[str] = []
            apis_used: List[str] = []
            for top, bottom in blocks:
                block_img = image.crop((0, top, image.size[0], bottom))
                text, model_used, api_used = run_single(block_img)
                parts.append(text)
                models_used.append(model_used)
                apis_used.append(api_used)
            combined = normalize_text("\n\n".join(parts))
            return (
                combined,
                ",".join(unique_list(models_used)),
                ",".join(unique_list(apis_used)),
            )

    return run_single(image)


def table_to_markdown(rows: List[List[str]], use_header: bool) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    normalized_rows: List[List[str]] = []
    for row in rows:
        normalized = [str(cell).strip() for cell in row]
        normalized += [""] * (col_count - len(normalized))
        normalized_rows.append(normalized)

    if use_header:
        header = normalized_rows[0]
        body = normalized_rows[1:]
    else:
        header = [f"col{i + 1}" for i in range(col_count)]
        body = normalized_rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def tables_to_markdown(tables: List[List[List[str]]], use_header: bool) -> str:
    md_tables = []
    for rows in tables:
        md = table_to_markdown(rows, use_header=use_header)
        if md:
            md_tables.append(md)
    return "\n\n".join(md_tables)


def write_page_cache(
    cache_dir: Path, page_number: int, width: int, content: str
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / page_filename(page_number, width)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def write_page_meta(cache_dir: Path, page_number: int, width: int, meta: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / page_meta_filename(page_number, width)
    path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def read_page_cache(cache_dir: Path, page_number: int, width: int) -> str:
    path = cache_dir / page_filename(page_number, width)
    return path.read_text(encoding="utf-8").strip()


def read_page_meta(cache_dir: Path, page_number: int, width: int) -> Optional[dict]:
    path = cache_dir / page_meta_filename(page_number, width)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stat_from_meta(
    meta: Optional[dict], page_number: int, content: str, cached: bool
) -> PageStat:
    if not meta:
        return PageStat(
            page=page_number,
            method="cached",
            seconds=0.0,
            chars=len(content),
            cached=cached,
        )
    return PageStat(
        page=page_number,
        method=meta.get("method", "cached"),
        seconds=float(meta.get("seconds", 0.0)),
        chars=int(meta.get("chars", len(content))),
        cached=cached,
        error=meta.get("error"),
        model=meta.get("model"),
        api_url=meta.get("api_url"),
        warning=meta.get("warning"),
    )


def write_per_page_output(
    per_page_dir: Path,
    page_number: int,
    width: int,
    content: str,
    page_header_format: str,
    include_header: bool,
) -> None:
    per_page_dir.mkdir(parents=True, exist_ok=True)
    page_content = content
    if include_header and page_header_format:
        header = page_header_format.format(page=page_number)
        page_content = f"{header}\n\n{page_content}".strip()
    path = per_page_dir / page_filename(page_number, width)
    path.write_text(page_content.rstrip() + "\n", encoding="utf-8")


def merge_pages(
    out_path: Path,
    title: str,
    selected_pages: List[int],
    content_by_page: Dict[int, str],
    separator: str,
    page_header_format: str,
    include_page_header: bool,
    include_title: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        if include_title and title:
            out_file.write(f"# {title}\n\n")

        for idx, page_number in enumerate(selected_pages):
            if idx > 0:
                out_file.write(separator)
            if include_page_header and page_header_format:
                header = page_header_format.format(page=page_number)
                out_file.write(f"{header}\n\n")

            content = content_by_page.get(page_number, "")
            if content:
                out_file.write(f"{content}\n")


def load_prompt_template(prompt_file: Optional[str]) -> Optional[str]:
    if not prompt_file:
        return None
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8").strip()


def build_config_snapshot(
    args: argparse.Namespace,
    pdf_path: Path,
    out_path: Path,
    cache_dir: Path,
    models: List[str],
    api_urls: List[str],
    providers: List[OCRProvider],
) -> dict:
    args_dict = vars(args).copy()
    if args_dict.get("api_key"):
        args_dict["api_key"] = "***"
    snapshot = {
        "version": SCRIPT_VERSION,
        "timestamp": now_iso(),
        "python": sys.version.split()[0],
        "pdf_path": str(pdf_path),
        "out_path": str(out_path),
        "cache_dir": str(cache_dir),
        "models": models,
        "api_urls": api_urls,
        "provider_count": len(providers),
        "args": args_dict,
    }
    return snapshot


def write_json_output(
    json_out: Path,
    config: dict,
    selected_pages: List[int],
    cleaned_pages: Dict[int, str],
    stats_by_page: Dict[int, PageStat],
    summary: dict,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_number in selected_pages:
        stat = stats_by_page.get(page_number)
        pages.append(
            {
                "page": page_number,
                "text": cleaned_pages.get(page_number, ""),
                "stats": asdict(stat) if stat else None,
            }
        )
    payload = {
        "config": config,
        "summary": summary,
        "pages": pages,
    }
    json_out.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def load_failed_pages_from_stats(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("summary", {}).get("error_pages", [])
    if pages:
        return sorted(set(int(p) for p in pages))
    result: List[int] = []
    for page_entry in data.get("pages", []):
        stats = page_entry.get("stats") or {}
        if stats and stats.get("error"):
            result.append(int(stats.get("page", 0)))
    return sorted(set(p for p in result if p > 0))


def find_failed_pages_in_cache(cache_dir: Path) -> List[int]:
    if not cache_dir.exists():
        return []
    result: List[int] = []
    for meta_path in cache_dir.glob("page_*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("error"):
            try:
                result.append(int(meta.get("page", 0)))
            except Exception:
                continue
    return sorted(set(p for p in result if p > 0))


def is_quality_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "quality check failed" in message or "ocr quality" in message


def pdf_to_markdown(
    pdf_path: Path,
    out_path: Path,
    title: str,
    mode: str,
    providers: List[OCRProvider],
    models: List[str],
    dpi: int,
    dpi_high: int,
    adaptive_dpi: bool,
    text_min_chars: int,
    text_min_alnum_ratio: float,
    text_min_avg_line: float,
    text_max_blank_ratio: float,
    max_side: int,
    image_format: str,
    detail: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    max_continuations: int,
    max_retries: int,
    sleep_base: float,
    include_pages: str,
    skip_pages: str,
    quiet: bool,
    cache_dir: Path,
    resume: bool,
    retry_failed: bool,
    per_page_dir: Optional[Path],
    page_header_format: str,
    include_page_header: bool,
    page_header_in_pages: bool,
    separator: str,
    merge_output: bool,
    include_title: bool,
    ocr_workers: int,
    ocr_rps: float,
    page_retries: int,
    prompt_template: Optional[str],
    prompt_language: str,
    auto_lang: bool,
    stats_out: Optional[Path],
    config_out: Optional[Path],
    json_out: Optional[Path],
    continue_on_error: bool,
    dry_run: bool,
    cache_images: bool,
    render_workers: int,
    layout_mode: str,
    column_gap_ratio: float,
    column_min_words: int,
    column_left_ratio: float,
    column_right_ratio: float,
    column_split: float,
    column_overlap: float,
    table_mode: str,
    table_backend: str,
    table_flavor: str,
    table_header: bool,
    postprocess: bool,
    dehyphenate: bool,
    reflow: bool,
    remove_hf: bool,
    hf_top: int,
    hf_bottom: int,
    hf_min_ratio: float,
    hf_min_pages: int,
    hf_strip_digits: bool,
    hf_use_layout: bool,
    hf_top_margin: float,
    hf_bottom_margin: float,
    hf_font_max_ratio: float,
    hf_line_tolerance: float,
    remove_page_numbers: bool,
    warn_short: int,
    warn_prefix: str,
    empty_placeholder: str,
    pre_hook: Optional[str],
    post_hook: Optional[str],
    dedupe: bool,
    dedupe_min_ratio: float,
    dedupe_min_pages: int,
    dedupe_min_chars: int,
    dedupe_strip_digits: bool,
    replacements: List[Tuple[str, str]],
    replace_mode: str,
    replace_ignore_case: bool,
    auto_heading: bool,
    heading_level: int,
    heading_max_len: int,
    ocr_quality_check: bool,
    ocr_quality_min_chars: int,
    ocr_quality_min_alnum: float,
    ocr_quality_max_symbol: float,
    ocr_quality_max_replace: float,
    preprocess_enabled: bool,
    denoise: bool,
    denoise_size: int,
    contrast: float,
    binarize: bool,
    binarize_threshold: int,
    deskew: bool,
    deskew_max_angle: float,
    deskew_step: float,
    split_blocks: bool,
    block_gap_ratio: float,
    block_min_ratio: float,
    block_ink_ratio: float,
    block_white_threshold: int,
    block_max_blocks: int,
    ocr_max_pixels: int,
    ocr_skip_large: bool,
    ocr_min_side: int,
    config_snapshot: dict,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    logger = logging.getLogger("pdf_to_markdown")
    start_time = time.monotonic()
    start_iso = now_iso()

    cache_dir.mkdir(parents=True, exist_ok=True)
    if per_page_dir and per_page_dir.resolve() == cache_dir.resolve():
        if not quiet:
            print(
                "per-page-dir is the same as cache-dir; per-page headers are disabled."
            )
        per_page_dir = None

    stats_by_page: Dict[int, PageStat] = {}
    precheck_seconds: Dict[int, float] = {}
    ocr_pages: List[int] = []
    layout_by_page: Dict[int, LayoutDecision] = {}
    language_by_page: Dict[int, str] = {}
    plan_by_page: Dict[int, str] = {}
    header_sigs: Optional[set] = None
    footer_sigs: Optional[set] = None

    render_pool: Optional[ProcessPoolExecutor] = None
    if render_workers > 0:
        render_pool = ProcessPoolExecutor(max_workers=render_workers)

    try:
        logger.info("Starting conversion for %s", pdf_path)
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            width = max(len(str(total_pages)), 2)
            selected_pages = resolve_pages(total_pages, include_pages, skip_pages)
            if not selected_pages:
                raise RuntimeError(
                    "No pages selected. Check --pages/--skip-pages options."
                )
            logger.info(
                "Selected %d/%d pages (mode=%s, workers=%d)",
                len(selected_pages),
                total_pages,
                mode,
                ocr_workers,
            )

            if remove_hf and hf_use_layout:
                logger.info("Detecting headers/footers with layout heuristics")
                header_sigs, footer_sigs = detect_headers_footers_with_layout(
                    pdf,
                    selected_pages,
                    top_margin=hf_top_margin,
                    bottom_margin=hf_bottom_margin,
                    min_ratio=hf_min_ratio,
                    min_pages=hf_min_pages,
                    strip_digits=hf_strip_digits,
                    font_max_ratio=hf_font_max_ratio,
                    line_tolerance=hf_line_tolerance,
                )

            table_extractor = TableExtractor(table_backend, table_flavor, quiet)
            if table_mode != "off" and not table_extractor.available():
                message = "Table extraction backend not available. Install camelot or tabula-py."
                if table_mode == "only":
                    raise RuntimeError(message)
                if not quiet:
                    print(message)

            for page_number in selected_pages:
                page = pdf.pages[page_number - 1]
                layout_by_page[page_number] = decide_layout(
                    page,
                    layout_mode=layout_mode,
                    gap_ratio=column_gap_ratio,
                    min_words=column_min_words,
                    left_ratio=column_left_ratio,
                    right_ratio=column_right_ratio,
                    default_split=column_split,
                )

                cache_path = cache_dir / page_filename(page_number, width)
                cached_meta = (
                    read_page_meta(cache_dir, page_number, width)
                    if cache_path.exists()
                    else None
                )
                if (
                    resume
                    and cache_path.exists()
                    and not (retry_failed and cached_meta and cached_meta.get("error"))
                ):
                    cached_content = read_page_cache(cache_dir, page_number, width)
                    stats_by_page[page_number] = stat_from_meta(
                        cached_meta,
                        page_number,
                        cached_content,
                        cached=True,
                    )
                    plan_by_page[page_number] = "cached"
                    continue

                need_text = mode in ("auto", "text") or auto_lang
                text = ""
                if need_text:
                    logger.info("Extracting text for page %d", page_number)
                    page_start = time.monotonic()
                    text = extract_text_page(pdf, page_number - 1)
                    precheck_seconds[page_number] = time.monotonic() - page_start

                language_hint = prompt_language
                if not language_hint and auto_lang:
                    language_hint = detect_language_hint(text)
                language_by_page[page_number] = language_hint

                if mode == "text":
                    content = text or empty_placeholder
                    if not dry_run:
                        write_page_cache(cache_dir, page_number, width, content)
                        stat = PageStat(
                            page=page_number,
                            method="text",
                            seconds=precheck_seconds.get(page_number, 0.0),
                            chars=len(content),
                        )
                        write_page_meta(cache_dir, page_number, width, asdict(stat))
                        stats_by_page[page_number] = stat
                    plan_by_page[page_number] = "text"
                    continue

                if mode == "auto":
                    if looks_like_meaningful_text(
                        text,
                        min_chars=text_min_chars,
                        min_alnum_ratio=text_min_alnum_ratio,
                        min_avg_line_len=text_min_avg_line,
                        max_blank_ratio=text_max_blank_ratio,
                    ):
                        content = text or empty_placeholder
                        if not dry_run:
                            write_page_cache(cache_dir, page_number, width, content)
                            stat = PageStat(
                                page=page_number,
                                method="text",
                                seconds=precheck_seconds.get(page_number, 0.0),
                                chars=len(content),
                            )
                            write_page_meta(cache_dir, page_number, width, asdict(stat))
                            stats_by_page[page_number] = stat
                        plan_by_page[page_number] = "text"
                        continue

                ocr_pages.append(page_number)
                plan_by_page[page_number] = "ocr"

            text_plan = sum(1 for method in plan_by_page.values() if method == "text")
            ocr_plan = sum(1 for method in plan_by_page.values() if method == "ocr")
            cached_plan = sum(
                1 for method in plan_by_page.values() if method == "cached"
            )
            logger.info(
                "Plan ready: pages=%d, text=%d, ocr=%d, cached=%d",
                len(selected_pages),
                text_plan,
                ocr_plan,
                cached_plan,
            )

            if dry_run:
                for page_number in selected_pages:
                    method = plan_by_page.get(page_number, "skip")
                    print(f"Page {page_number}: {method}")
                print(
                    f"Summary: pages={len(selected_pages)}, text={text_plan}, ocr={ocr_plan}, cached={cached_plan}"
                )
                if ocr_plan and not providers:
                    print("Warning: OCR needed but no API key configured.")
                return

            if ocr_pages and not providers:
                raise RuntimeError(
                    "OCR required but no API key configured. "
                    "Set SILICONFLOW_API_KEY or pass --api-key."
                )

            if ocr_pages and table_mode != "off" and table_extractor.available():
                logger.info("Attempting table extraction for %d pages", len(ocr_pages))
                remaining_pages: List[int] = []
                for page_number in ocr_pages:
                    table_start = time.monotonic()
                    tables, backend = table_extractor.extract_tables(
                        pdf_path, page_number
                    )
                    if tables:
                        content = tables_to_markdown(tables, use_header=table_header)
                        write_page_cache(cache_dir, page_number, width, content)
                        total_seconds = precheck_seconds.get(page_number, 0.0) + (
                            time.monotonic() - table_start
                        )
                        stat = PageStat(
                            page=page_number,
                            method="table",
                            seconds=total_seconds,
                            chars=len(content),
                        )
                        write_page_meta(cache_dir, page_number, width, asdict(stat))
                        stats_by_page[page_number] = stat
                        if not quiet and backend:
                            print(f"Table extracted page {page_number} via {backend}")
                    else:
                        remaining_pages.append(page_number)
                ocr_pages = remaining_pages

            if table_mode == "only" and ocr_pages:
                for page_number in ocr_pages:
                    content = empty_placeholder
                    write_page_cache(cache_dir, page_number, width, content)
                    total_seconds = precheck_seconds.get(page_number, 0.0)
                    stat = PageStat(
                        page=page_number,
                        method="table",
                        seconds=total_seconds,
                        chars=len(content),
                    )
                    write_page_meta(cache_dir, page_number, width, asdict(stat))
                    stats_by_page[page_number] = stat
                ocr_pages = []

            if ocr_pages:
                rate_limiter = RateLimiter(ocr_rps) if ocr_rps > 0 else None
                workers = max(1, min(ocr_workers, len(ocr_pages)))
                logger.info(
                    "Running OCR on %d pages (workers=%d, rps=%.2f)",
                    len(ocr_pages),
                    workers,
                    ocr_rps,
                )

                def run_ocr_with_dpi(
                    page_number: int, dpi_value: int
                ) -> Tuple[str, str, str]:
                    image_path, delete_after = render_image_to_path(
                        pdf_path,
                        page_number,
                        dpi=dpi_value,
                        image_format=image_format,
                        cache_dir=cache_dir,
                        width=width,
                        cache_images=cache_images,
                        render_pool=render_pool,
                    )
                    with Image.open(image_path) as img:
                        image = img.copy()
                    if delete_after:
                        image_path.unlink(missing_ok=True)
                    layout = layout_by_page.get(
                        page_number, LayoutDecision(mode="single")
                    )
                    language_hint = language_by_page.get(page_number, prompt_language)
                    return ocr_image_by_layout(
                        image=image,
                        layout=layout,
                        providers=providers,
                        models=models,
                        page_number=page_number,
                        detail=detail,
                        timeout=timeout,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_continuations=max_continuations,
                        max_retries=max_retries,
                        sleep_base=sleep_base,
                        rate_limiter=rate_limiter,
                        prompt_template=prompt_template,
                        language_hint=language_hint,
                        image_format=image_format,
                        max_side=max_side,
                        column_overlap=column_overlap,
                        quality_check=ocr_quality_check,
                        quality_min_chars=ocr_quality_min_chars,
                        quality_min_alnum=ocr_quality_min_alnum,
                        quality_max_symbol=ocr_quality_max_symbol,
                        quality_max_replace=ocr_quality_max_replace,
                        preprocess_enabled=preprocess_enabled,
                        denoise=denoise,
                        denoise_size=denoise_size,
                        contrast=contrast,
                        binarize=binarize,
                        binarize_threshold=binarize_threshold,
                        deskew=deskew,
                        deskew_max_angle=deskew_max_angle,
                        deskew_step=deskew_step,
                        split_blocks=split_blocks,
                        block_gap_ratio=block_gap_ratio,
                        block_min_ratio=block_min_ratio,
                        block_ink_ratio=block_ink_ratio,
                        block_white_threshold=block_white_threshold,
                        block_max_blocks=block_max_blocks,
                        ocr_max_pixels=ocr_max_pixels,
                        ocr_skip_large=ocr_skip_large,
                        ocr_min_side=ocr_min_side,
                    )

                def ocr_worker(page_number: int) -> PageStat:
                    ocr_start = time.monotonic()
                    attempts = max(1, page_retries)
                    last_error: Optional[Exception] = None
                    content = ""
                    model_used = ""
                    api_url_used = ""
                    success = False

                    for attempt in range(1, attempts + 1):
                        try:
                            content, model_used, api_url_used = run_ocr_with_dpi(
                                page_number, dpi
                            )
                            success = True
                            break
                        except Exception as exc:
                            if (
                                adaptive_dpi
                                and is_quality_error(exc)
                                and dpi_high > dpi
                            ):
                                try:
                                    content, model_used, api_url_used = (
                                        run_ocr_with_dpi(page_number, dpi_high)
                                    )
                                    success = True
                                    break
                                except Exception as high_exc:
                                    last_error = high_exc
                            else:
                                last_error = exc

                            if attempt < attempts:
                                time.sleep(sleep_base * attempt)

                    ocr_seconds = time.monotonic() - ocr_start
                    total_seconds = precheck_seconds.get(page_number, 0.0) + ocr_seconds

                    if not success:
                        error_message = (
                            f"OCR failed after {attempts} retries: {last_error}"
                        )
                        content = (
                            f"[Error processing page {page_number}: {error_message}]"
                        )
                        write_page_cache(cache_dir, page_number, width, content)
                        stat = PageStat(
                            page=page_number,
                            method="error",
                            seconds=total_seconds,
                            chars=len(content),
                            error=error_message,
                        )
                        write_page_meta(cache_dir, page_number, width, asdict(stat))
                        return stat

                    write_page_cache(cache_dir, page_number, width, content)
                    stat = PageStat(
                        page=page_number,
                        method="ocr",
                        seconds=total_seconds,
                        chars=len(content),
                        model=model_used,
                        api_url=api_url_used,
                    )
                    write_page_meta(cache_dir, page_number, width, asdict(stat))
                    return stat

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_page = {
                        executor.submit(ocr_worker, page): page for page in ocr_pages
                    }
                    completed_pages = 0
                    for future in as_completed(future_to_page):
                        page_number = future_to_page[future]
                        try:
                            stat = future.result()
                            stats_by_page[page_number] = stat
                            completed_pages += 1
                            if not quiet:
                                if stat.method == "error":
                                    print(
                                        f"OCR page {page_number} failed: {stat.error}"
                                    )
                                else:
                                    print(
                                        f"OCR page {page_number} done in {stat.seconds:.1f}s"
                                    )
                            if stat.method == "error":
                                logger.error(
                                    "OCR page %d failed: %s", page_number, stat.error
                                )
                            else:
                                logger.info(
                                    "OCR page %d done in %.1fs",
                                    page_number,
                                    stat.seconds,
                                )
                            logger.info(
                                "OCR progress: %d/%d pages",
                                completed_pages,
                                len(ocr_pages),
                            )
                        except Exception as exc:
                            error_message = str(exc)
                            if not continue_on_error:
                                raise
                            content = f"[Error processing page {page_number}: {error_message}]"
                            write_page_cache(cache_dir, page_number, width, content)
                            stat = PageStat(
                                page=page_number,
                                method="error",
                                seconds=precheck_seconds.get(page_number, 0.0),
                                chars=len(content),
                                error=error_message,
                            )
                            write_page_meta(cache_dir, page_number, width, asdict(stat))
                            stats_by_page[page_number] = stat
                            if not quiet:
                                print(f"OCR page {page_number} failed: {error_message}")
                            logger.error(
                                "OCR page %d failed: %s", page_number, error_message
                            )
                            completed_pages += 1
                            logger.info(
                                "OCR progress: %d/%d pages",
                                completed_pages,
                                len(ocr_pages),
                            )

        raw_pages = {
            page: read_page_cache(cache_dir, page, width) for page in selected_pages
        }
        methods_by_page = {
            page: stats_by_page.get(page, PageStat(page, "unknown", 0.0, 0)).method
            for page in selected_pages
        }

        cleaned_pages, warnings, dedupe_removed = postprocess_pages(
            raw_pages=raw_pages,
            methods_by_page=methods_by_page,
            pdf_path=pdf_path,
            postprocess=postprocess,
            dehyphenate=dehyphenate,
            reflow=reflow,
            remove_hf=remove_hf,
            hf_top=hf_top,
            hf_bottom=hf_bottom,
            hf_min_ratio=hf_min_ratio,
            hf_min_pages=hf_min_pages,
            hf_strip_digits=hf_strip_digits,
            remove_page_numbers=remove_page_numbers,
            warn_short=warn_short,
            warn_prefix=warn_prefix,
            empty_placeholder=empty_placeholder,
            pre_hook=pre_hook,
            post_hook=post_hook,
            continue_on_error=continue_on_error,
            quiet=quiet,
            header_sigs=header_sigs,
            footer_sigs=footer_sigs,
            dedupe=dedupe,
            dedupe_min_ratio=dedupe_min_ratio,
            dedupe_min_pages=dedupe_min_pages,
            dedupe_min_chars=dedupe_min_chars,
            dedupe_strip_digits=dedupe_strip_digits,
            replacements=replacements,
            replace_mode=replace_mode,
            replace_ignore_case=replace_ignore_case,
            auto_heading=auto_heading,
            heading_level=heading_level,
            heading_max_len=heading_max_len,
        )

        for page_number, warning in warnings.items():
            if warning and page_number in stats_by_page:
                stats_by_page[page_number].warning = warning

        if merge_output:
            logger.info("Writing merged output to %s", out_path)
            merge_pages(
                out_path=out_path,
                title=title,
                selected_pages=selected_pages,
                content_by_page=cleaned_pages,
                separator=separator,
                page_header_format=page_header_format,
                include_page_header=include_page_header,
                include_title=include_title,
            )

        if per_page_dir:
            logger.info("Writing per-page output to %s", per_page_dir)
            for page_number in selected_pages:
                content = cleaned_pages.get(page_number, "")
                write_per_page_output(
                    per_page_dir,
                    page_number,
                    width,
                    content,
                    page_header_format,
                    page_header_in_pages,
                )

        elapsed_total = time.monotonic() - start_time
        stats_list = [stats_by_page[p] for p in sorted(stats_by_page)]
        text_pages = sum(1 for s in stats_list if s.method == "text")
        table_pages = sum(1 for s in stats_list if s.method == "table")
        ocr_pages_count = sum(1 for s in stats_list if s.method == "ocr")
        cached_pages = sum(1 for s in stats_list if s.cached)
        error_pages = [s.page for s in stats_list if s.error]

        summary = {
            "total_pages": len(selected_pages),
            "text_pages": text_pages,
            "table_pages": table_pages,
            "ocr_pages": ocr_pages_count,
            "cached_pages": cached_pages,
            "error_pages": error_pages,
            "dedupe_removed": dedupe_removed,
            "elapsed_seconds": elapsed_total,
        }

        if not quiet:
            processed = text_pages + ocr_pages_count + table_pages
            ocr_ratio = (ocr_pages_count / processed) if processed else 0.0
            print(
                "Summary: "
                f"pages={len(selected_pages)}, text={text_pages}, table={table_pages}, ocr={ocr_pages_count}, "
                f"cached={cached_pages}, errors={len(error_pages)}, ocr_ratio={ocr_ratio:.1%}"
            )
            if dedupe_removed:
                print(f"Deduped paragraphs: {dedupe_removed}")
            if error_pages:
                print(f"Failed pages: {error_pages}")
            if merge_output:
                print(f"Output: {out_path}")
            if per_page_dir:
                print(f"Per-page output: {per_page_dir}")
            print(f"Cache: {cache_dir}")
        logger.info(
            "Completed conversion in %.1fs (pages=%d, errors=%d)",
            elapsed_total,
            len(selected_pages),
            len(error_pages),
        )

        if stats_out:
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": config_snapshot,
                "pdf": str(pdf_path),
                "output": str(out_path) if merge_output else None,
                "cache_dir": str(cache_dir),
                "selected_pages": selected_pages,
                "start_time": start_iso,
                "end_time": now_iso(),
                "summary": summary,
                "pages": [asdict(stat) for stat in stats_list],
            }
            stats_out.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )

        if json_out:
            write_json_output(
                json_out=json_out,
                config=config_snapshot,
                selected_pages=selected_pages,
                cleaned_pages=cleaned_pages,
                stats_by_page=stats_by_page,
                summary=summary,
            )

        if config_out:
            config_out.parent.mkdir(parents=True, exist_ok=True)
            config_out.write_text(
                json.dumps(config_snapshot, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

    finally:
        if render_pool:
            render_pool.shutdown(wait=True)


def build_parser() -> argparse.ArgumentParser:
    default_render_workers = max(1, (os.cpu_count() or 2))
    default_ocr_workers = max(2, (os.cpu_count() or 2))
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using text extraction with optional OCR fallback."
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--out", default=None, help="Output Markdown path")
    parser.add_argument(
        "--title", default=None, help="Top-level Markdown title (default: PDF filename)"
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "text", "ocr"],
        help="auto=extract text then OCR fallback, text=extract only, ocr=OCR only",
    )
    parser.add_argument(
        "--pages", default="", help="Process only these pages, e.g. 1,3-5"
    )
    parser.add_argument("--skip-pages", default="", help="Skip pages, e.g. 2,10-12")
    parser.add_argument(
        "--api-key", default=None, help="API key(s), comma-separated for multiple URLs"
    )
    parser.add_argument(
        "--api-url", default=API_URL, help=f"OCR API URL (default: {API_URL})"
    )
    parser.add_argument(
        "--api-url-fallback",
        default=None,
        help="Fallback OCR API URLs (comma-separated)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="VLM model(s), comma-separated"
    )
    parser.add_argument(
        "--model-fallback", default=None, help="Fallback model names (comma-separated)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PDF render DPI for OCR pages"
    )
    parser.add_argument(
        "--dpi-high", type=int, default=450, help="High DPI for adaptive OCR retries"
    )
    parser.add_argument(
        "--adaptive-dpi",
        dest="adaptive_dpi",
        action="store_true",
        help="Retry OCR with higher DPI on quality failure",
    )
    parser.add_argument(
        "--no-adaptive-dpi",
        dest="adaptive_dpi",
        action="store_false",
        help="Disable adaptive DPI",
    )
    parser.set_defaults(adaptive_dpi=False)
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=200,
        help="Min chars to accept extracted text",
    )
    parser.add_argument(
        "--text-min-alnum-ratio",
        type=float,
        default=0.25,
        help="Min alnum ratio to accept extracted text",
    )
    parser.add_argument(
        "--text-min-avg-line",
        type=float,
        default=12.0,
        help="Min average line length to accept extracted text",
    )
    parser.add_argument(
        "--text-max-blank-ratio",
        type=float,
        default=0.6,
        help="Max blank line ratio to accept extracted text",
    )
    parser.add_argument(
        "--max-side", type=int, default=2000, help="Max side length for OCR image"
    )
    parser.add_argument(
        "--imgfmt", default="PNG", choices=["PNG", "JPG", "JPEG"], help="Image format"
    )
    parser.add_argument(
        "--detail", default="high", choices=["low", "high"], help="Image detail hint"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="API request timeout in seconds"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per OCR call"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-continuations",
        type=int,
        default=2,
        help="If truncated, continue this many times",
    )
    parser.add_argument("--max-retries", type=int, default=4, help="Max API retries")
    parser.add_argument(
        "--page-retries", type=int, default=3, help="Max OCR page retries"
    )
    parser.add_argument(
        "--sleep-base", type=float, default=1.2, help="Retry backoff base seconds"
    )
    parser.add_argument(
        "--separator", default=DEFAULT_SEPARATOR, help="Separator between pages"
    )
    parser.add_argument(
        "--page-header", default="## Page {page}", help="Page header format"
    )
    parser.add_argument(
        "--no-page-header", action="store_true", help="Disable page headers"
    )
    parser.add_argument(
        "--page-header-in-pages",
        action="store_true",
        help="Include page headers in per-page output files",
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Disable top-level title"
    )
    parser.add_argument(
        "--per-page-dir", default=None, help="Directory for per-page Markdown files"
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="Skip merged output file"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Directory for cached per-page results (default: {DEFAULT_CACHE_DIRNAME}/<pdf>)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse cached page results if present"
    )
    parser.add_argument(
        "--cache-images", action="store_true", help="Cache rendered page images for OCR"
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=default_render_workers,
        help="Render workers (0=disable)",
    )
    parser.add_argument(
        "--ocr-workers", type=int, default=default_ocr_workers, help="OCR concurrency"
    )
    parser.add_argument(
        "--ocr-rps", type=float, default=0.0, help="OCR requests per second (0=disable)"
    )
    parser.add_argument(
        "--prompt-file", default=None, help="Prompt template file for OCR"
    )
    parser.add_argument(
        "--prompt-lang", default="", help="Language hint for OCR prompt"
    )
    parser.add_argument(
        "--auto-lang",
        dest="auto_lang",
        action="store_true",
        help="Auto-detect OCR language",
    )
    parser.add_argument(
        "--no-auto-lang",
        dest="auto_lang",
        action="store_false",
        help="Disable OCR language auto-detection",
    )
    parser.set_defaults(auto_lang=True)
    parser.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "single", "columns"],
        help="OCR layout",
    )
    parser.add_argument(
        "--column-gap-ratio", type=float, default=0.08, help="Min column gap ratio"
    )
    parser.add_argument(
        "--column-min-words", type=int, default=20, help="Min words to detect columns"
    )
    parser.add_argument(
        "--column-left-ratio", type=float, default=0.45, help="Left column max ratio"
    )
    parser.add_argument(
        "--column-right-ratio", type=float, default=0.55, help="Right column min ratio"
    )
    parser.add_argument(
        "--column-split", type=float, default=0.5, help="Split ratio for forced columns"
    )
    parser.add_argument(
        "--column-overlap",
        type=float,
        default=0.02,
        help="Overlap ratio for column OCR",
    )
    parser.add_argument(
        "--split-blocks",
        action="store_true",
        help="Split page into blocks for OCR",
    )
    parser.add_argument(
        "--block-gap-ratio",
        type=float,
        default=0.02,
        help="Min gap ratio for block split",
    )
    parser.add_argument(
        "--block-min-ratio", type=float, default=0.08, help="Min block height ratio"
    )
    parser.add_argument(
        "--block-ink-ratio", type=float, default=0.01, help="Max ink ratio for gaps"
    )
    parser.add_argument(
        "--block-white-threshold",
        type=int,
        default=245,
        help="White threshold for gaps",
    )
    parser.add_argument(
        "--block-max-blocks", type=int, default=6, help="Max blocks per page"
    )
    parser.add_argument(
        "--table-mode",
        default="off",
        choices=["off", "auto", "only"],
        help="Table extraction mode",
    )
    parser.add_argument(
        "--table-backend",
        default="auto",
        choices=["auto", "camelot", "tabula"],
        help="Table extraction backend",
    )
    parser.add_argument(
        "--table-flavor",
        default="lattice",
        choices=["lattice", "stream"],
        help="Table extraction flavor",
    )
    parser.add_argument(
        "--table-header",
        dest="table_header",
        action="store_true",
        help="Use first row as table header",
    )
    parser.add_argument(
        "--no-table-header",
        dest="table_header",
        action="store_false",
        help="Do not treat the first row as header",
    )
    parser.set_defaults(table_header=True)
    parser.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Enable post-processing (default)",
    )
    parser.add_argument(
        "--no-postprocess",
        dest="postprocess",
        action="store_false",
        help="Disable post-processing",
    )
    parser.set_defaults(postprocess=True)
    parser.add_argument(
        "--dehyphenate",
        dest="dehyphenate",
        action="store_true",
        help="Merge hyphenated lines",
    )
    parser.add_argument(
        "--no-dehyphenate",
        dest="dehyphenate",
        action="store_false",
        help="Disable dehyphenate",
    )
    parser.set_defaults(dehyphenate=True)
    parser.add_argument(
        "--reflow", dest="reflow", action="store_true", help="Reflow paragraphs"
    )
    parser.add_argument(
        "--no-reflow", dest="reflow", action="store_false", help="Disable reflow"
    )
    parser.set_defaults(reflow=True)
    parser.add_argument(
        "--remove-hf",
        dest="remove_hf",
        action="store_true",
        help="Remove repeated headers/footers",
    )
    parser.add_argument(
        "--no-remove-hf",
        dest="remove_hf",
        action="store_false",
        help="Disable header/footer removal",
    )
    parser.set_defaults(remove_hf=True)
    parser.add_argument(
        "--hf-top", type=int, default=2, help="Header line count to inspect"
    )
    parser.add_argument(
        "--hf-bottom", type=int, default=2, help="Footer line count to inspect"
    )
    parser.add_argument(
        "--hf-min-ratio",
        type=float,
        default=0.6,
        help="Min repeat ratio for header/footer",
    )
    parser.add_argument(
        "--hf-min-pages", type=int, default=3, help="Min pages for header/footer"
    )
    parser.add_argument(
        "--hf-strip-digits",
        dest="hf_strip_digits",
        action="store_true",
        help="Strip digits when matching headers/footers",
    )
    parser.add_argument(
        "--no-hf-strip-digits",
        dest="hf_strip_digits",
        action="store_false",
        help="Keep digits when matching headers/footers",
    )
    parser.set_defaults(hf_strip_digits=True)
    parser.add_argument(
        "--hf-use-layout",
        dest="hf_use_layout",
        action="store_true",
        help="Use layout-aware header/footer detection",
    )
    parser.add_argument(
        "--no-hf-use-layout",
        dest="hf_use_layout",
        action="store_false",
        help="Disable layout-aware header/footer detection",
    )
    parser.set_defaults(hf_use_layout=True)
    parser.add_argument(
        "--hf-top-margin", type=float, default=0.12, help="Header margin ratio"
    )
    parser.add_argument(
        "--hf-bottom-margin", type=float, default=0.12, help="Footer margin ratio"
    )
    parser.add_argument(
        "--hf-font-max-ratio",
        type=float,
        default=0.9,
        help="Max font size ratio for HF",
    )
    parser.add_argument(
        "--hf-line-tolerance", type=float, default=3.0, help="Line grouping tolerance"
    )
    parser.add_argument(
        "--remove-page-number",
        dest="remove_page_numbers",
        action="store_true",
        help="Remove page number lines in header/footer",
    )
    parser.add_argument(
        "--no-remove-page-number",
        dest="remove_page_numbers",
        action="store_false",
        help="Keep page number lines",
    )
    parser.set_defaults(remove_page_numbers=True)
    parser.add_argument(
        "--warn-short",
        type=int,
        default=0,
        help="Warn if page content is shorter than this",
    )
    parser.add_argument("--warn-prefix", default="[WARN]", help="Warning prefix text")
    parser.add_argument(
        "--empty-placeholder",
        default="[Empty page]",
        help="Placeholder for empty pages",
    )
    parser.add_argument(
        "--pre-hook", default=None, help="Command to preprocess each page content"
    )
    parser.add_argument(
        "--post-hook", default=None, help="Command to postprocess each page content"
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="Remove repeated paragraphs across pages"
    )
    parser.add_argument(
        "--dedupe-min-ratio", type=float, default=0.6, help="Dedupe min ratio"
    )
    parser.add_argument(
        "--dedupe-min-pages", type=int, default=3, help="Dedupe min pages"
    )
    parser.add_argument(
        "--dedupe-min-chars", type=int, default=80, help="Dedupe min paragraph length"
    )
    parser.add_argument(
        "--dedupe-strip-digits",
        dest="dedupe_strip_digits",
        action="store_true",
        help="Strip digits in paragraph signatures",
    )
    parser.add_argument(
        "--no-dedupe-strip-digits",
        dest="dedupe_strip_digits",
        action="store_false",
        help="Keep digits in paragraph signatures",
    )
    parser.set_defaults(dedupe_strip_digits=True)
    parser.add_argument(
        "--replace-file", default=None, help="Replacement dictionary file"
    )
    parser.add_argument(
        "--replace-mode",
        default="plain",
        choices=["plain", "word"],
        help="Replace mode",
    )
    parser.add_argument(
        "--replace-ignore-case",
        action="store_true",
        help="Case-insensitive replacements",
    )
    parser.add_argument(
        "--auto-headings", action="store_true", help="Auto-detect headings"
    )
    parser.add_argument(
        "--heading-level", type=int, default=2, help="Base heading level"
    )
    parser.add_argument(
        "--heading-max-len", type=int, default=80, help="Max heading line length"
    )
    parser.add_argument(
        "--ocr-quality",
        dest="ocr_quality",
        action="store_true",
        help="Enable OCR quality checks",
    )
    parser.add_argument(
        "--no-ocr-quality",
        dest="ocr_quality",
        action="store_false",
        help="Disable OCR quality checks",
    )
    parser.set_defaults(ocr_quality=True)
    parser.add_argument("--ocr-min-chars", type=int, default=80, help="OCR min chars")
    parser.add_argument(
        "--ocr-min-alnum-ratio", type=float, default=0.2, help="OCR min alnum ratio"
    )
    parser.add_argument(
        "--ocr-max-symbol-ratio", type=float, default=0.6, help="OCR max symbol ratio"
    )
    parser.add_argument(
        "--ocr-max-replace-ratio",
        type=float,
        default=0.02,
        help="OCR max replacement ratio",
    )
    parser.add_argument(
        "--ocr-max-pixels", type=int, default=0, help="Max pixels for OCR input"
    )
    parser.add_argument(
        "--ocr-min-side", type=int, default=600, help="Min side length when downscaling"
    )
    parser.add_argument(
        "--ocr-skip-large", action="store_true", help="Skip OCR on large images"
    )
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Enable OCR image preprocessing (default)",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Disable OCR image preprocessing",
    )
    parser.set_defaults(preprocess=True)
    parser.add_argument(
        "--denoise", dest="denoise", action="store_true", help="Enable denoise"
    )
    parser.add_argument(
        "--no-denoise", dest="denoise", action="store_false", help="Disable denoise"
    )
    parser.set_defaults(denoise=True)
    parser.add_argument(
        "--denoise-size", type=int, default=3, help="Denoise filter size"
    )
    parser.add_argument(
        "--contrast", type=float, default=1.4, help="Contrast enhancement factor"
    )
    parser.add_argument(
        "--binarize", dest="binarize", action="store_true", help="Enable binarization"
    )
    parser.add_argument(
        "--no-binarize",
        dest="binarize",
        action="store_false",
        help="Disable binarization",
    )
    parser.set_defaults(binarize=True)
    parser.add_argument(
        "--binarize-threshold",
        type=int,
        default=0,
        help="Binarization threshold (0=auto)",
    )
    parser.add_argument(
        "--deskew", dest="deskew", action="store_true", help="Enable deskew"
    )
    parser.add_argument(
        "--no-deskew", dest="deskew", action="store_false", help="Disable deskew"
    )
    parser.set_defaults(deskew=True)
    parser.add_argument(
        "--deskew-max-angle", type=float, default=2.0, help="Deskew max angle"
    )
    parser.add_argument(
        "--deskew-step", type=float, default=0.5, help="Deskew angle step"
    )
    parser.add_argument(
        "--stats-out", default=None, help="Write run stats JSON to this path"
    )
    parser.add_argument(
        "--config-out", default=None, help="Write config snapshot JSON to this path"
    )
    parser.add_argument(
        "--json-out", default=None, help="Write structured output JSON to this path"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry failed pages in cache"
    )
    parser.add_argument(
        "--retry-failed-from", default=None, help="Retry failed pages from stats JSON"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Deterministic output mode"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue on page errors"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List planned page strategies and exit"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=sorted(LOG_LEVELS.keys()),
        help="Log verbosity for progress output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.quiet, args.log_level)

    pdf_path = Path(args.pdf)
    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".md")
    title = args.title or pdf_path.stem
    mode = args.mode

    if args.deterministic:
        args.temperature = 0.0
        args.ocr_workers = 1
        args.render_workers = 1
        args.ocr_rps = 0.0
        args.page_retries = 1

    env_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    api_urls = parse_api_urls(args.api_url, args.api_url_fallback)
    api_keys = parse_api_keys(args.api_key, env_key, len(api_urls))
    providers = build_providers(api_urls, api_keys)
    models = parse_models(args.model, args.model_fallback)

    separator = decode_escaped(args.separator)
    include_page_header = not args.no_page_header
    include_title = not args.no_title
    merge_output = not args.no_merge

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else pdf_path.parent / DEFAULT_CACHE_DIRNAME / pdf_path.stem
    )
    per_page_dir = Path(args.per_page_dir) if args.per_page_dir else None

    retry_pages: List[int] = []
    if args.retry_failed_from:
        retry_pages = load_failed_pages_from_stats(Path(args.retry_failed_from))
    elif args.retry_failed:
        retry_pages = find_failed_pages_in_cache(cache_dir)
    if retry_pages:
        args.pages = ",".join(str(p) for p in retry_pages)

    prompt_template = load_prompt_template(args.prompt_file)
    replacements = load_replacements(args.replace_file)

    config_snapshot = build_config_snapshot(
        args,
        pdf_path=pdf_path,
        out_path=out_path,
        cache_dir=cache_dir,
        models=models,
        api_urls=api_urls,
        providers=providers,
    )

    pdf_to_markdown(
        pdf_path=pdf_path,
        out_path=out_path,
        title=title,
        mode=mode,
        providers=providers,
        models=models,
        dpi=args.dpi,
        dpi_high=args.dpi_high,
        adaptive_dpi=args.adaptive_dpi,
        text_min_chars=args.text_min_chars,
        text_min_alnum_ratio=args.text_min_alnum_ratio,
        text_min_avg_line=args.text_min_avg_line,
        text_max_blank_ratio=args.text_max_blank_ratio,
        max_side=args.max_side,
        image_format=args.imgfmt,
        detail=args.detail,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_continuations=args.max_continuations,
        max_retries=args.max_retries,
        sleep_base=args.sleep_base,
        include_pages=args.pages,
        skip_pages=args.skip_pages,
        quiet=args.quiet,
        cache_dir=cache_dir,
        resume=args.resume,
        retry_failed=bool(retry_pages),
        per_page_dir=per_page_dir,
        page_header_format=args.page_header,
        include_page_header=include_page_header,
        page_header_in_pages=args.page_header_in_pages,
        separator=separator,
        merge_output=merge_output,
        include_title=include_title,
        ocr_workers=args.ocr_workers,
        ocr_rps=args.ocr_rps,
        page_retries=args.page_retries,
        prompt_template=prompt_template,
        prompt_language=args.prompt_lang,
        auto_lang=args.auto_lang,
        stats_out=Path(args.stats_out) if args.stats_out else None,
        config_out=Path(args.config_out) if args.config_out else None,
        json_out=Path(args.json_out) if args.json_out else None,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        cache_images=args.cache_images,
        render_workers=args.render_workers,
        layout_mode=args.layout,
        column_gap_ratio=args.column_gap_ratio,
        column_min_words=args.column_min_words,
        column_left_ratio=args.column_left_ratio,
        column_right_ratio=args.column_right_ratio,
        column_split=args.column_split,
        column_overlap=args.column_overlap,
        table_mode=args.table_mode,
        table_backend=args.table_backend,
        table_flavor=args.table_flavor,
        table_header=args.table_header,
        postprocess=args.postprocess,
        dehyphenate=args.dehyphenate,
        reflow=args.reflow,
        remove_hf=args.remove_hf,
        hf_top=args.hf_top,
        hf_bottom=args.hf_bottom,
        hf_min_ratio=args.hf_min_ratio,
        hf_min_pages=args.hf_min_pages,
        hf_strip_digits=args.hf_strip_digits,
        hf_use_layout=args.hf_use_layout,
        hf_top_margin=args.hf_top_margin,
        hf_bottom_margin=args.hf_bottom_margin,
        hf_font_max_ratio=args.hf_font_max_ratio,
        hf_line_tolerance=args.hf_line_tolerance,
        remove_page_numbers=args.remove_page_numbers,
        warn_short=args.warn_short,
        warn_prefix=args.warn_prefix,
        empty_placeholder=args.empty_placeholder,
        pre_hook=args.pre_hook,
        post_hook=args.post_hook,
        dedupe=args.dedupe,
        dedupe_min_ratio=args.dedupe_min_ratio,
        dedupe_min_pages=args.dedupe_min_pages,
        dedupe_min_chars=args.dedupe_min_chars,
        dedupe_strip_digits=args.dedupe_strip_digits,
        replacements=replacements,
        replace_mode=args.replace_mode,
        replace_ignore_case=args.replace_ignore_case,
        auto_heading=args.auto_headings,
        heading_level=args.heading_level,
        heading_max_len=args.heading_max_len,
        ocr_quality_check=args.ocr_quality,
        ocr_quality_min_chars=args.ocr_min_chars,
        ocr_quality_min_alnum=args.ocr_min_alnum_ratio,
        ocr_quality_max_symbol=args.ocr_max_symbol_ratio,
        ocr_quality_max_replace=args.ocr_max_replace_ratio,
        preprocess_enabled=args.preprocess,
        denoise=args.denoise,
        denoise_size=args.denoise_size,
        contrast=args.contrast,
        binarize=args.binarize,
        binarize_threshold=args.binarize_threshold,
        deskew=args.deskew,
        deskew_max_angle=args.deskew_max_angle,
        deskew_step=args.deskew_step,
        split_blocks=args.split_blocks,
        block_gap_ratio=args.block_gap_ratio,
        block_min_ratio=args.block_min_ratio,
        block_ink_ratio=args.block_ink_ratio,
        block_white_threshold=args.block_white_threshold,
        block_max_blocks=args.block_max_blocks,
        ocr_max_pixels=args.ocr_max_pixels,
        ocr_skip_large=args.ocr_skip_large,
        ocr_min_side=args.ocr_min_side,
        config_snapshot=config_snapshot,
    )


if __name__ == "__main__":
    main()
