#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List


DEFAULT_MAX_LABELS = 80
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def _read_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_label(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.upper()


def _split_labels(raw: str) -> List[str]:
    labels = [item.strip() for item in re.split(r"[,\n]+", raw or "") if item.strip()]
    return labels


def _extract_candidates(text: str) -> Iterable[str]:
    # Capture acronyms like NFC, CCC, ISO-15118
    for match in re.finditer(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", text):
        yield match.group(0)
    # Capture capitalized phrases like "Vehicle OEM Server"
    phrase_regex = r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,4})\b"
    for match in re.finditer(phrase_regex, text):
        yield match.group(0)


def generate_labels_heuristic(text: str, max_labels: int, min_len: int) -> List[str]:
    counter: Counter[str] = Counter()
    for cand in _extract_candidates(text):
        label = _normalize_label(cand)
        if len(label) < min_len:
            continue
        if not re.search(r"[A-Z]", label):
            continue
        counter[label] += 1
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [label for label, _ in ranked[:max_labels]]


def _strip_fences(raw: str) -> str:
    content = raw.strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return content


def generate_labels_llm(
    text: str,
    model: str,
    max_labels: int,
    api_key: str | None,
    base_url: str | None,
    max_chars: int,
) -> List[str]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai is not installed. pip install openai") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (or pass --llm-api-key).")

    trimmed = text.strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars]

    prompt = (
        "Extract domain-specific entity labels from the text. "
        f"Return a JSON array of up to {max_labels} labels. "
        "Labels must be uppercase snake_case, concise, and reusable. "
        "Only return the JSON array."
    )

    client = OpenAI(api_key=key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": trimmed},
        ],
        temperature=0,
        max_tokens=512,
    )
    content = response.choices[0].message.content or "[]"
    content = _strip_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _split_labels(content)
    labels: List[str] = []
    if isinstance(data, dict):
        data = data.get("labels", [])
    if isinstance(data, list):
        for item in data:
            label = _normalize_label(str(item))
            if label:
                labels.append(label)
    return labels[:max_labels]


def _read_existing_labels(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    labels = []
    seen = set()
    for item in _split_labels(raw):
        normalized = _normalize_label(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            labels.append(normalized)
    return labels


def _write_labels(path: Path, labels: List[str], merge: bool) -> None:
    if merge:
        existing = _read_existing_labels(path)
        seen = set(existing)
        merged = list(existing)
        for item in labels:
            normalized = _normalize_label(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
        labels = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to PDF or text file")
    parser.add_argument("--text", help="Raw text input")
    parser.add_argument("--output", default=None, help="Write labels to this file")
    parser.add_argument("--merge", action="store_true", help="Merge with existing output file")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Label generation mode",
    )
    parser.add_argument("--max-labels", type=int, default=DEFAULT_MAX_LABELS)
    parser.add_argument("--min-label-len", type=int, default=4)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-max-chars", type=int, default=12000)
    args = parser.parse_args()

    if not args.input and not args.text:
        raise SystemExit("Provide --input or --text.")

    if args.text:
        text = args.text
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.suffix.lower() == ".pdf":
            text = _read_pdf_text(input_path)
        else:
            text = _read_text_file(input_path)

    if args.mode == "llm":
        labels = generate_labels_llm(
            text,
            model=args.llm_model,
            max_labels=args.max_labels,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            max_chars=args.llm_max_chars,
        )
    else:
        labels = generate_labels_heuristic(text, args.max_labels, args.min_label_len)

    if args.output:
        _write_labels(Path(args.output), labels, merge=args.merge)
    else:
        print("\n".join(labels))


if __name__ == "__main__":
    main()
