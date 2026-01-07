import json
import logging
import os
import re
import textwrap
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_GLINER_LABELS = [
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "DATE",
    "TECH",
    "CRYPTO",
    "STANDARD",
]


def _find_span(text: str, name: str) -> tuple[int, int] | None:
    if not text or not name:
        return None
    lower_text = text.lower()
    lower_name = name.lower()
    start = lower_text.find(lower_name)
    if start < 0:
        return None
    return start, start + len(name)


def _normalize_entities(
    entities: List[Dict[str, Any]], text: str | None = None
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in entities or []:
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name:
            continue
        confidence = item.get("confidence")
        if confidence is None:
            confidence = item.get("score")
        start_char = item.get("start_char")
        if start_char is None:
            start_char = item.get("start")
        end_char = item.get("end_char")
        if end_char is None:
            end_char = item.get("end")
        if (start_char is None or end_char is None) and text:
            span = _find_span(text, name)
            if span:
                start_char, end_char = span
        payload: Dict[str, Any] = {"name": name, "type": etype or "UNKNOWN"}
        if confidence is not None:
            payload["confidence"] = float(confidence)
        if start_char is not None and end_char is not None:
            payload["start_char"] = int(start_char)
            payload["end_char"] = int(end_char)
        normalized.append(payload)
    return normalized


def _normalize_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for item in relations or []:
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        rel = str(item.get("relation", "")).strip()
        if not source or not target:
            continue
        normalized.append({"source": source, "target": target, "relation": rel or "RELATED"})
    return normalized


def _env_flag(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_attr(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def _debug_llm_output() -> bool:
    return os.getenv("LLM_DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _setup_llm_debug_logging() -> None:
    if not _debug_llm_output():
        return
    logger = logging.getLogger()
    if getattr(logger, "_llm_debug_configured", False):
        return
    log_path = os.getenv("LLM_DEBUG_LOG_PATH", "logs/langextract_raw.log")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger._llm_debug_configured = True


def _get_retry_settings() -> tuple[int, float]:
    attempts = os.getenv("LLM_RETRY_COUNT", "3")
    backoff = os.getenv("LLM_RETRY_BACKOFF_SECONDS", "2")
    try:
        attempts_val = max(1, int(attempts))
    except ValueError:
        attempts_val = 3
    try:
        backoff_val = max(0.0, float(backoff))
    except ValueError:
        backoff_val = 2.0
    return attempts_val, backoff_val


def build_spacy_pipeline(model: str, ruler_json: str | None = None):
    import spacy

    nlp = spacy.load(model)
    if ruler_json:
        ruler = nlp.add_pipe("entity_ruler", last=True)
        ruler.overwrite_ents = False
        path = Path(ruler_json)
        if not path.exists():
            raise FileNotFoundError(path)

        def load_patterns(file_path: Path) -> list:
            try:
                data = json.loads(file_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid EntityRuler JSON: {file_path}") from exc
            patterns = data.get("patterns") if isinstance(data, dict) else data
            if not isinstance(patterns, list):
                raise ValueError(
                    f"EntityRuler JSON must be a list or {{\"patterns\": [...]}} format: {file_path}"
                )
            if not patterns:
                raise ValueError(f"EntityRuler JSON has no patterns: {file_path}")
            for idx, item in enumerate(patterns):
                if not isinstance(item, dict) or "label" not in item or "pattern" not in item:
                    raise ValueError(f"Invalid pattern at index {idx} in {file_path}: {item}")
            return patterns

        all_patterns = []
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found in rules directory: {path}")
            for file_path in json_files:
                all_patterns.extend(load_patterns(file_path))
        else:
            all_patterns = load_patterns(path)

        ruler.add_patterns(all_patterns)
    return nlp


def parse_gliner_labels(raw: str | None) -> List[str]:
    if raw is None:
        return list(DEFAULT_GLINER_LABELS)
    candidate = raw.strip()
    if not candidate:
        return list(DEFAULT_GLINER_LABELS)
    path = Path(candidate)
    if path.exists() and path.is_file():
        content = path.read_text(encoding="utf-8")
        labels = [item.strip() for item in re.split(r"[,\n]+", content) if item.strip()]
        return labels or list(DEFAULT_GLINER_LABELS)
    labels = [item.strip() for item in candidate.split(",") if item.strip()]
    return labels or list(DEFAULT_GLINER_LABELS)


def _resolve_gliner_model(model: str) -> str:
    if model and Path(model).exists():
        return model
    env_path = os.getenv("GLINER_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    return model


def _gliner_local_only(resolved_model: str) -> bool:
    if resolved_model and Path(resolved_model).exists():
        return True
    offline = os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE")
    if offline and offline.strip().lower() not in {"0", "false", "no"}:
        return True
    local_only = os.getenv("GLINER_LOCAL_ONLY")
    if local_only and local_only.strip().lower() not in {"0", "false", "no"}:
        return True
    return False


@lru_cache(maxsize=2)
def build_gliner_model(model: str):
    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError("gliner is not installed. pip install gliner") from exc

    resolved_model = _resolve_gliner_model(model)
    local_only = _gliner_local_only(resolved_model)
    if local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return GLiNER.from_pretrained(resolved_model, local_files_only=local_only)


def extract_entities_spacy(
    text: str, model: str = "en_core_web_sm", ruler_json: str | None = None
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    nlp = build_spacy_pipeline(model, ruler_json=ruler_json)
    doc = nlp(text)
    entities = [
        {
            "name": ent.text,
            "type": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]
    return _normalize_entities(entities), []


def extract_entities_gliner(
    text: str,
    labels: List[str] | None = None,
    model: str = "urchade/gliner_mediumv2",
    threshold: float = 0.3,
    gliner_model=None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    label_list = labels or list(DEFAULT_GLINER_LABELS)
    model_instance = gliner_model or build_gliner_model(model)
    predictions = model_instance.predict_entities(text, label_list, threshold=threshold)
    entities = [
        {
            "name": item.get("text"),
            "type": item.get("label"),
            "confidence": item.get("score"),
            "start_char": item.get("start") or item.get("start_char"),
            "end_char": item.get("end") or item.get("end_char"),
        }
        for item in predictions
    ]
    return _normalize_entities(entities, text=text), []


def _gliner_predictions_to_entities(predictions: List[Dict[str, Any]] | None) -> List[Dict[str, str]]:
    entities = []
    for item in predictions or []:
        name = _coerce_attr(item.get("text"))
        label = _coerce_attr(item.get("label"))
        entities.append(
            {
                "name": name,
                "type": label,
                "confidence": item.get("score"),
                "start_char": item.get("start") or item.get("start_char"),
                "end_char": item.get("end") or item.get("end_char"),
            }
        )
    return _normalize_entities(entities)


def extract_entities_gliner_batch(
    texts: List[str],
    labels: List[str] | None = None,
    model: str = "urchade/gliner_mediumv2",
    threshold: float = 0.3,
    gliner_model=None,
) -> List[List[Dict[str, str]]]:
    label_list = labels or list(DEFAULT_GLINER_LABELS)
    model_instance = gliner_model or build_gliner_model(model)
    predictions = None
    batch_method = getattr(model_instance, "batch_predict_entities", None)
    if callable(batch_method):
        predictions = batch_method(texts, label_list, threshold=threshold)
    if predictions is None:
        batch_method = getattr(model_instance, "predict_entities_batch", None)
        if callable(batch_method):
            predictions = batch_method(texts, label_list, threshold=threshold)
    if predictions is None:
        try:
            predictions = model_instance.predict_entities(texts, label_list, threshold=threshold)
        except Exception:
            predictions = None
    if not isinstance(predictions, list) or len(predictions) != len(texts):
        predictions = [
            model_instance.predict_entities(text, label_list, threshold=threshold) for text in texts
        ]
    return [_gliner_predictions_to_entities(items) for items in predictions]


def extract_entities_azure(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    from openai import AzureOpenAI

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    missing = [name for name, value in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_API_VERSION", api_version),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
    ] if not value]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI env vars: " + ", ".join(missing)
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    system_prompt = (
        "Extract named entities and relations from the input text. "
        "Return JSON only, no extra text. "
        "Schema: {\"entities\":[{\"name\":\"\",\"type\":\"\"}], "
        "\"relations\":[{\"source\":\"\",\"relation\":\"\",\"target\":\"\"}]}. "
        "Use concise entity types like PERSON, ORG, PRODUCT, GPE, DATE, TECH, CRYPTO, STANDARD."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    entities = _normalize_entities(data.get("entities", []))
    relations = _normalize_relations(data.get("relations", []))
    return entities, relations


def _extract_json_payload(raw: str) -> str:
    if not raw:
        return "{}"
    content = raw.strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
        content = "\n".join(lines).strip()
    return content or "{}"


def extract_entities_gemini(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    import google.generativeai as genai

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = (
        "Extract named entities and relations from the input text. "
        "Return JSON only, no extra text. "
        "Schema: {\"entities\":[{\"name\":\"\",\"type\":\"\"}], "
        "\"relations\":[{\"source\":\"\",\"relation\":\"\",\"target\":\"\"}]}. "
        "Use concise entity types like PERSON, ORG, PRODUCT, GPE, DATE, TECH, CRYPTO, STANDARD.\n\n"
        f"Text:\n{text}"
    )

    if _debug_llm_output():
        _setup_llm_debug_logging()
        print("LLM prompt (gemini):")
        print(prompt)
        logging.debug("LLM prompt (gemini): %s", prompt)

    response = model.generate_content(prompt)
    raw_content = getattr(response, "text", "") or ""
    if _debug_llm_output():
        print("LLM raw output (gemini):")
        print(raw_content)
        logging.debug("LLM raw output (gemini): %s", raw_content)
    content = _extract_json_payload(raw_content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    entities = _normalize_entities(data.get("entities", []), text=text)
    relations = _normalize_relations(data.get("relations", []))
    return entities, relations


def extract_entities_langextract(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    try:
        import langextract as lx
    except Exception as exc:
        raise RuntimeError("langextract is not installed. pip install langextract") from exc

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("LANGEXTRACT_API_KEY")
    model_id = os.getenv("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
    model_url = os.getenv("LANGEXTRACT_MODEL_URL")

    fence_output = _env_flag("LANGEXTRACT_FENCE_OUTPUT")
    use_schema_constraints = _env_flag("LANGEXTRACT_USE_SCHEMA_CONSTRAINTS")
    if use_schema_constraints is None:
        use_schema_constraints = True

    lower_model = model_id.lower()
    if lower_model.startswith("gpt-") or lower_model.startswith("o"):
        if _env_flag("LANGEXTRACT_USE_SCHEMA_CONSTRAINTS") is None:
            use_schema_constraints = False
        if _env_flag("LANGEXTRACT_FENCE_OUTPUT") is None:
            fence_output = True

    prompt = textwrap.dedent(
        """\
        You are an information extraction system. Extract high-signal entities and explicit
        relations from the input. Use exact text spans from the input for extraction_text.
        Prefer canonical names (no pronouns) and avoid generic words (e.g., "system", "device")
        unless they are clearly named entities in the text.

        Entity rules:
        - Include people, orgs, products, standards, protocols, locations, and key technical terms.
        - Use concise, normalized entity types: PERSON, ORG, PRODUCT, GPE, DATE, TECH, CRYPTO, STANDARD.
        - Deduplicate entities by exact surface form; do not invent new names.

        Relation rules:
        - Only create relations that are stated or strongly implied in the text.
        - Use short relation labels in UPPER_SNAKE_CASE (e.g., EMPLOYED_BY, PART_OF, USES, LOCATED_IN).
        - Source/target must match entity text spans.

        Classes:
        - entity: attributes {"type": "..."}.
        - relation: attributes {"source": "...", "relation": "...", "target": "..."}.
        """
    )

    examples = [
        lx.data.ExampleData(
            text="Alice works at Acme and knows Bob.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Alice",
                    attributes={"type": "PERSON"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Acme",
                    attributes={"type": "ORG"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Bob",
                    attributes={"type": "PERSON"},
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="works at",
                    attributes={
                        "source": "Alice",
                        "relation": "EMPLOYED_BY",
                        "target": "Acme",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="knows",
                    attributes={
                        "source": "Alice",
                        "relation": "KNOWS",
                        "target": "Bob",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text="ISO 15118 enables Plug&Charge for the Digital Key project in Berlin.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="ISO 15118",
                    attributes={"type": "STANDARD"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Plug&Charge",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Digital Key",
                    attributes={"type": "PRODUCT"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Berlin",
                    attributes={"type": "GPE"},
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="enables",
                    attributes={
                        "source": "ISO 15118",
                        "relation": "ENABLES",
                        "target": "Plug&Charge",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="for",
                    attributes={
                        "source": "Plug&Charge",
                        "relation": "USED_IN",
                        "target": "Digital Key",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="in",
                    attributes={
                        "source": "Digital Key",
                        "relation": "LOCATED_IN",
                        "target": "Berlin",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "Car Connectivity Consortium (CCC) defines Digital Key. "
                "Digital Key uses AES-256 and ECC for encryption and key exchange. "
                "ISO/IEC 18013-5 specifies mDL security requirements."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Car Connectivity Consortium",
                    attributes={"type": "ORG"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="CCC",
                    attributes={"type": "ORG"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Digital Key",
                    attributes={"type": "PRODUCT"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="AES-256",
                    attributes={"type": "CRYPTO"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="ECC",
                    attributes={"type": "CRYPTO"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="ISO/IEC 18013-5",
                    attributes={"type": "STANDARD"},
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="defines",
                    attributes={
                        "source": "Car Connectivity Consortium",
                        "relation": "DEFINES",
                        "target": "Digital Key",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="uses",
                    attributes={
                        "source": "Digital Key",
                        "relation": "USES",
                        "target": "AES-256",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="uses",
                    attributes={
                        "source": "Digital Key",
                        "relation": "USES",
                        "target": "ECC",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="specifies",
                    attributes={
                        "source": "ISO/IEC 18013-5",
                        "relation": "SPECIFIES",
                        "target": "mDL security requirements",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "The mobile endpoint uses PKI for authentication. "
                "Host Card Emulation (HCE) enables the mobile device to emulate a card for the vehicle. "
                "The vehicle endpoint validates the mobile endpoint over NFC."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="mobile endpoint",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="PKI",
                    attributes={"type": "CRYPTO"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Host Card Emulation",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="HCE",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="mobile device",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="vehicle",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="vehicle endpoint",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="NFC",
                    attributes={"type": "TECH"},
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="uses",
                    attributes={
                        "source": "mobile endpoint",
                        "relation": "USES",
                        "target": "PKI",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="enables",
                    attributes={
                        "source": "Host Card Emulation",
                        "relation": "ENABLES",
                        "target": "mobile device",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="emulate",
                    attributes={
                        "source": "mobile device",
                        "relation": "EMULATES",
                        "target": "card",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="validates",
                    attributes={
                        "source": "vehicle endpoint",
                        "relation": "VALIDATES",
                        "target": "mobile endpoint",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relation",
                    extraction_text="over",
                    attributes={
                        "source": "vehicle endpoint",
                        "relation": "COMMUNICATES_OVER",
                        "target": "NFC",
                    },
                ),
            ],
        ),
    ]

    if _debug_llm_output():
        _setup_llm_debug_logging()
        print("LLM prompt (langextract):")
        print(prompt)
        logging.debug("LLM prompt (langextract): %s", prompt)

    def _run_langextract(
        fence: bool | None, schema_constraints: bool | None
    ):
        return lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            api_key=api_key,
            model_url=model_url,
            fence_output=fence,
            use_schema_constraints=schema_constraints,
            show_progress=False,
            debug=_debug_llm_output(),
        )

    max_attempts, backoff_seconds = _get_retry_settings()
    retry_plan: list[tuple[bool | None, bool | None]] = [
        (fence_output, use_schema_constraints),
        (True, True),
        (True, False),
    ]
    while len(retry_plan) < max_attempts:
        retry_plan.append((True, False))

    for attempt in range(1, max_attempts + 1):
        try:
            fence, schema_constraints = retry_plan[attempt - 1]
            result = _run_langextract(fence, schema_constraints)
            break
        except Exception as exc:
            logging.exception("LangExtract attempt %s failed", attempt)
            if attempt >= max_attempts:
                print(f"LangExtract retry failed, skipping paragraph: {exc}")
                return [], []
            print(
                f"LangExtract parse failed, retrying with strict JSON output "
                f"(attempt {attempt + 1}/{max_attempts}): {exc}"
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)

    document = result[0] if isinstance(result, list) else result
    extractions = document.extractions or []
    entities: List[Dict[str, str]] = []
    relations: List[Dict[str, str]] = []

    if _debug_llm_output():
        print("LLM structured output (langextract):")
        for extraction in extractions:
            cls = (extraction.extraction_class or "").strip()
            text_value = (extraction.extraction_text or "").strip()
            attrs = extraction.attributes or {}
            print({"class": cls, "text": text_value, "attributes": attrs})

    for extraction in extractions:
        cls = (extraction.extraction_class or "").strip().lower()
        text_value = (extraction.extraction_text or "").strip()
        attrs = extraction.attributes or {}
        if cls == "entity":
            entities.append(
                {
                    "name": text_value,
                    "type": _coerce_attr(attrs.get("type")) or "UNKNOWN",
                }
            )
        elif cls == "relation":
            relations.append(
                {
                    "source": _coerce_attr(attrs.get("source")),
                    "relation": _coerce_attr(attrs.get("relation")),
                    "target": _coerce_attr(attrs.get("target")),
                }
            )

    return _normalize_entities(entities, text=text), _normalize_relations(relations)
