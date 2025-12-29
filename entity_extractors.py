import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for item in entities or []:
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name:
            continue
        normalized.append({"name": name, "type": etype or "UNKNOWN"})
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


def extract_entities_spacy(
    text: str, model: str = "en_core_web_sm", ruler_json: str | None = None
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    nlp = build_spacy_pipeline(model, ruler_json=ruler_json)
    doc = nlp(text)
    entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
    return _normalize_entities(entities), []


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

    response = model.generate_content(prompt)
    content = _extract_json_payload(getattr(response, "text", "") or "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    entities = _normalize_entities(data.get("entities", []))
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
        Extract entities and relations in order of appearance.
        Use exact text spans from the input for extraction_text.

        Classes:
        - entity: attributes {"type": "..."} (e.g., PERSON, ORG, PRODUCT, GPE, DATE, TECH).
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
        )
    ]

    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=api_key,
        fence_output=fence_output,
        use_schema_constraints=use_schema_constraints,
        show_progress=False,
    )

    document = result[0] if isinstance(result, list) else result
    extractions = document.extractions or []
    entities: List[Dict[str, str]] = []
    relations: List[Dict[str, str]] = []

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

    return _normalize_entities(entities), _normalize_relations(relations)
