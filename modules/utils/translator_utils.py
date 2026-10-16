import ast
import base64
import json
import logging
import re
import numpy as np
from .textblock import TextBlock
import imkit as imk

logger = logging.getLogger(__name__)


MODEL_MAP = {
    "Custom": "",  
    "Deepseek": "deepseek-v4-flash", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude-4.6-Sonnet": "claude-sonnet-4-6",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.5-Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini-3.1-Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json


def _clean_extracted_value(raw_val: str) -> str:
    """Helper to clean and unwrap an extracted raw string value from fallback regex parsing."""
    cleaned = raw_val.strip()
    # Strip trailing JSON punctuation like commas, semicolons, and closing braces
    cleaned = re.sub(r'[,;\s\}]+$', '', cleaned).strip()

    # Strip matching outer quotes: "", '', «», “”, etc.
    if len(cleaned) >= 2:
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        elif (cleaned.startswith('“') and cleaned.endswith('”')) or \
             (cleaned.startswith('«') and cleaned.endswith('»')):
            cleaned = cleaned[1:-1]

    # Unescape common escape sequences if present
    cleaned = cleaned.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
    return cleaned.strip()


def _normalize_dict_keys(d: dict) -> dict[str, str]:
    """Normalize dictionary keys so that variations like Block_0, block-0, block 0, or 0 map to block_0."""
    normalized = {}
    for k, v in d.items():
        k_str = str(k).strip()
        m = re.match(r'^(?:block[_\s\-]?(\d+)|(\d+))$', k_str, re.IGNORECASE)
        val_str = "" if v is None else str(v)
        if m:
            idx = m.group(1) or m.group(2)
            norm_key = f"block_{idx}"
            normalized[norm_key] = val_str
        else:
            normalized[k_str] = val_str
    return normalized


def _repair_missing_colons(text: str) -> str:
    """
    Repairs missing colons after JSON object keys like "block_6" or "scene_description".
    Handles common LLM formatting mistakes:
      - 3 quotes total (merged quote between key and value): "block_6 "...value", -> "block_6": "value",
      - 4 quotes total (two separate quoted strings without colon): "block_6" "value", -> "block_6": "value",
      - Key in quotes followed by unquoted value without colon: "block_6" *Тук-тук*, -> "block_6": *Тук-тук*,
      - Unquoted key followed by value without colon: block_6 "value", -> block_6: "value",
    """
    # 1. Four quotes / two separate quoted strings without colon:
    # E.g. "block_6" "value" or "block_6 " "value"
    text = re.sub(
        r'(?m)(?:^|(?<=[\{,]))(\s*[\"\']\s*(?:block[_\s\-]?\d+|scene_description|[a-zA-Z0-9_\-]+)\s*[\"\'])\s*(?!:)\s*([\"\'«“])',
        r'\1: \2',
        text,
    )

    # 2. Three quotes / merged quote between key and value:
    # E.g. "block_6 "...Как же я жалок.",
    text = re.sub(
        r'(?m)(?:^|(?<=[\{,]))(\s*[\"\']\s*(?:block[_\s\-]?\d+|scene_description|[a-zA-Z0-9_\-]+)\s*)[\"\'](?!\s*:)\s*([^":,\r\n\s][^\r\n]*?)([\"\'»”])(\s*[,}\r\n]|$)',
        r'\1": "\2\3\4',
        text,
    )

    # 3. Key in quotes followed by unquoted value without colon:
    # E.g. "block_6" *Тук-тук*, or "block_6 " *Тук-тук*,
    text = re.sub(
        r'(?m)(?:^|(?<=[\{,]))(\s*[\"\']\s*(?:block[_\s\-]?\d+|scene_description|[a-zA-Z0-9_\-]+)\s*[\"\'])\s*(?!:)\s*([^":,\{\}\[\]\s\r\n][^\r\n]*)',
        r'\1: \2',
        text,
    )

    # 4. Unquoted key followed by value without colon:
    # E.g. block_6 "value",
    text = re.sub(
        r'(?m)(?:^|(?<=[\{,]))(\s*(?:block[_\s\-]?\d+|scene_description))\s*(?!:)\s*([\"\'«“\*])',
        r'\1: \2',
        text,
    )

    return text


def parse_damaged_json(json_string: str, expected_keys: list[str] = None) -> dict[str, str]:
    """
    Attempts to repair and parse a malformed JSON string returned by LLM.
    Handles common LLM formatting issues:
      - Missing colons after keys (e.g. "block_6 "...value",)
      - Trailing commas before closing braces/brackets
      - Unquoted text values (e.g., "block_1": *Тук-тук*,)
      - Unescaped quotes inside values
      - Missing commas between properties
      - Python-style dictionaries with single quotes
      - Truncated or incomplete JSON wrappers
      - Direct fallback regex key-value extraction

    Args:
        json_string: Raw or partial JSON string from the LLM.
        expected_keys: Optional list of keys (e.g. ['block_0', 'block_1', ...]) that must be present.

    Returns:
        dict[str, str]: Successfully parsed and normalized dictionary.

    Raises:
        ValueError: If JSON cannot be parsed or if expected keys are missing.
    """
    if not json_string or not json_string.strip():
        raise ValueError("Cannot parse empty LLM response.")

    # 1. Clean markdown code fences if any
    cleaned = re.sub(r"```(?:json)?\s*", "", json_string, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # 2. Locate JSON object boundary if braces exist
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate_text = cleaned[first_brace : last_brace + 1]
    elif first_brace != -1:
        candidate_text = cleaned[first_brace:] + "\n}"
    else:
        candidate_text = "{\n" + cleaned + "\n}"

    result_dict = None

    # Step A: Try standard json.loads with strict=False (allows raw newlines/tabs inside strings)
    try:
        data = json.loads(candidate_text, strict=False)
        if isinstance(data, dict):
            result_dict = data
    except Exception:
        pass

    repaired = candidate_text

    # Step B: Syntax repairs (comments, missing colons, trailing commas, unquoted strings, missing commas)
    if result_dict is None:
        # 1. Remove single-line comments // ... and multi-line comments /* ... */
        repaired = re.sub(r"(?m)^\s*//.*?$", "", repaired)
        repaired = re.sub(r'(?<![:"\'\\])//.*', "", repaired)
        repaired = re.sub(r'/\*[\s\S]*?\*/', '', repaired)

        # 2. Fix missing colons after keys (e.g. "block_6 "...val" or "block_6" "val")
        repaired = _repair_missing_colons(repaired)

        # 3. Fix trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # 4. Fix unquoted string values (e.g. "block_1": *Тук-тук*,)
        def quote_unquoted_value(match):
            prefix = match.group(1)
            val = match.group(2).strip()
            suffix = match.group(3)
            # Check if already quoted or a valid JSON literal / number
            if (
                val.startswith(('"', "'", "{", "["))
                or val in ("true", "false", "null", "True", "False", "None")
                or re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", val)
            ):
                return match.group(0)
            val_escaped = val.replace('\\"', '"').replace('"', '\\"')
            return f'{prefix}"{val_escaped}"{suffix}'

        repaired = re.sub(
            r'([\"\'\w\-]+\s*:\s*)([^\s",\{\}\[\]][^\r\n,\{\}]*?)(\s*[,}\r\n])',
            quote_unquoted_value,
            repaired,
        )

        # 5. Fix missing commas between properties (e.g. "block_0": "val"\n "block_1": "val")
        repaired = re.sub(
            r'(\"|\d|true|false|null)\s*[\r\n]+\s*([\"\'\w\-]+\s*:)',
            r'\1,\n\2',
            repaired,
        )

        # Clean any trailing commas that might have been introduced
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            data = json.loads(repaired, strict=False)
            if isinstance(data, dict):
                result_dict = data
        except Exception:
            pass

    # Step C: Try ast.literal_eval for python-style dicts with single quotes
    if result_dict is None:
        try:
            py_dict_str = repaired
            py_dict_str = re.sub(r"\btrue\b", "True", py_dict_str)
            py_dict_str = re.sub(r"\bfalse\b", "False", py_dict_str)
            py_dict_str = re.sub(r"\bnull\b", "None", py_dict_str)
            py_dict_str = re.sub(r",\s*([}\]])", r"\1", py_dict_str)
            data = ast.literal_eval(py_dict_str)
            if isinstance(data, dict):
                result_dict = data
        except Exception:
            pass

    # Step D: Specialized regex block extractor (fallback for badly malformed or truncated text)
    if result_dict is None:
        result_dict = {}
        source_text = _repair_missing_colons(repaired if repaired else cleaned)
        key_regex = re.compile(
            r'([\"\'`]?)\s*(block[_\s\-]?\d+|scene_description|[a-zA-Z0-9_\-]+)\s*\1\s*(?::|\s*(?=[\"\'«“\*]))\s*',
            flags=re.MULTILINE | re.IGNORECASE,
        )
        matches = list(key_regex.finditer(source_text))
        for i, m in enumerate(matches):
            key = m.group(2)
            val_start = m.end()
            val_end = matches[i + 1].start() if i + 1 < len(matches) else len(source_text)
            raw_val = source_text[val_start:val_end]
            result_dict[key] = _clean_extracted_value(raw_val)

    if not isinstance(result_dict, dict) or not result_dict:
        raise ValueError("Failed to parse or repair LLM JSON response.")

    normalized_dict = _normalize_dict_keys(result_dict)

    # Validate that all expected keys were successfully extracted
    if expected_keys is not None:
        missing = [k for k in expected_keys if k not in normalized_dict]
        if missing:
            raise ValueError(
                f"Missing required blocks in parsed translation: {missing}. "
                f"Available blocks: {list(normalized_dict.keys())}"
            )

    return normalized_dict


def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    if not json_string:
        raise ValueError("No JSON found in LLM response (empty response).")
    json_string = remove_think_block(json_string)

    expected_keys = [f"block_{idx}" for idx in range(len(blk_list))]

    # Try standard json extraction and parsing first
    match = re.search(r"\{[\s\S]*\}", json_string)
    extracted_json_string = match.group(0) if match else json_string

    translation_dict = None
    try:
        data = json.loads(extracted_json_string)
        if isinstance(data, dict):
            translation_dict = _normalize_dict_keys(data)
        else:
            raise ValueError(f"Expected JSON object in LLM response, got {type(data).__name__}")
    except Exception as e:
        logger.warning(
            "json.loads failed on LLM response (%s). Attempting to repair and parse damaged JSON...",
            e,
        )
        translation_dict = parse_damaged_json(json_string, expected_keys=expected_keys)

    # Check if all blocks were successfully extracted
    missing_keys = [k for k in expected_keys if k not in translation_dict]
    if missing_keys:
        raise ValueError(
            f"Not all translation blocks are present in LLM response. "
            f"Missing blocks: {missing_keys}. Available blocks: {list(translation_dict.keys())}"
        )

    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        blk.translation = translation_dict[block_key]

def remove_think_block(text: str):
    if not text:
        return text
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper() 
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper()
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)

def has_translatable_content(text: str | None) -> bool:
    """True when source text contains a letter or number worth translating."""
    if not text:
        return False
    return any(ch.isalnum() for ch in text)

def is_renderable_translation(translation: str | None) -> bool:
    """True if the render stage should draw this translation.

    Punctuation-only translations (an echoed "?", "!?", "...") aren't worth
    redrawing — the original artwork already shows the same thing. Anything
    gated on rendering (like inpainting) must skip them too, otherwise the
    bubble gets cleaned with nothing drawn over it. Unlike a length check,
    this keeps legitimate single-character translations (e.g. "何", "5").
    """
    return has_translatable_content(translation)
