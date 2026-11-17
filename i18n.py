"""Dynamic localization for DiaCheck using Google Cloud Translation Basic (v2).

Only fixed interface text is translated. Patient answers are never sent to this API.
The app keeps a manually reviewed base set of locales and can add all target languages
returned by the configured Google Cloud Translation project.
"""
from __future__ import annotations

import html
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

API_BASE = "https://translation.googleapis.com/language/translate/v2"


class LocalizationError(RuntimeError):
    """Raised when the translation provider cannot serve a localization request."""


def _read_json(url: str, timeout: int = 15) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "DiaCheck/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LocalizationError("Could not contact the translation service.") from exc


def list_supported_languages(api_key: str) -> list[tuple[str, str]]:
    """Return (language_code, display_name) pairs exposed by Cloud Translation Basic."""
    if not api_key:
        raise LocalizationError("A Google Cloud Translation API key is required.")
    params = urlencode({"target": "en", "model": "nmt", "key": api_key})
    payload = _read_json(f"{API_BASE}/languages?{params}")
    if "error" in payload:
        raise LocalizationError("The translation service rejected the language-list request.")
    languages = payload.get("data", {}).get("languages", [])
    result = []
    for item in languages:
        code = item.get("language")
        name = item.get("name") or code
        if code and item.get("language"):
            result.append((str(code), str(name)))
    if not result:
        raise LocalizationError("No supported languages were returned by the translation service.")
    return sorted(set(result), key=lambda item: item[1].casefold())


def _translate_batch(texts: list[str], target: str, api_key: str) -> list[str]:
    """Translate one batch. Cloud Translation Basic accepts up to 128 q values."""
    form_fields = [("q", value) for value in texts]
    form_fields.extend([
        ("source", "en"),
        ("target", target),
        ("format", "text"),
    ])
    body = urlencode(form_fields).encode("utf-8")
    request = Request(
        f"{API_BASE}?key={quote(api_key)}",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "DiaCheck/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LocalizationError("The translation service could not translate this language.") from exc
    if "error" in payload:
        raise LocalizationError("The translation service rejected this translation request.")
    translated = payload.get("data", {}).get("translations", [])
    if len(translated) != len(texts):
        raise LocalizationError("The translation service returned an incomplete result.")
    return [html.unescape(item.get("translatedText", "")) for item in translated]


def translate_catalog(catalog: dict[str, Any], target: str, api_key: str) -> dict[str, Any]:
    """Translate all string leaves in a nested UI catalog, preserving its structure."""
    if not api_key:
        raise LocalizationError("A Google Cloud Translation API key is required.")

    unique_texts: list[str] = []
    seen: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value not in seen:
                seen.add(value)
                unique_texts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(catalog)
    translated_map: dict[str, str] = {}
    for start in range(0, len(unique_texts), 100):
        batch = unique_texts[start:start + 100]
        translated_batch = _translate_batch(batch, target, api_key)
        translated_map.update(zip(batch, translated_batch))

    def rebuild(value: Any) -> Any:
        if isinstance(value, str):
            return translated_map.get(value, value)
        if isinstance(value, dict):
            return {key: rebuild(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rebuild(child) for child in value]
        return value

    return rebuild(catalog)


def merge_catalog(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay reviewed translations on top of machine translations."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_catalog(result[key], value)
        else:
            result[key] = value
    return result
