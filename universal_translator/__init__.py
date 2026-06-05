"""Public package exports for universal_translator."""

from .phrases import (
    get_phrase_entry,
    list_language_entries,
    list_latin_phrases,
    list_phrase_entries,
    list_phrase_languages,
    load_phrase_database,
)
from .translator import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_LANGUAGES,
    TranslationResult,
    UniversalTranslator,
    list_supported_languages,
    resolve_language_code,
    resolve_language_name,
    translate_from_latin,
    translate_text,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_SOURCE_LANGUAGE",
    "DEFAULT_TARGET_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "TranslationResult",
    "UniversalTranslator",
    "get_phrase_entry",
    "list_language_entries",
    "list_latin_phrases",
    "list_phrase_entries",
    "list_phrase_languages",
    "load_phrase_database",
    "list_supported_languages",
    "resolve_language_code",
    "resolve_language_name",
    "translate_from_latin",
    "translate_text",
]
