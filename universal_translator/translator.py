from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .phrases import (
    build_language_alias_map,
    get_language_entry_map,
    get_phrase_entry,
    list_language_entries,
    list_latin_phrases,
    load_phrase_database,
    lookup_phrase,
    normalize_language_key,
)

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_SOURCE_LANGUAGE = "lat_Latn"
DEFAULT_TARGET_LANGUAGE = "eng_Latn"
CLI_VERSION = "0.1.0"
LATIN_TO_ENGLISH_MODEL_NAME = "Helsinki-NLP/opus-mt-ine-en"
ENGLISH_TO_LATIN_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ine"

LANGUAGE_ENTRIES = list_language_entries()
SUPPORTED_LANGUAGES: Dict[str, str] = {
    entry["name"]: entry["code"] for entry in LANGUAGE_ENTRIES
}
LANGUAGE_ALIASES: Dict[str, str] = build_language_alias_map()
LANGUAGE_ENTRY_MAP = get_language_entry_map()
CODE_TO_LANGUAGE: Dict[str, str] = {
    entry["code"]: entry["name"] for entry in LANGUAGE_ENTRIES
}

KNOWN_COMMANDS = {
    "translate",
    "batch",
    "languages",
    "phrases",
    "lookup",
    "stats",
    "doctor",
    "examples",
}

console = Console()


def _normalize_language_code(language: str) -> str:
    key = normalize_language_key(language)
    return LANGUAGE_ALIASES.get(key, language.strip())


def _supported_language_error(language: str) -> ValueError:
    preview = ", ".join(list_supported_languages(limit=8))
    return ValueError(
        f"Unsupported language {language!r}. Use an NLLB code or one of the built-in aliases. "
        f"Examples: {preview}."
    )


def resolve_language_code(language: str) -> str:
    code = _normalize_language_code(language)
    if code in CODE_TO_LANGUAGE:
        return code
    raise _supported_language_error(language)


def resolve_language_name(language: str) -> str:
    code = resolve_language_code(language)
    return CODE_TO_LANGUAGE[code]


def list_supported_languages(limit: Optional[int] = None) -> List[str]:
    names = sorted(SUPPORTED_LANGUAGES)
    if limit is None:
        return names
    return names[:limit]


@dataclass
class TranslationResult:
    text: str
    source_language: str
    target_language: str
    model_name: str
    provider: str = "model"

    @property
    def source_language_name(self) -> str:
        return CODE_TO_LANGUAGE.get(self.source_language, self.source_language)

    @property
    def target_language_name(self) -> str:
        return CODE_TO_LANGUAGE.get(self.target_language, self.target_language)

    def to_serializable(self) -> Dict[str, str]:
        payload = asdict(self)
        payload["source_language_name"] = self.source_language_name
        payload["target_language_name"] = self.target_language_name
        return payload


class UniversalTranslator:
    """Reusable wrapper around Meta's NLLB translation models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._specialized_tokenizers: Dict[str, object] = {}
        self._specialized_models: Dict[str, object] = {}

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def _load_specialized_translation_stack(self, model_name: str):
        tokenizer = self._specialized_tokenizers.get(model_name)
        model = self._specialized_models.get(model_name)
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            self._specialized_tokenizers[model_name] = tokenizer
            self._specialized_models[model_name] = model
        return tokenizer, model

    def translate(
        self,
        text: str,
        source_language: str = DEFAULT_SOURCE_LANGUAGE,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        max_new_tokens: int = 256,
    ) -> TranslationResult:
        if not text or not text.strip():
            raise ValueError("Text to translate cannot be empty.")

        source_code = resolve_language_code(source_language)
        target_code = resolve_language_code(target_language)

        phrase_translation = self._translate_from_phrase_database(
            text=text,
            source_language=source_code,
            target_language=target_code,
        )
        if phrase_translation is not None:
            return TranslationResult(
                text=phrase_translation,
                source_language=source_code,
                target_language=target_code,
                model_name=self.model_name,
                provider="phrase_database",
            )

        if source_code == DEFAULT_SOURCE_LANGUAGE or target_code == DEFAULT_SOURCE_LANGUAGE:
            return self._translate_with_latin_source(
                text=text,
                source_language=source_code,
                target_language=target_code,
                max_new_tokens=max_new_tokens,
            )

        return self._translate_with_nllb(
            text=text,
            source_code=source_code,
            target_code=target_code,
            max_new_tokens=max_new_tokens,
        )

    def _translate_with_nllb(
        self,
        text: str,
        source_code: str,
        target_code: str,
        max_new_tokens: int,
    ) -> TranslationResult:
        tokenizer = self.tokenizer
        model = self.model

        tokenizer.src_lang = source_code
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise ValueError(
                f"Target language {target_code!r} is not available in the loaded tokenizer."
            )

        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )
        translated_text = tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0]

        return TranslationResult(
            text=translated_text,
            source_language=source_code,
            target_language=target_code,
            model_name=self.model_name,
            provider="model",
        )

    def _translate_with_latin_source(
        self,
        text: str,
        source_language: str,
        target_language: str,
        max_new_tokens: int,
    ) -> TranslationResult:
        if source_language == DEFAULT_SOURCE_LANGUAGE and target_language == DEFAULT_TARGET_LANGUAGE:
            return self._translate_latin_to_english(text=text, max_new_tokens=max_new_tokens)

        if source_language == DEFAULT_SOURCE_LANGUAGE:
            english_result = self._translate_latin_to_english(
                text=text,
                max_new_tokens=max_new_tokens,
            )
            if target_language == DEFAULT_TARGET_LANGUAGE:
                return english_result

            bridged_result = self._translate_with_nllb(
                text=english_result.text,
                source_code=DEFAULT_TARGET_LANGUAGE,
                target_code=target_language,
                max_new_tokens=max_new_tokens,
            )
            bridged_result.provider = "latin_bridge"
            return TranslationResult(
                text=bridged_result.text,
                source_language=source_language,
                target_language=target_language,
                model_name=f"{LATIN_TO_ENGLISH_MODEL_NAME} -> {self.model_name}",
                provider="latin_bridge",
            )

        if target_language == DEFAULT_SOURCE_LANGUAGE and source_language == DEFAULT_TARGET_LANGUAGE:
            return self._translate_english_to_latin(text=text, max_new_tokens=max_new_tokens)

        if target_language == DEFAULT_SOURCE_LANGUAGE:
            english_result = self._translate_with_nllb(
                text=text,
                source_code=source_language,
                target_code=DEFAULT_TARGET_LANGUAGE,
                max_new_tokens=max_new_tokens,
            )
            latin_result = self._translate_english_to_latin(
                text=english_result.text,
                max_new_tokens=max_new_tokens,
            )
            return TranslationResult(
                text=latin_result.text,
                source_language=source_language,
                target_language=target_language,
                model_name=f"{self.model_name} -> {ENGLISH_TO_LATIN_MODEL_NAME}",
                provider="latin_bridge",
            )

        raise ValueError("Unexpected Latin translation routing state.")

    def _translate_latin_to_english(self, text: str, max_new_tokens: int) -> TranslationResult:
        tokenizer, model = self._load_specialized_translation_stack(LATIN_TO_ENGLISH_MODEL_NAME)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)
        translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return TranslationResult(
            text=translated_text,
            source_language=DEFAULT_SOURCE_LANGUAGE,
            target_language=DEFAULT_TARGET_LANGUAGE,
            model_name=LATIN_TO_ENGLISH_MODEL_NAME,
            provider="latin_model",
        )

    def _translate_english_to_latin(self, text: str, max_new_tokens: int) -> TranslationResult:
        tokenizer, model = self._load_specialized_translation_stack(ENGLISH_TO_LATIN_MODEL_NAME)
        prefixed_text = f">>lat_Latn<< {text}"
        inputs = tokenizer(prefixed_text, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)
        translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return TranslationResult(
            text=translated_text,
            source_language=DEFAULT_TARGET_LANGUAGE,
            target_language=DEFAULT_SOURCE_LANGUAGE,
            model_name=ENGLISH_TO_LATIN_MODEL_NAME,
            provider="latin_model",
        )

    def translate_batch(
        self,
        texts: Iterable[str],
        source_language: str = DEFAULT_SOURCE_LANGUAGE,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        max_new_tokens: int = 256,
    ) -> List[TranslationResult]:
        normalized_texts = [text for text in texts if text and text.strip()]
        if not normalized_texts:
            raise ValueError("At least one non-empty text is required for batch translation.")

        return [
            self.translate(
                text=text,
                source_language=source_language,
                target_language=target_language,
                max_new_tokens=max_new_tokens,
            )
            for text in normalized_texts
        ]

    def translate_from_latin(
        self,
        text: str,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        max_new_tokens: int = 256,
    ) -> TranslationResult:
        return self.translate(
            text=text,
            source_language=DEFAULT_SOURCE_LANGUAGE,
            target_language=target_language,
            max_new_tokens=max_new_tokens,
        )

    def translate_to_english(
        self,
        text: str,
        source_language: str = DEFAULT_SOURCE_LANGUAGE,
        max_new_tokens: int = 256,
    ) -> TranslationResult:
        return self.translate(
            text=text,
            source_language=source_language,
            target_language=DEFAULT_TARGET_LANGUAGE,
            max_new_tokens=max_new_tokens,
        )

    def _translate_from_phrase_database(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Optional[str]:
        if source_language != DEFAULT_SOURCE_LANGUAGE:
            return None
        return lookup_phrase(text, target_language)


def translate_text(
    text: str,
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 256,
) -> str:
    translator = UniversalTranslator(model_name=model_name)
    return translator.translate(
        text=text,
        source_language=source_language,
        target_language=target_language,
        max_new_tokens=max_new_tokens,
    ).text


def translate_from_latin(
    text: str,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 256,
) -> str:
    translator = UniversalTranslator(model_name=model_name)
    return translator.translate_from_latin(
        text=text,
        target_language=target_language,
        max_new_tokens=max_new_tokens,
    ).text


def _read_lines_from_file(path: str) -> List[str]:
    content = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def _result_to_table(result: TranslationResult, original_text: str) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Input", original_text)
    table.add_row("Output", result.text)
    table.add_row("Source", f"{result.source_language_name} ({result.source_language})")
    table.add_row("Target", f"{result.target_language_name} ({result.target_language})")
    table.add_row("Provider", result.provider)
    table.add_row("Model", result.model_name)
    return table


def _print_result(result: TranslationResult, original_text: str, as_json: bool) -> None:
    if as_json:
        payload = result.to_serializable()
        payload["input_text"] = original_text
        console.print_json(json.dumps(payload))
        return

    console.print(
        Panel(
            _result_to_table(result, original_text),
            title="Translation",
            border_style="green",
        )
    )


def _build_languages_table(show_codes: bool, query: Optional[str], limit: Optional[int]) -> Table:
    table = Table(title="Supported Languages", header_style="bold magenta")
    table.add_column("Language", style="cyan")
    if show_codes:
        table.add_column("Code", style="green", no_wrap=True)
    table.add_column("Script", style="white")

    rows = LANGUAGE_ENTRIES
    if query:
        needle = query.lower()
        rows = [
            entry
            for entry in rows
            if needle in entry["name"].lower() or needle in entry["code"].lower()
        ]
    if limit is not None:
        rows = rows[:limit]

    for entry in rows:
        if show_codes:
            table.add_row(entry["name"], entry["code"], entry["script"])
        else:
            table.add_row(entry["name"], entry["script"])
    return table


def _build_phrase_table(limit: Optional[int], query: Optional[str]) -> Table:
    phrases = list_latin_phrases()
    if query:
        needle = query.lower()
        phrases = [phrase for phrase in phrases if needle in phrase.lower()]
    if limit is not None:
        phrases = phrases[:limit]

    table = Table(title="Latin Phrase Database", header_style="bold magenta")
    table.add_column("#", style="green", no_wrap=True)
    table.add_column("Phrase", style="cyan")
    for index, phrase in enumerate(phrases, start=1):
        table.add_row(str(index), phrase)
    return table


def _print_examples() -> None:
    examples = Table(title="CLI Examples", header_style="bold magenta")
    examples.add_column("Command", style="cyan")
    examples.add_column("What It Does", style="white")
    examples.add_row(
        'universal-translator translate "Salve amice"',
        "Translate a single phrase with the default Latin -> English flow.",
    )
    examples.add_row(
        "universal-translator translate --source latin --target french \"carpe diem\"",
        "Translate a Latin phrase into French.",
    )
    examples.add_row(
        "universal-translator batch --file phrases.txt --target spanish",
        "Translate multiple lines from a text file.",
    )
    examples.add_row(
        "universal-translator lookup \"memento mori\" --show-entry",
        "Inspect the local phrase database entry for an exact phrase.",
    )
    examples.add_row(
        "universal-translator languages --search arabic --show-codes",
        "Search the 200+ language registry.",
    )
    examples.add_row(
        "universal-translator stats",
        "Show package, phrase, and language registry counts.",
    )
    console.print(examples)


def _print_stats() -> None:
    metadata = load_phrase_database()["metadata"]
    table = Table(title="Translator Stats", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CLI Version", CLI_VERSION)
    table.add_row("Model", DEFAULT_MODEL_NAME)
    table.add_row("Latin -> English Model", LATIN_TO_ENGLISH_MODEL_NAME)
    table.add_row("English -> Latin Model", ENGLISH_TO_LATIN_MODEL_NAME)
    table.add_row("Phrase Entries", str(metadata["curated_phrase_translation_count"]))
    table.add_row("Language Entries", str(metadata["language_entry_count"]))
    table.add_row("Base Languages", str(metadata["base_language_count"]))
    table.add_row("Default Source", f"{DEFAULT_SOURCE_LANGUAGE} ({resolve_language_name(DEFAULT_SOURCE_LANGUAGE)})")
    table.add_row("Default Target", f"{DEFAULT_TARGET_LANGUAGE} ({resolve_language_name(DEFAULT_TARGET_LANGUAGE)})")
    console.print(table)


def _print_doctor() -> None:
    table = Table(title="Environment Doctor", header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Torch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Device", torch.cuda.get_device_name(0))
    table.add_row("Phrase Entries", str(load_phrase_database()["metadata"]["curated_phrase_translation_count"]))
    table.add_row("Language Entries", str(len(LANGUAGE_ENTRIES)))
    table.add_row("Default Model", DEFAULT_MODEL_NAME)
    table.add_row("Latin Fallback", LATIN_TO_ENGLISH_MODEL_NAME)
    console.print(table)


def _handle_translate(args: argparse.Namespace) -> None:
    result = UniversalTranslator(model_name=args.model).translate(
        text=args.text,
        source_language=args.source,
        target_language=args.target,
        max_new_tokens=args.max_new_tokens,
    )
    _print_result(result, args.text, args.json)


def _handle_batch(args: argparse.Namespace) -> None:
    texts: List[str] = []
    if args.file:
        texts.extend(_read_lines_from_file(args.file))
    texts.extend(args.texts or [])

    if not texts:
        raise ValueError("Provide at least one text or use --file for batch translation.")

    results = UniversalTranslator(model_name=args.model).translate_batch(
        texts=texts,
        source_language=args.source,
        target_language=args.target,
        max_new_tokens=args.max_new_tokens,
    )

    if args.json:
        payload = []
        for original, result in zip(texts, results):
            item = result.to_serializable()
            item["input_text"] = original
            payload.append(item)
        console.print_json(json.dumps(payload))
        return

    table = Table(title="Batch Translation", header_style="bold magenta")
    table.add_column("#", style="green", no_wrap=True)
    table.add_column("Input", style="cyan")
    table.add_column("Output", style="white")
    table.add_column("Provider", style="yellow", no_wrap=True)
    for index, (original, result) in enumerate(zip(texts, results), start=1):
        table.add_row(str(index), original, result.text, result.provider)
    console.print(table)


def _handle_languages(args: argparse.Namespace) -> None:
    console.print(_build_languages_table(args.show_codes, args.search, args.limit))


def _handle_phrases(args: argparse.Namespace) -> None:
    console.print(_build_phrase_table(args.limit, args.search))


def _handle_lookup(args: argparse.Namespace) -> None:
    entry = get_phrase_entry(args.phrase)
    if entry is None:
        raise ValueError(f"No exact phrase entry found for {args.phrase!r}.")

    if args.show_entry or args.json:
        if args.json:
            console.print_json(json.dumps(entry))
            return
        table = Table(title=f"Phrase Lookup: {entry['latin']}", header_style="bold magenta")
        table.add_column("Language", style="cyan")
        table.add_column("Translation", style="white")
        for code, translation in entry["translations"].items():
            table.add_row(f"{resolve_language_name(code)} ({code})", translation)
        console.print(table)
        return

    target = args.target or DEFAULT_TARGET_LANGUAGE
    translation = lookup_phrase(args.phrase, resolve_language_code(target))
    if translation is None:
        raise ValueError(
            f"Phrase found, but no curated translation exists for target language {target!r}."
        )
    console.print(Panel(translation, title=entry["latin"], border_style="green"))


def _handle_stats(_: argparse.Namespace) -> None:
    _print_stats()


def _handle_doctor(_: argparse.Namespace) -> None:
    _print_doctor()


def _handle_examples(_: argparse.Namespace) -> None:
    _print_examples()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="universal-translator",
        description="A polished CLI for Latin-first translation, phrase lookup, and language discovery.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate a single text snippet.",
    )
    translate_parser.add_argument("text", help="The text to translate.")
    translate_parser.add_argument("--source", default=DEFAULT_SOURCE_LANGUAGE)
    translate_parser.add_argument("--target", default=DEFAULT_TARGET_LANGUAGE)
    translate_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    translate_parser.add_argument("--max-new-tokens", type=int, default=256)
    translate_parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    translate_parser.set_defaults(handler=_handle_translate)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Translate multiple lines from arguments or a text file.",
    )
    batch_parser.add_argument("texts", nargs="*", help="Texts to translate.")
    batch_parser.add_argument("--file", help="Path to a UTF-8 text file with one phrase per line.")
    batch_parser.add_argument("--source", default=DEFAULT_SOURCE_LANGUAGE)
    batch_parser.add_argument("--target", default=DEFAULT_TARGET_LANGUAGE)
    batch_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    batch_parser.add_argument("--max-new-tokens", type=int, default=256)
    batch_parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    batch_parser.set_defaults(handler=_handle_batch)

    languages_parser = subparsers.add_parser(
        "languages",
        help="Browse and search the supported language registry.",
    )
    languages_parser.add_argument("--search", help="Filter by name or code.")
    languages_parser.add_argument("--show-codes", action="store_true", help="Show NLLB codes.")
    languages_parser.add_argument("--limit", type=int, help="Limit the number of rows displayed.")
    languages_parser.set_defaults(handler=_handle_languages)

    phrases_parser = subparsers.add_parser(
        "phrases",
        help="Browse the built-in Latin phrase database.",
    )
    phrases_parser.add_argument("--search", help="Filter phrases by substring.")
    phrases_parser.add_argument("--limit", type=int, help="Limit the number of phrases displayed.")
    phrases_parser.set_defaults(handler=_handle_phrases)

    lookup_parser = subparsers.add_parser(
        "lookup",
        help="Look up a phrase entry in the local database.",
    )
    lookup_parser.add_argument("phrase", help="Exact Latin phrase to look up.")
    lookup_parser.add_argument("--target", help="Target language alias or code for quick lookup.")
    lookup_parser.add_argument(
        "--show-entry",
        action="store_true",
        help="Show the full translation entry across all curated languages.",
    )
    lookup_parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    lookup_parser.set_defaults(handler=_handle_lookup)

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show translator, phrase, and language registry statistics.",
    )
    stats_parser.set_defaults(handler=_handle_stats)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect local runtime and dependency health.",
    )
    doctor_parser.set_defaults(handler=_handle_doctor)

    examples_parser = subparsers.add_parser(
        "examples",
        help="Show example CLI commands.",
    )
    examples_parser.set_defaults(handler=_handle_examples)

    return parser


def _normalize_argv(argv: Optional[Sequence[str]]) -> List[str]:
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    if not args:
        return ["examples"]
    if args[0] in KNOWN_COMMANDS or args[0].startswith("-"):
        return args
    return ["translate", *args]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    normalized_argv = _normalize_argv(argv)
    args = parser.parse_args(normalized_argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return

    try:
        args.handler(args)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
