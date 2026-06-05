# pylatin-translator

[![Release](https://github.com/your-username/language-translator/actions/workflows/release.yml/badge.svg)](https://github.com/your-username/language-translator/actions/workflows/release.yml)
[![PyPI version](https://img.shields.io/pypi/v/pylatin-translator.svg)](https://pypi.org/project/pylatin-translator/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylatin-translator.svg)](https://pypi.org/project/pylatin-translator/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

`pylatin-translator` is a Python-first Latin translation toolkit with three major goals:

1. Make Latin -> English translation pleasant from the command line.
2. Provide a reusable Python API for application developers.
3. Combine curated phrase lookups with model-based translation so the project can be both practical and extensible.

This repository started from a small package skeleton and has grown into a richer tool with:

- a large local phrase JSON database
- a polished command-line interface
- 200+ supported language-script entries in its registry
- dedicated alternate Latin translation fallbacks
- release automation for package and CLI artifacts

The package is opinionated in a useful way:

- Latin is treated as a first-class use case.
- Exact phrase lookups are preferred when available.
- Rich terminal output is used for a friendlier CLI.
- Structured JSON output is available for automation.
- A standalone executable release flow is included for non-Python users.

This README is intentionally long and detailed. It is designed to serve as:

- a project overview
- a usage guide
- a CLI manual
- a packaging and release reference
- an implementation map
- a contributor onboarding document

## Table of Contents

1. Project Overview
2. Why This Project Exists
3. What The Project Does Well
4. Current Translation Strategy
5. Feature Summary
6. Supported Workflows
7. Installation
8. Quick Start
9. Python API
10. CLI Overview
11. CLI Command Reference
12. CLI Examples
13. Output Formats
14. Phrase Database
15. Language Registry
16. Latin Translation Routing
17. Translation Providers
18. Release Artifacts
19. Packaging Details
20. Repository Layout
21. Development Workflow
22. Troubleshooting
23. Limitations
24. Roadmap Ideas
25. FAQ
26. Contributing Notes
27. License

## Project Overview

`pylatin-translator` is built around a layered translation design.

At a high level, the project does not rely on a single translation mechanism. Instead, it routes requests through a sequence of increasingly general strategies:

1. Curated phrase database lookup for exact Latin matches.
2. Dedicated Latin translation models for Latin <-> English.
3. General multilingual models for non-Latin work.
4. Bridge translation through English when Latin is involved with another language.

This gives the project a better balance between:

- speed
- explainability
- extensibility
- offline-style local lookup behavior
- broader multilingual reach

The project is implemented as a Python package with the import path `universal_translator`, while the distribution name is `pylatin-translator`.

That means:

- the package you install is named `pylatin-translator`
- the module you import is `universal_translator`
- the CLI entrypoint is `universal-translator`

## Why This Project Exists

Latin translation sits in an awkward spot in the modern tooling ecosystem.

Many multilingual models do not prioritize Latin.
Some general-purpose models can translate short Latin phrases, but not consistently.
Other projects include Latin in broad multilingual group models, but the user experience around them is often rough.

This repository exists to smooth out that experience.

The core design goals are:

- make Latin translation approachable
- keep the CLI comfortable for non-experts
- expose enough metadata to make the tool scriptable
- retain transparency about where translations come from
- make room for curated domain-specific phrase expansion

## What The Project Does Well

The current version is especially strong for these scenarios:

- exact lookup of common Latin mottos and phrases
- command-line exploration of Latin phrases
- quick Latin -> English translation for known short text
- browsing and searching a large multilingual registry
- using a clean Python API in a script or app
- producing simple release artifacts for package and CLI distribution

## Current Translation Strategy

The project uses multiple sources.

### Phrase Database

If the input is Latin and the text exactly matches a phrase in the curated database, the project uses that entry first.

This path is:

- fast
- deterministic
- transparent
- easy to audit

### Dedicated Latin Fallback

If a curated phrase match is not found, Latin can fall back to:

- `Helsinki-NLP/opus-mt-ine-en` for Latin -> English
- `Helsinki-NLP/opus-mt-en-ine` for English -> Latin

These are model-based paths and can handle more free-form input than the phrase database, though output quality can vary.

### General Multilingual Model

For non-Latin multilingual translation, the project uses:

- `facebook/nllb-200-distilled-600M`

This model provides the broad registry support that powers the larger cross-language workflow.

### Bridge Translation

If Latin is involved but the target is neither direct English nor a curated phrase lookup, the tool can route through English as an intermediate representation.

For example:

- Latin -> English -> French
- Spanish -> English -> Latin

This is not a perfect solution, but it gives the project wider functional coverage than a strict one-model design.

## Feature Summary

- Curated local phrase database stored in `universal_translator/phrases.py`
- 200+ language-script entries in the generated registry
- Rich CLI using `rich`
- JSON output for automation
- Batch translation command
- Phrase lookup command
- Language registry browser
- Environment diagnostics command
- Release automation for package and standalone CLI
- Flit packaging
- PyInstaller release builder

## Supported Workflows

The current project supports several distinct ways of working.

### Workflow 1: Quick CLI Translation

You have a phrase and want a quick answer.

Example:

```bash
universal-translator "Salve amice"
```

### Workflow 2: Structured CLI Translation

You want explicit source and target settings.

Example:

```bash
universal-translator translate --source latin --target english "memento mori"
```

### Workflow 3: Batch Translation

You want to translate many lines in one pass.

Example:

```bash
universal-translator batch --file phrases.txt --target spanish
```

### Workflow 4: Phrase Database Exploration

You want to browse or inspect the curated phrase collection.

Examples:

```bash
universal-translator phrases --limit 25
universal-translator phrases --search amor
universal-translator lookup "carpe diem" --show-entry
```

### Workflow 5: Language Registry Exploration

You want to search the full registry of language names and codes.

Examples:

```bash
universal-translator languages --show-codes
universal-translator languages --search arabic
universal-translator languages --search latn --show-codes
```

### Workflow 6: Python Integration

You want to import the package in your own script or service.

### Workflow 7: Release Build

You want package artifacts and a standalone CLI executable.

## Installation

### Requirements

- Python 3.10 or later
- Internet access for first-time model downloads
- enough disk space for Hugging Face model caching

### Main Dependencies

- `transformers`
- `torch`
- `sentencepiece`
- `pycountry`
- `rich`
- `sacremoses`

### Install For Local Development

```bash
python -m pip install flit
flit install
```

### Install Release Extras

```bash
python -m pip install .[release]
```

This installs tools needed for standalone artifact generation, including `PyInstaller`.

### Install From Built Wheel

If you already built the wheel:

```bash
python -m pip install dist/pylatin_translator-0.1.0-py3-none-any.whl
```

### Fresh Environment Example

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install flit
flit install
```

## Quick Start

### Fastest Possible Command

```bash
universal-translator "carpe diem"
```

### Translate With Explicit Command Syntax

```bash
universal-translator translate "carpe diem"
```

### Translate Into Another Language

```bash
universal-translator translate --source latin --target french "memento mori"
```

### Inspect Supported Languages

```bash
universal-translator languages --show-codes --limit 20
```

### Inspect Phrase Entry

```bash
universal-translator lookup "veni vidi vici" --show-entry
```

### Get JSON Output

```bash
universal-translator translate "carpe diem" --json
```

## Python API

The package exports a small but useful set of functions and classes.

### Main Imports

```python
from universal_translator import UniversalTranslator
from universal_translator import TranslationResult
from universal_translator import translate_text
from universal_translator import translate_from_latin
from universal_translator import get_phrase_entry
from universal_translator import list_phrase_entries
from universal_translator import list_latin_phrases
from universal_translator import list_language_entries
from universal_translator import list_supported_languages
from universal_translator import resolve_language_code
from universal_translator import resolve_language_name
```

### Basic Translator Example

```python
from universal_translator import UniversalTranslator

translator = UniversalTranslator()
result = translator.translate("carpe diem")
print(result.text)
print(result.provider)
```

### Translate With Specific Languages

```python
from universal_translator import UniversalTranslator

translator = UniversalTranslator()
result = translator.translate(
    text="memento mori",
    source_language="latin",
    target_language="french",
)
print(result.text)
```

### Translate From Latin Helper

```python
from universal_translator import translate_from_latin

text = translate_from_latin("omnia vincit amor", target_language="english")
print(text)
```

### Use Batch Translation

```python
from universal_translator import UniversalTranslator

translator = UniversalTranslator()
results = translator.translate_batch(
    texts=[
        "carpe diem",
        "memento mori",
        "veni vidi vici",
    ],
    source_language="latin",
    target_language="english",
)

for item in results:
    print(item.text, item.provider)
```

### Inspect Phrase Entries

```python
from universal_translator import get_phrase_entry

entry = get_phrase_entry("carpe diem")
print(entry)
```

### Inspect Language Entries

```python
from universal_translator import list_language_entries

entries = list_language_entries()
print(entries[0])
print(len(entries))
```

### Resolve Language Codes

```python
from universal_translator import resolve_language_code

print(resolve_language_code("english"))
print(resolve_language_code("fra_Latn"))
print(resolve_language_code("South Levantine Arabic"))
```

### Understand `TranslationResult`

`TranslationResult` includes:

- `text`
- `source_language`
- `target_language`
- `model_name`
- `provider`

Useful derived properties:

- `source_language_name`
- `target_language_name`

The `provider` field is especially helpful because it tells you which path produced the translation:

- `phrase_database`
- `latin_model`
- `latin_bridge`
- `model`

## CLI Overview

The CLI is built around subcommands, but it still keeps a soft landing for casual use.

If you run:

```bash
universal-translator "salve"
```

the CLI treats it as:

```bash
universal-translator translate "salve"
```

If you run the command with no arguments:

```bash
universal-translator
```

it shows example usage via the `examples` command.

This is intentional:

- beginners get a gentle experience
- advanced users still have explicit commands
- scripts can use structured subcommands

## CLI Command Reference

This section documents every CLI command in detail.

### `translate`

Translate a single text snippet.

Basic form:

```bash
universal-translator translate "carpe diem"
```

Options:

- `--source`
- `--target`
- `--model`
- `--max-new-tokens`
- `--json`

Example:

```bash
universal-translator translate --source latin --target english "arma virumque cano"
```

Example with JSON:

```bash
universal-translator translate "carpe diem" --json
```

Expected behaviors:

- exact phrase lookup uses curated output
- Latin misses may use Latin model fallback
- non-Latin multilingual requests use the NLLB path

### `batch`

Translate multiple inputs in one invocation.

Accepted inputs:

- positional text values
- a text file using `--file`
- both together

Example with direct values:

```bash
universal-translator batch "carpe diem" "memento mori" "veni vidi vici"
```

Example with file:

```bash
universal-translator batch --file phrases.txt --target german
```

Options:

- `--file`
- `--source`
- `--target`
- `--model`
- `--max-new-tokens`
- `--json`

When using JSON output, each result includes:

- input text
- output text
- source language code
- target language code
- model name
- provider

### `languages`

Browse and search the language registry.

This registry is generated from the official NLLB tokenizer codes plus the manual Latin entry used by the project.

Example:

```bash
universal-translator languages
```

Search example:

```bash
universal-translator languages --search arabic
```

Show codes:

```bash
universal-translator languages --show-codes
```

Limit rows:

```bash
universal-translator languages --limit 10
```

Options:

- `--search`
- `--show-codes`
- `--limit`

### `phrases`

Browse the curated Latin phrase database.

Example:

```bash
universal-translator phrases
```

Search example:

```bash
universal-translator phrases --search amor
```

Limit example:

```bash
universal-translator phrases --limit 50
```

Options:

- `--search`
- `--limit`

### `lookup`

Look up one exact phrase entry.

This command is useful when you want to inspect the local curated data without triggering model translation.

Quick lookup:

```bash
universal-translator lookup "carpe diem"
```

Full entry:

```bash
universal-translator lookup "carpe diem" --show-entry
```

Specific target:

```bash
universal-translator lookup "carpe diem" --target french
```

JSON form:

```bash
universal-translator lookup "carpe diem" --show-entry --json
```

Options:

- `--target`
- `--show-entry`
- `--json`

### `stats`

Show high-level project statistics.

This command summarizes:

- CLI version
- default model
- Latin fallback models
- phrase entry count
- language entry count
- base language count
- default source and target

Usage:

```bash
universal-translator stats
```

### `doctor`

Show local environment diagnostics.

This command helps identify runtime issues.

It reports:

- Python version
- platform
- Torch version
- whether CUDA is available
- phrase count
- language count
- default model
- Latin fallback model

Usage:

```bash
universal-translator doctor
```

### `examples`

Show ready-to-copy command examples.

Usage:

```bash
universal-translator examples
```

## CLI Examples

This section gives a broader set of real examples.

### Translate a known phrase

```bash
universal-translator translate "memento mori"
```

### Translate a Latin phrase to Portuguese

```bash
universal-translator translate --source latin --target portuguese "omnia vincit amor"
```

### Translate English to Latin

```bash
universal-translator translate --source english --target latin "knowledge is power"
```

### Translate several Latin phrases in one call

```bash
universal-translator batch \
  "carpe diem" \
  "memento mori" \
  "in vino veritas"
```

### Translate lines from a file into French

```bash
universal-translator batch --file phrases.txt --target french
```

### Search for Arabic variants in the registry

```bash
universal-translator languages --search arabic --show-codes
```

### Search for all Latin-script registry entries

```bash
universal-translator languages --search latn --show-codes
```

### Show only a few supported languages

```bash
universal-translator languages --limit 15
```

### Show phrases related to love

```bash
universal-translator phrases --search amor
```

### Show a complete phrase entry

```bash
universal-translator lookup "scientia potentia est" --show-entry
```

### Get JSON output for scripting

```bash
universal-translator translate "carpe diem" --json
```

### Save JSON output to a file

```bash
universal-translator translate "carpe diem" --json > result.json
```

### Inspect environment state

```bash
universal-translator doctor
```

### Get project stats

```bash
universal-translator stats
```

## Output Formats

The CLI supports two broad output styles.

### Rich Human Output

Most commands default to rich terminal output:

- tables
- panels
- consistent formatting
- clearer visual grouping

This is ideal for humans reading the terminal directly.

### JSON Output

Selected commands support `--json`.

This is ideal for:

- shell scripting
- integration tests
- automation
- wrappers
- downstream tooling

Commands that support JSON:

- `translate`
- `batch`
- `lookup`

## Phrase Database

The phrase database lives in:

```text
universal_translator/phrases.py
```

It is implemented as:

- a JSON string payload embedded in Python
- parsed on demand
- cached with `lru_cache`

This file currently stores:

- curated Latin phrase and word entries
- helper functions for indexing and lookup
- generated language registry metadata

### Why Keep The Database In A Python File

There are tradeoffs here.

Benefits:

- no separate asset loading logic
- simple packaging
- easy import path
- easier shipping in single-file distributions

Costs:

- large source file
- harder manual editing at scale
- less ideal for collaborative data entry

For this project stage, the Python-embedded JSON approach is acceptable and keeps the release flow simple.

### Phrase Entry Shape

Each phrase entry follows this pattern:

```json
{
  "latin": "carpe diem",
  "category": "phrase",
  "translations": {
    "eng_Latn": "seize the day",
    "fra_Latn": "profite du jour",
    "spa_Latn": "aprovecha el dia",
    "deu_Latn": "nutze den tag",
    "ita_Latn": "cogli l attimo",
    "por_Latn": "aproveita o dia"
  }
}
```

### Current Curated Translation Languages

The curated phrase entries currently focus on:

- English
- French
- Spanish
- German
- Italian
- Portuguese

These entries are intentionally hand-curated rather than generated at runtime.

### Phrase Database APIs

The phrase module exposes:

- `load_phrase_database()`
- `get_phrase_entry()`
- `lookup_phrase()`
- `list_phrase_entries()`
- `list_latin_phrases()`
- `list_phrase_languages()`

### Phrase Lookup Behavior

Phrase lookup is exact after normalization.

Normalization currently:

- trims whitespace
- lowercases text
- collapses repeated internal whitespace

That means:

- `carpe diem`
- `  carpe   diem  `
- `CARPE DIEM`

all normalize to the same phrase key.

## Language Registry

The language registry is one of the more important internal pieces of the project.

It provides:

- readable language names
- NLLB codes
- script names
- alias resolution

### Registry Source

The registry is built from:

- the official NLLB tokenizer codes bundled with `transformers`
- one manual Latin entry added by the project

This gives the project:

- compatibility with the model ecosystem
- a stable source of truth
- better maintainability than a handwritten giant table

### Registry Counts

At the time of writing in this repository:

- language-script entries: 203
- base languages: 197

These counts may change if the upstream tokenizer list or local manual additions change.

### Registry APIs

The registry-related APIs include:

- `list_language_entries()`
- `list_supported_languages()`
- `resolve_language_code()`
- `resolve_language_name()`

### Alias Behavior

Users do not always want to type raw codes like `fra_Latn`.

The CLI and API therefore accept:

- full NLLB codes
- human-readable names
- some convenience aliases

Examples:

- `english`
- `french`
- `latin`
- `South Levantine Arabic`
- `fra_Latn`

## Latin Translation Routing

Latin is a special case in this project.

### Why Latin Needs Special Handling

The multilingual model strategy that works for many modern languages is not as clean for Latin.

Latin support is scattered across model ecosystems.
Performance varies by model family.
The best route depends on whether the text is:

- a known classical phrase
- a short motto
- a free-form sentence
- English that needs to become Latin
- Latin that needs to bridge to another target language

### Routing Order

When Latin is involved, the project uses the following order:

1. exact curated phrase lookup
2. Latin <-> English specialized model path
3. English bridge route when translating Latin to another supported target or back

### Why Phrase Lookup Comes First

Known phrases often benefit most from:

- stable translations
- culturally familiar wording
- predictable output

Phrase-first routing helps avoid weird model paraphrases on canonical phrases.

### Why Dedicated Latin Models Come Next

If a phrase is not in the curated database, the next best route is usually a Latin-capable translation model rather than pretending every multilingual model handles Latin equally well.

### Why Bridge Through English

Latin <-> all-other-language coverage is difficult to maintain directly.

English provides:

- the strongest shared intermediate path
- better general model support
- simpler routing logic

It is not perfect, but it is pragmatic.

## Translation Providers

The project exposes translation provenance through the `provider` field.

### `phrase_database`

Used when a local curated phrase entry matches exactly.

### `latin_model`

Used when a dedicated Latin model handled the translation directly.

### `latin_bridge`

Used when the tool translated through English to or from Latin.

### `model`

Used when the general multilingual NLLB path handled the translation.

This is useful because users and developers can make decisions based on route quality.

For example:

- prefer `phrase_database` when available
- review `latin_model` output on difficult poetry
- be cautious with `latin_bridge` for precision-critical translation

## Release Artifacts

The repository supports two release artifact types.

### Python Package Artifacts

These are built via Flit:

- source distribution
- wheel

Example:

```bash
flit build
```

### Standalone CLI Artifacts

These are built via PyInstaller:

- single-file executable

The release builder script is:

```text
scripts/build_release.py
```

### Local Release Flow

```bash
python -m pip install .[release]
python scripts/build_release.py
```

This creates artifacts and copies them into:

```text
release-artifacts/
```

### GitHub Actions Release Workflow

The workflow file is:

```text
.github/workflows/release.yml
```

It currently:

- runs on version tags
- builds on Windows, macOS, and Linux
- uploads artifacts for each platform

### Artifact Types You Should Expect

- `pylatin_translator-<version>.tar.gz`
- `pylatin_translator-<version>-py3-none-any.whl`
- standalone CLI executable for each release OS

## Packaging Details

The project uses Flit.

### Why Flit

Flit is a good fit here because:

- the package layout is simple
- publishing metadata is straightforward
- the project benefits from lightweight build config

### Distribution Name vs Import Name

This repository uses:

- distribution name: `pylatin-translator`
- import package: `universal_translator`

This is normal in Python packaging, but worth remembering.

### Important Metadata File

The package metadata is configured in:

```text
pyproject.toml
```

### Main Packaging Commands

Build:

```bash
flit build
```

Install locally:

```bash
flit install
```

Install release extras:

```bash
python -m pip install .[release]
```

## Repository Layout

Current high-level layout:

```text
.
├── .github/
│   └── workflows/
│       └── release.yml
├── scripts/
│   └── build_release.py
├── universal_translator/
│   ├── __init__.py
│   ├── __main__.py
│   ├── phrases.py
│   └── translator.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

### File Responsibilities

`universal_translator/translator.py`

- CLI
- routing logic
- core translator class
- result data structure

`universal_translator/phrases.py`

- curated phrase data
- phrase lookup helpers
- language registry generation

`universal_translator/__init__.py`

- public package exports

`universal_translator/__main__.py`

- module entrypoint

`scripts/build_release.py`

- local release artifact builder

`.github/workflows/release.yml`

- CI release artifact workflow

## Development Workflow

If you want to keep developing the project locally, this is a good loop.

### 1. Install

```bash
python -m pip install flit
flit install
```

### 2. Run CLI examples

```bash
python -m universal_translator examples
```

### 3. Inspect stats

```bash
python -m universal_translator stats
```

### 4. Test exact phrase lookup

```bash
python -m universal_translator translate "carpe diem"
```

### 5. Test Latin fallback

```bash
python -m universal_translator translate "arma virumque cano"
```

### 6. Rebuild package artifacts

```bash
flit build
```

### 7. Build release artifacts when needed

```bash
python -m pip install .[release]
python scripts/build_release.py
```

## Troubleshooting

This section covers common rough edges.

### Problem: First translation is slow

Cause:

- model download
- model cache initialization
- weight loading

What to expect:

- the first call can take significantly longer than later calls

### Problem: Hugging Face cache warning about symlinks on Windows

Cause:

- Windows symlink permissions are restricted in many environments

Impact:

- the cache still works
- it may use more disk space

This is a warning, not usually a functional failure.

### Problem: Latin output looks odd

Cause:

- Latin model quality varies
- poetic source can be hard
- bridge translations compound errors

Recommendation:

- prefer phrase database hits for canonical short phrases
- treat model output as assistive rather than absolute truth

### Problem: `CUDA Available` is false

Cause:

- CPU-only torch install
- no supported GPU
- no configured CUDA environment

Impact:

- translations still work
- inference may be slower

### Problem: a language name is not recognized

Try:

- `universal-translator languages --search <name>`
- use `--show-codes`
- pass the exact NLLB code if needed

### Problem: batch command returns an error

Cause:

- empty input
- file missing
- all lines blank

Check:

- file path
- file encoding
- that there is at least one non-empty line

### Problem: release build fails

Check:

- `python -m pip install .[release]`
- `PyInstaller` availability
- write permissions for `dist/` and `release-artifacts/`

## Limitations

This project is useful, but it is not magic.

### Important Current Limits

- Latin model quality is uneven on difficult or poetic text.
- Phrase database coverage is large but not exhaustive.
- Curated phrase translations currently focus on six target languages.
- Bridge translation can reduce precision.
- The phrase database being embedded in Python is convenient but not ideal long-term for very large datasets.
- Model downloads still require internet access on first use.

### What This Project Is Not

- not a scholarly critical edition tool
- not a full morphology analyzer
- not a parser for inflected Latin grammar
- not a guaranteed high-accuracy translation system for all Latin prose and poetry

## Roadmap Ideas

Potential future directions:

- split phrase data into external JSON assets
- add search ranking for fuzzy phrase lookup
- add morphology-aware Latin normalization
- add confidence and route scoring
- add caching for repeated model results
- add unit tests and golden phrase snapshots
- add CSV export for phrase inventory
- add shell completion scripts
- add web or TUI frontends
- add pronunciation or transliteration helpers

## FAQ

### Why is the package called `pylatin-translator` but imported as `universal_translator`?

Because the distribution name and import package name do not have to be identical in Python packaging.

### Does this work offline?

Phrase lookup works once the package is installed.
Model-based translation requires model files to exist locally.
If they are not cached yet, the first use requires downloads.

### Can I use this for non-Latin languages?

Yes.
The registry and the NLLB model path support broader multilingual workflows.

### Is Latin really supported by the main NLLB path?

The package treats Latin specially instead of assuming equal support across all models.
That is why it includes dedicated Latin routing.

### Can I automate this tool from scripts?

Yes.
Use the Python API or the CLI with `--json`.

### Can I ship this as a standalone executable?

Yes.
Use the release extras and the PyInstaller-based release builder.

### Can I add more phrases?

Yes.
The current phrase database is intentionally easy to extend, though it is now very large.

### Can I swap models?

Yes.
The main `translate` path accepts a `--model` flag for the NLLB route, and the code can be edited further if you want configurable Latin fallback models too.

## Contributing Notes

If you want to keep improving this project, here are the safest patterns.

### Good Changes

- add tested phrase entries
- improve CLI help text
- refine language aliases
- add better release documentation
- add tests
- improve route transparency

### Be Careful With

- silently changing phrase translations
- removing aliases
- renaming commands without documentation updates
- changing release artifact names
- introducing huge dependencies without a clear reason

### Suggested Contribution Order

1. Improve correctness.
2. Improve observability.
3. Improve UX.
4. Improve packaging and release flow.
5. Improve scale and maintainability.

## License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

## Final Notes

This README is intentionally long because the project now covers several different concerns:

- translation behavior
- CLI usage
- packaging
- release flow
- phrase database structure
- model routing

If you are a casual user, start with:

- `Install`
- `Quick Start`
- `CLI Examples`

If you are integrating the tool into code, start with:

- `Python API`
- `Translation Providers`
- `Latin Translation Routing`

If you are packaging or publishing the tool, start with:

- `Release Artifacts`
- `Packaging Details`
- `Repository Layout`

If you are maintaining the project, start with:

- `Development Workflow`
- `Troubleshooting`
- `Contributing Notes`

That should give you a strong working picture of how the repository is meant to behave today, where it is strongest, and where future improvements will be most valuable.
