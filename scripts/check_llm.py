#!/usr/bin/env python3
"""Report whether Bern's language-model layer is configured, and test it if it is.

Bern answers questions it was trained on with a local classifier. Everything else — general
knowledge, explanations, writing, anything open-ended — needs a language model. That layer is
off unless ``BERN_LLM_API_KEY`` is set, so this script exists to answer one question before
anyone wonders why Bern refuses general questions: *is a model wired up, and does it work?*

It reads settings from the environment and prints what it finds. It never prints the key.

Usage:
    PYTHONPATH=. python3 scripts/check_llm.py
    PYTHONPATH=. python3 scripts/check_llm.py --ask "who wrote hamlet"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bern import llm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Bern's language-model configuration.")
    parser.add_argument("--ask", metavar="QUESTION", help="send one question and print the answer")
    parser.add_argument("--session", default="check", help="conversation id for --ask")
    args = parser.parse_args()

    config = llm.LLMConfig()
    print(f"BERN_LLM_API_KEY    {'set (' + str(len(config.api_key.strip())) + ' chars)' if config.available else 'NOT SET'}")
    print(f"BERN_LLM_MODEL      {config.model}")
    print(f"BERN_LLM_BASE_URL   {config.base_url}")
    print()

    if not config.available:
        print("The language model is off. Bern will answer only what it was trained on and")
        print("refuse the rest, saying so.")
        print()
        print("To enable general chat, set a key and re-run:")
        print("  export BERN_LLM_API_KEY=...            # any OpenAI-compatible provider")
        print("  export BERN_LLM_BASE_URL=...           # optional: Ollama, llama.cpp, LM Studio")
        print("  export BERN_LLM_MODEL=...              # optional: model name at that endpoint")
        return 1

    if not args.ask:
        print("A key is present. Add --ask \"your question\" to send one and see the answer.")
        return 0

    try:
        result = llm.session(args.session).ask(args.ask)
    except llm.LLMError as problem:
        print(f"The model is configured but the call failed:\n  {problem}")
        return 2

    print(f"answer ({result['engine']}):")
    print(result["answer"] or "(empty answer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
