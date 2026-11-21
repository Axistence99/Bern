#!/usr/bin/env python3
"""Build the browser demo into one self-contained HTML file.

`web/demo/index.html` has to work with no network at all: it is opened from a `file://` URL or
inside an iframe whose sandbox blocks external requests. So every dependency is inlined —
React, ReactDOM, the compiled Bern runtime, the intent model and the Tailwind CSS — instead of
loaded from a CDN, where none of it would arrive.

The JavaScript sources are readable and committed (`app.jsx`, and the runtime comes from
`react-native/bernFeatures.ts` + `bernOnDevice.ts`). This script is the only build step.

    python3 scripts/build_web_demo.py

Needs `node` and, on first run, downloads the Tailwind standalone binary into `.cache/`
(gitignored). Neither is needed to *serve* the demo, only to rebuild it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "web" / "demo"
RUNTIME_SOURCES = [REPO / "react-native" / "bernFeatures.ts", REPO / "react-native" / "bernOnDevice.ts"]
INTENT_MODEL = REPO / "models" / "ondevice" / "intent.json"
CACHE = REPO / ".cache"
NODE_MODULES = CACHE / "node_modules"

TAILWIND_URL = "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.5/tailwindcss-linux-x64"
REACT_URLS = {
    "react.js": "https://unpkg.com/react@18.3.1/umd/react.production.min.js",
    "react-dom.js": "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js",
}


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(command)}\n{result.stdout}{result.stderr}")
    return result


def ensure_react() -> dict[str, str]:
    """Fetch React's UMD builds once and keep them in the cache."""
    CACHE.mkdir(exist_ok=True)
    sources = {}
    for name, url in REACT_URLS.items():
        path = CACHE / name
        if not path.exists():
            print(f"  downloading {name} ...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        sources[name] = path.read_text(encoding="utf-8")
    return sources


def ensure_tailwind() -> Path:
    binary = CACHE / "tailwindcss"
    if not binary.exists():
        print("  downloading the Tailwind standalone binary (107 MB) ...", flush=True)
        with urllib.request.urlopen(TAILWIND_URL, timeout=300) as response:
            binary.write_bytes(response.read())
        binary.chmod(0o755)
    return binary


def ensure_typescript() -> Path:
    """The same compiler the parity check uses; installed into the cache, not the repo."""
    compiler = NODE_MODULES / ".bin" / "tsc"
    if not compiler.exists():
        print("  installing typescript ...", flush=True)
        CACHE.mkdir(exist_ok=True)
        run(["npm", "install", "--silent", "--no-audit", "--no-fund", "--prefix", str(CACHE), "typescript"])
    return compiler


def compile_runtime() -> tuple[str, str]:
    """Compile the React Native TypeScript to one ES module the browser can import.

    `--lib ES2020` is deliberate: the runtime must not touch any browser API, and compiling it
    without the DOM lib is what proves that.
    """
    compiler = ensure_typescript()
    out = CACHE / "runtime"
    if out.exists():
        shutil.rmtree(out)
    run([
        str(compiler), "--outDir", str(out), "--module", "ES2020", "--target", "ES2020",
        "--lib", "ES2020", "--strict", "--skipLibCheck",
        *[str(source) for source in RUNTIME_SOURCES],
    ])
    features = (out / "bernFeatures.js").read_text(encoding="utf-8")
    ondevice = (out / "bernOnDevice.js").read_text(encoding="utf-8")
    # The two files import each other by name; one module is simpler to inline.
    ondevice = ondevice.replace("from './bernFeatures'", "from './bernFeatures.js'")
    return features, ondevice


def bundle_runtime(features: str, ondevice: str) -> str:
    """Stitch the two ES modules into one script, scoped so nothing leaks onto `window`.

    The two files import from each other by module name. Rather than teach the browser to
    resolve that, both are concatenated inside a single IIFE with the import statement removed
    and the exports stripped — the names are already in scope for each other.
    """
    parts = []
    for source in (features, ondevice):
        for line in source.splitlines():
            if line.startswith("import "):
                continue
            if line.startswith("export class "):
                line = line[len("export "):]
            elif line.startswith("export function "):
                line = line[len("export "):]
            elif line.startswith("export const "):
                line = line[len("export "):]
            parts.append(line)

    body = "\n".join(parts)
    return (
        "window.__BERN_RUNTIME__ = (function () {\n"
        "\"use strict\";\n"
        + body
        + "\nreturn { BernOnDevice, glyphFeatures, inkMask, normaliseGlyph, softmaxScores, unionRow };\n"
        "})();\n"
    )


def compile_app(compiler: Path) -> str:
    """Compile app.jsx to plain ES2020 with `React.createElement` calls.

    Using TypeScript's own JSX transform rather than inlining Babel: Babel would add three
    megabytes to a page whose whole point is to load with no network. `--jsx react` emits the
    classic runtime, which is what React's UMD build expects — a JSX pragma of
    `React.createElement` makes the factory name explicit instead of relying on globals.
    """
    source_dir = CACHE / "app"
    source_dir.mkdir(parents=True, exist_ok=True)
    staged = source_dir / "app.tsx"
    # `@ts-nocheck` because this is JavaScript with JSX, not typed TypeScript: React arrives as
    # a UMD global and there are no type definitions for it. The compiler is being used purely
    # as a JSX transform here, and a type error would say nothing about whether the page works.
    staged.write_text(
        "// @ts-nocheck\n" + (DEMO / "app.jsx").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run([
        str(compiler), "--jsx", "react", "--jsxFactory", "React.createElement",
        "--target", "ES2020", "--module", "ES2020", "--lib", "ES2020,DOM",
        "--allowJs", "false", "--skipLibCheck", "--noEmitOnError", "false",
        "--outDir", str(source_dir), str(staged),
    ])
    emitted = source_dir / "app.js"
    if not emitted.exists():
        raise SystemExit("TypeScript emitted no output for app.jsx")
    compiled = emitted.read_text(encoding="utf-8")
    return "\n".join(line for line in compiled.splitlines() if not line.startswith("export "))


def build_tailwind() -> str:
    """Compile Tailwind against the demo sources, dark mode driven by a class."""
    binary = ensure_tailwind()
    entry = CACHE / "tailwind-input.css"
    entry.write_text(
        '@import "tailwindcss";\n'
        f'@source "{DEMO / "app.jsx"}";\n'
        '@custom-variant dark (&:where(.dark, .dark *));\n',
        encoding="utf-8",
    )
    output = CACHE / "tailwind-output.css"
    run([str(binary), "-i", str(entry), "-o", str(output), "--minify"])
    return output.read_text(encoding="utf-8")


def main() -> int:
    for path in [*RUNTIME_SOURCES, INTENT_MODEL, DEMO / "app.jsx"]:
        if not path.exists():
            raise SystemExit(f"missing {path}")

    print("react ...", flush=True)
    react = ensure_react()

    print("runtime ...", flush=True)
    features, ondevice = compile_runtime()
    runtime = bundle_runtime(features, ondevice)

    print("tailwind ...", flush=True)
    css = build_tailwind()

    print("app ...", flush=True)
    app = compile_app(ensure_typescript())
    app = app.replace("__INTENT_BUNDLE__", "window.__BERN__.INTENT_BUNDLE")
    # Check the code, not the comment that documents the placeholders.
    unresolved = [token for token in ("__INTENT_BUNDLE__",) if token in app]
    if unresolved:
        raise SystemExit(f"build placeholders survived substitution: {unresolved}")

    model = INTENT_MODEL.read_text(encoding="utf-8")
    # json.dumps for the wrapper so the model text lands verbatim inside a JS literal.
    runtime_script = "window.__BERN__ = { INTENT_BUNDLE: " + model + " };\n" + runtime

    html = PAGE.format(
        css=css,
        react=react["react.js"],
        react_dom=react["react-dom.js"],
        runtime_setup=runtime_script,
        app=app,
    )
    destination = DEMO / "index.html"
    destination.write_text(html, encoding="utf-8")
    print(f"\nwrote {destination.relative_to(REPO)}  ({destination.stat().st_size / 1024:.0f} KB)")
    return 0


PAGE = """<!doctype html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bern — try the model</title>
<style>
{css}
html, body {{ margin: 0; background: #020617; }}
</style>
</head>
<body>
<div id="root"></div>
<script>{react}</script>
<script>{react_dom}</script>
<script>{runtime_setup}</script>
<script>
{app}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
