#!/usr/bin/env python3
"""Build the browser demo into one self-contained HTML file.

`web/demo/index.html` has to work with no network at all: it is opened from a `file://` URL or
inside an iframe whose sandbox blocks external requests. So every dependency is inlined —
React, ReactDOM, the compiled Bern runtime, the intent model, the two typefaces and the
stylesheet — instead of loaded from a CDN, where none of it would arrive.

The JavaScript sources are readable and committed (`app.jsx`, and the runtime comes from
`react-native/bernFeatures.ts` + `bernOnDevice.ts`). This script is the only build step.

    python3 scripts/build_web_demo.py

Needs `node`. On first run it downloads React and two OFL typefaces into `.cache/`
(gitignored). None of it is needed to *view* the demo, only to rebuild it.
"""

from __future__ import annotations

import base64
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
ASSETS = DEMO / "assets"
CACHE = REPO / ".cache"
NODE_MODULES = CACHE / "node_modules"

FONT_URLS = {
    "instrument-serif.woff2": "https://fonts.gstatic.com/s/instrumentserif/v5/jizBRFtNs2ka5fXjeivQ4LroWlx-6zUTjg.woff2",
    "plex-mono-400.woff2": "https://fonts.gstatic.com/s/ibmplexmono/v23/-F63fjptAgt5VM-kVkqdyU8n1i8q1w.woff2",
    "plex-mono-600.woff2": "https://fonts.gstatic.com/s/ibmplexmono/v23/-F6qfjptAgt5VM-kVkqdyU8n3vAOwlBFgg.woff2",
}

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


def ensure_fonts() -> dict[str, str]:
    """Fetch the two typefaces once and cache them.

    Both are OFL-licensed. They are inlined rather than linked because the page must render
    correctly with no network — a webfont that fails to load is worse than no webfont, since
    the fallback stack changes every measurement in the layout.
    """
    CACHE.mkdir(exist_ok=True)
    faces = {}
    for name, url in FONT_URLS.items():
        path = CACHE / name
        if not path.exists():
            print(f"  downloading {name} ...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        faces[name] = base64.b64encode(path.read_bytes()).decode("ascii")
    return faces


def font_face_css(faces: dict[str, str]) -> str:
    """@font-face rules with the binaries embedded."""
    rules = [
        ("Instrument Serif", 400, "normal", faces["instrument-serif.woff2"]),
        ("Plex Mono", 400, "normal", faces["plex-mono-400.woff2"]),
        ("Plex Mono", 600, "normal", faces["plex-mono-600.woff2"]),
    ]
    return "\n".join(
        f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
        f"font-display:block;src:url(data:font/woff2;base64,{payload}) format('woff2');}}"
        for family, weight, style, payload in rules
    )


def data_uri(path: Path) -> str:
    """Inline a PNG as a data URI.

    The page has to work with no network, so the mascot cannot be a URL. The files used here
    are the pre-shrunk `-inline` copies, committed at their display size rather than resized at
    build time, which keeps this script free of Pillow.
    """
    if not path.exists():
        raise SystemExit(f"missing mascot image {path}")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


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
    # `--noEmitOnError` is deliberately absent and the exit status is checked instead: tsc can
    # emit half-compiled output while returning non-zero, and a page built from stale JSX fails
    # silently in the browser rather than here.
    result = subprocess.run([
        str(compiler), "--jsx", "react", "--jsxFactory", "React.createElement",
        "--target", "ES2020", "--module", "ES2020", "--lib", "ES2020,DOM",
        "--allowJs", "false", "--skipLibCheck", "--outDir", str(source_dir), str(staged),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"app.jsx did not compile:\n{result.stdout}{result.stderr}")
    emitted = source_dir / "app.js"
    if not emitted.exists():
        raise SystemExit("TypeScript emitted no output for app.jsx")
    compiled = emitted.read_text(encoding="utf-8")
    return "\n".join(line for line in compiled.splitlines() if not line.startswith("export "))


def ensure_typescript() -> Path:
    """The same compiler the parity check uses; installed into the cache, not the repo."""
    compiler = NODE_MODULES / ".bin" / "tsc"
    if not compiler.exists():
        print("  installing typescript ...", flush=True)
        CACHE.mkdir(exist_ok=True)
        run(["npm", "install", "--silent", "--no-audit", "--no-fund", "--prefix", str(CACHE), "typescript"])
    return compiler


def compile_runtime() -> tuple[str, str]:
    """Compile the React Native TypeScript to plain script.

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
            if line.startswith("export "):
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


def main() -> int:
    for path in [*RUNTIME_SOURCES, INTENT_MODEL, DEMO / "app.jsx"]:
        if not path.exists():
            raise SystemExit(f"missing {path}")

    print("react ...", flush=True)
    react = ensure_react()

    print("runtime ...", flush=True)
    features, ondevice = compile_runtime()
    runtime = bundle_runtime(features, ondevice)

    print("styles ...", flush=True)
    css = font_face_css(ensure_fonts()) + "\n" + (DEMO / "styles.css").read_text(encoding="utf-8")

    print("app ...", flush=True)
    app = compile_app(ensure_typescript())
    app = app.replace("__INTENT_BUNDLE__", "window.__BERN__.INTENT_BUNDLE")
    app = app.replace("__MASCOT_FACE__", json.dumps(data_uri(ASSETS / "logo-mark.png")))
    # Check the code, not the comment that documents the placeholders.
    unresolved = [
        token for token in ("__INTENT_BUNDLE__", "__MASCOT_FACE__") if token in app
    ]
    if unresolved:
        raise SystemExit(f"build placeholders survived substitution: {unresolved}")

    # The glyph and symbol bundles are the character readers; without them the photo path would
    # silently be classifying images as intents, which is exactly the kind of wrong answer that
    # still looks confident.
    intent = INTENT_MODEL.read_text(encoding="utf-8")
    glyph = (INTENT_MODEL.parent / "glyph.json").read_text(encoding="utf-8")
    symbol = (INTENT_MODEL.parent / "symbol.json").read_text(encoding="utf-8")
    runtime_script = (
        "window.__BERN__ = {"
        " INTENT_BUNDLE: " + intent + ","
        " GLYPH_BUNDLE: " + glyph + ","
        " SYMBOL_BUNDLE: " + symbol +
        " };\n" + runtime
    )

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
<html lang="en" data-theme="ink">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bern — try the model</title>
<style>
{css}
html, body {{ margin: 0; background: var(--paper); }}
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
