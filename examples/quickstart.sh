#!/usr/bin/env bash
# Exercise every Bern endpoint with curl.
#
# Start the server first:      python3 server.py
# Then, in another terminal:   ./examples/quickstart.sh
#
# Point at a different host:   BERN=http://192.168.1.20:8000 ./examples/quickstart.sh
# Include file uploads:        NOTE_FILE=note.jpg PDF_FILE=lecture.pdf ./examples/quickstart.sh
set -uo pipefail

BERN="${BERN:-http://localhost:8000}"
say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

say "health"
curl -s "$BERN/api/health" | head -c 400; echo

say "chat — price question"
curl -s -X POST "$BERN/api/chat" -H 'Content-Type: application/json' \
  -d '{"message":"magkano ang bigas"}' | head -c 400; echo

say "chat — Taglish study request"
curl -s -X POST "$BERN/api/chat" -H 'Content-Type: application/json' \
  -d '{"message":"gumawa ng quiz mula sa notes ko"}' | head -c 400; echo

say "prices — search SKUs"
curl -s "$BERN/api/ph-prices/skus?q=bigas&limit=3" | head -c 400; echo

say "nutrition — search"
curl -s "$BERN/api/nutrition?q=adobo&limit=2" | head -c 400; echo

say "nutrition — meal total"
curl -s -X POST "$BERN/api/nutrition/meal" -H 'Content-Type: application/json' \
  -d '{"text":"2 pc chickenjoy and 1 cup rice"}' | head -c 400; echo

say "budget basket"
curl -s -X POST "$BERN/api/ph-prices/budget" -H 'Content-Type: application/json' \
  -d '{"budget_php":500}' | head -c 400; echo

say "extract — pasted text"
curl -s -X POST "$BERN/api/extract" -H 'Content-Type: application/json' \
  -d '{"text":"Ang nutrisyon ay ang proseso ng pagkuha ng sustansya."}' | head -c 300; echo

say "study pack from pasted text"
curl -s -X POST "$BERN/api/study" -H 'Content-Type: application/json' \
  -d '{"text":"Ang nutrisyon ay ang proseso ng pagkuha ng mga sustansya mula sa pagkain. Ang protina ay isang sustansya na tumutulong sa pagbuo ng kalamnan.","flashcards":3,"quiz_items":2}' \
  | head -c 700; echo

# File uploads need real paths, so they only run when you provide them.
if [[ -n "${NOTE_FILE:-}" ]]; then
  say "handwriting — $NOTE_FILE"
  curl -s -X POST "$BERN/api/handwriting" -F "image=@$NOTE_FILE" | head -c 500; echo
fi
if [[ -n "${PDF_FILE:-}" ]]; then
  say "extract — $PDF_FILE"
  curl -s -X POST "$BERN/api/extract" -F "file=@$PDF_FILE" | head -c 400; echo
  say "study pack — $PDF_FILE"
  curl -s -X POST "$BERN/api/study" -F "file=@$PDF_FILE" | head -c 600; echo
fi

cat <<'DONE'

Done. To include file uploads, re-run with:
  NOTE_FILE=note.jpg PDF_FILE=lecture.pdf ./examples/quickstart.sh
DONE
