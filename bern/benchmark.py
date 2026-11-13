"""Held-out benchmark for Bern: does a realistic message reach the right skill and get a sensible answer?

The benchmark (data/eval/bern_benchmark.json) is written independently of the training data and split into
  * dev  - used for error analysis while training,
  * test - only scored for the final report (``--show-failures dev`` never prints test items).

Metrics
  routing_accuracy      chat() intent == expected   (expected "unknown" = Bern must admit it does not know)
  classifier_accuracy   predict() intent == expected (in-scope items only; the raw ML model, no rules)
  ood_rejection         share of out-of-scope messages Bern correctly flags as unknown
  false_reject_rate     share of in-scope messages wrongly flagged unknown
  answer_pass           reply contains the required text (e.g. "4410", "kcal", "₱") - checked on routed-correctly items

Usage:  python3 -m bern.benchmark [--show-failures dev]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BASE_DIR / "data" / "eval" / "bern_benchmark.json"


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text("utf-8"))["items"]


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    in_scope = [r for r in rows if r["expected"] != "unknown"]
    ood = [r for r in rows if r["expected"] == "unknown"]
    clf = [r for r in in_scope if r["clf_intent"] is not None]
    checked = [r for r in rows if r["contains_ok"] is not None and r["route_ok"]]
    per_intent: dict[str, dict[str, Any]] = {}
    for tag in sorted({r["expected"] for r in rows}):
        sub = [r for r in rows if r["expected"] == tag]
        per_intent[tag] = {"n": len(sub), "routing_accuracy": round(sum(r["route_ok"] for r in sub) / len(sub), 4)}
    confusion = Counter((r["expected"], r["got"]) for r in rows if not r["route_ok"])
    return {
        "n": n,
        "routing_accuracy": round(sum(r["route_ok"] for r in rows) / n, 4),
        "in_scope_routing_accuracy": round(sum(r["route_ok"] for r in in_scope) / max(1, len(in_scope)), 4),
        "classifier_accuracy": round(sum(r["clf_intent"] == r["expected"] for r in clf) / max(1, len(clf)), 4),
        "ood_rejection": round(sum(r["route_ok"] for r in ood) / max(1, len(ood)), 4) if ood else None,
        "false_reject_rate": round(sum(r["got"] == "unknown" for r in in_scope) / max(1, len(in_scope)), 4),
        "answer_pass": round(sum(r["contains_ok"] for r in checked) / max(1, len(checked)), 4) if checked else None,
        "per_intent": per_intent,
        "top_confusions": [{"expected": e, "got": g, "count": c} for (e, g), c in confusion.most_common(8)],
    }


def evaluate(bern: Any, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run every benchmark item through ``bern.chat`` and ``bern.predict``."""
    items = items if items is not None else load_benchmark()
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        res = bern.chat(it["text"], session_id=f"bench-{i}")
        clf = bern.predict(it["text"])["intent"] if it["intent"] != "unknown" else None
        reply = res["reply"].lower()
        contains_ok = all(s.lower() in reply for s in it.get("contains", [])) if it.get("contains") else None
        rows.append({
            "text": it["text"], "expected": it["intent"], "got": res["intent"], "split": it.get("split", "dev"),
            "route_ok": res["intent"] == it["intent"], "clf_intent": clf, "contains_ok": contains_ok,
            "confidence": res.get("confidence"), "reply": res["reply"][:140].replace("\n", " "),
        })
    report: dict[str, Any] = {"all": _score(rows), "rows": rows}
    for split in sorted({r["split"] for r in rows}):
        report[split] = _score([r for r in rows if r["split"] == split])
    return report


def format_summary(report: dict[str, Any]) -> str:
    lines = []
    for split in [k for k in ("dev", "test", "test2") if k in report] + ["all"]:
        s = report[split]
        pct = lambda v: "n/a" if v is None else f"{v * 100:5.1f}%"  # noqa: E731
        lines.append(
            f"{split:5} n={s['n']:3d} | routing {pct(s['routing_accuracy'])} | classifier {pct(s['classifier_accuracy'])} | "
            f"OOD rejected {pct(s['ood_rejection'])} | false-reject {pct(s['false_reject_rate'])} | answers ok {pct(s['answer_pass'])}"
        )
    return "\n".join(lines)


def main() -> None:
    from bern.engine import BernAI

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-failures", choices=["dev", "all", "none"], default="dev",
                    help="print failing items (default: dev only - test items stay unseen)")
    ap.add_argument("--json", type=Path, help="write the full report to this file")
    args = ap.parse_args()
    report = evaluate(BernAI())
    print(format_summary(report))
    if args.show_failures != "none":
        print("\nFailures:")
        for r in report["rows"]:
            if (not r["route_ok"] or r["contains_ok"] is False) and (args.show_failures == "all" or r["split"] == "dev"):
                why = "" if r["route_ok"] else f"expected {r['expected']} got {r['got']} (conf {r['confidence']})"
                if r["contains_ok"] is False:
                    why += " | reply missing required text"
                print(f"  [{r['split']}] {r['text']!r}: {why}\n        -> {r['reply']}")
    if args.json:
        args.json.write_text(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1), "utf-8")


if __name__ == "__main__":
    main()
