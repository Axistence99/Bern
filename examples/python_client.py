"""Minimal Bern client using only the standard library.

Useful for scripts, notebooks, and as copy-paste material when integrating Bern into
another Python service. No `requests` dependency, so it runs anywhere Python does.

    python3 examples/python_client.py http://localhost:8000

Start the server first (`python3 server.py`) — every call here is a real HTTP request.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class BernClient:
    """A thin, typed-by-convention wrapper over Bern's REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing --------------------------------------------------------------------

    def _request(self, path: str, *, method: str = "GET",
                 json_body: Any = None, data: bytes | None = None,
                 headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Send one request and decode the JSON response.

        Raises:
            BernError: when the server answers with a non-2xx status, carrying the
                server's own ``error`` message rather than a bare status code.
        """
        body = data
        request_headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
                message = payload.get("error", raw)
            except json.JSONDecodeError:
                message = raw
            raise BernError(message, exc.code) from exc

    def _upload(self, path: str, field: str, file_path: str | Path,
                extra: dict[str, str] | None = None) -> dict[str, Any]:
        """POST one file as multipart/form-data, built by hand for stdlib-only use."""
        file_path = Path(file_path)
        boundary = f"----bern{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        parts: list[bytes] = []
        for key, value in (extra or {}).items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
            f"filename=\"{file_path.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
        )
        parts.append(file_path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())

        return self._request(path, method="POST", data=b"".join(parts),
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})

    # -- endpoints -------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("/api/health")

    def chat(self, message: str, session_id: str = "python") -> dict[str, Any]:
        return self._request("/api/chat", method="POST",
                             json_body={"message": message, "session_id": session_id})

    def predict(self, message: str) -> dict[str, Any]:
        return self._request("/api/predict", method="POST", json_body={"message": message})

    def handwriting(self, image_path: str | Path, backend: str = "auto",
                    case_mode: str = "auto") -> dict[str, Any]:
        return self._upload("/api/handwriting", "image", image_path,
                            {"backend": backend, "case_mode": case_mode})

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        return self._upload("/api/extract", "file", file_path)

    def study(self, file_path: str | Path, flashcards: int = 20, quiz_items: int = 10,
              true_false_items: int = 8) -> dict[str, Any]:
        return self._upload("/api/study", "file", file_path, {
            "flashcards": str(flashcards),
            "quiz_items": str(quiz_items),
            "true_false_items": str(true_false_items),
        })

    def study_text(self, text: str, flashcards: int = 20, quiz_items: int = 10) -> dict[str, Any]:
        return self._request("/api/study", method="POST",
                             json_body={"text": text, "flashcards": flashcards, "quiz_items": quiz_items})

    def prices(self, q: str = "", category: str = "") -> dict[str, Any]:
        return self._request(f"/api/ph-prices?q={q}&category={category}")

    def skus(self, q: str = "", limit: int = 10) -> dict[str, Any]:
        return self._request(f"/api/ph-prices/skus?q={q}&limit={limit}")

    def budget(self, budget_php: float) -> dict[str, Any]:
        return self._request("/api/ph-prices/budget", method="POST",
                             json_body={"budget_php": budget_php})

    def nutrition(self, q: str, limit: int = 5) -> dict[str, Any]:
        return self._request(f"/api/nutrition?q={q}&limit={limit}")

    def meal(self, text: str) -> dict[str, Any]:
        return self._request("/api/nutrition/meal", method="POST", json_body={"text": text})

    def count_money(self, image_path: str | Path) -> dict[str, Any]:
        return self._upload("/api/count-money", "image", image_path)


class BernError(RuntimeError):
    """Raised for a non-2xx response, carrying the server's message and status."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    client = BernClient(base)

    print("health:", client.health())

    reply = client.chat("magkano ang bigas")
    print(f"\nchat  -> intent={reply['intent']} confidence={reply['confidence']:.2f}")
    print("        ", reply["reply"].splitlines()[0][:100])

    reply = client.chat("gumawa ng quiz mula sa notes ko")
    print(f"\nchat  -> intent={reply['intent']}")

    skus = client.skus("bigas", limit=3)
    print(f"\nskus  -> {skus['count']} matches of {skus['total_skus']} total")

    meal = client.meal("2 pc chickenjoy and 1 cup rice")
    print(f"\nmeal  -> keys: {list(meal)[:6]}")

    pack = client.study_text(
        "Ang nutrisyon ay ang proseso ng pagkuha ng mga sustansya mula sa pagkain. "
        "Ang protina ay isang sustansya na tumutulong sa pagbuo ng kalamnan.",
        flashcards=3, quiz_items=2,
    )
    print(f"\nstudy -> {len(pack['flashcards'])} cards, {len(pack['quiz'])} quiz items, "
          f"{len(pack['reviewer']['sections'])} reviewer sections")


if __name__ == "__main__":
    main()
