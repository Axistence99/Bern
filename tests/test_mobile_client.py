"""Keep the shipped React Native client honest about the server it talks to.

`react-native/bernApi.ts` is a contract with the API written in another language, so nothing
here can type-check it from Python. What can be checked is the part that silently rots: the
endpoint paths it calls and the multipart field names it uses. Both changed once already —
the client did not know about `engine`, chat memory or `/api/tutor` because the backend grew
after it was written, and nobody noticed because no test compared them.

These tests read the TypeScript as text. That is deliberate: it is the only way to catch a
renamed route from the Python side, and it costs nothing to run.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENT_PATH = REPO_ROOT / "react-native" / "bernApi.ts"
SERVER_PATH = REPO_ROOT / "server.py"


def _client_source() -> str:
    return CLIENT_PATH.read_text(encoding="utf-8")


def _server_routes() -> set[str]:
    """Every path pattern registered in the server's route table."""
    source = SERVER_PATH.read_text(encoding="utf-8")
    return set(re.findall(r'\(r"(/[^"]+)"', source))


class TestClientEndpointsExist:
    def test_client_file_is_present(self):
        assert CLIENT_PATH.exists(), "react-native/bernApi.ts is missing"

    def test_every_called_path_is_a_registered_route(self):
        source = _client_source()
        called = set(re.findall(r'"(/api/[a-z0-9\-]+)"', source))
        assert called, "no /api/ paths found in the client — did the extraction break?"

        routes = _server_routes()
        # Registered routes may carry regex escapes; compare on the literal path.
        normalised = {route.replace("\\", "") for route in routes}

        missing = sorted(path for path in called if path not in normalised)
        assert not missing, f"client calls routes the server does not serve: {missing}"

    def test_tutor_route_is_both_called_and_served(self):
        # Added after the client was first written; this is the exact class of drift the
        # rest of this file exists to catch.
        assert "/api/tutor" in _client_source()
        assert "/api/tutor" in {r.replace("\\", "") for r in _server_routes()}


class TestMultipartFieldNames:
    """The server reads specific field names, and a mismatch is a 400 at runtime."""

    def test_handwriting_posts_the_field_the_server_reads(self):
        client = _client_source()
        server = SERVER_PATH.read_text(encoding="utf-8")

        # The handwriting method passes its field name explicitly.
        handwriting = client[client.index("handwriting("):client.index("handwriting(") + 900]
        assert '"image"' in handwriting, "handwriting() must post the 'image' field"
        assert '"image"' in server, "server must read the 'image' field"
        assert 'self.request.files["image"]' in server

    def test_extract_posts_the_field_the_server_reads(self):
        client = _client_source()
        server = SERVER_PATH.read_text(encoding="utf-8")
        extract = client[client.index("extractFile("):client.index("extractFile(") + 400]
        assert "fileForm(uri, filename)" in extract, "extractFile() must use the default 'file' field"
        assert 'self.request.files["file"]' in server


class TestClientKnowsCurrentResponseShape:
    """Fields the server returns must be declared, or TypeScript callers silently lose them."""

    def test_chat_response_declares_memory_fields(self):
        source = _client_source()
        for field in ("resolved_as", "conversation", "entities", "engine"):
            assert field in source, f"ChatResponse does not declare '{field}'"

    def test_chat_passes_the_engine_option(self):
        source = _client_source()
        chat = source[source.index("chat(message"):source.index("chat(message") + 700]
        assert "engine" in chat, "chat() must forward the engine option"

    def test_health_exposes_language_model_flag(self):
        source = _client_source()
        health = source[source.index("health()"):source.index("health()") + 300]
        assert "language_model" in health

    def test_handwriting_response_declares_repairs(self):
        source = _client_source()
        assert "repairs" in source, "the server reports photo repairs; the client should type it"
