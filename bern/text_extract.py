"""Turn study material into plain text, whatever container it arrives in.

Bern is meant to be pointed at a student's notes and lecture files. Those come as photos
of handwriting, as PDFs exported from slides, as Word documents, or as plain text. This
module normalises all of them into one ``{text, source_type, pages, ...}`` dictionary that
:mod:`bern.study` can consume.

**No third-party dependencies.** ``pypdf``, ``pdfminer`` and ``python-docx`` are used when
they happen to be installed, but every format has a working stdlib fallback, so the
feature never disappears along with a virtualenv:

* ``.pdf``  — ``pypdf`` if present, otherwise a built-in extractor that inflates FlateDecode
  content streams with :mod:`zlib` and reads the text operators directly.
* ``.docx`` — :mod:`zipfile` + :mod:`xml.etree.ElementTree` over ``word/document.xml``.
* ``.txt`` / ``.md`` / ``.csv`` — decoded directly, with BOM and encoding detection.
* images — handed to :class:`bern.handwriting.HandwritingReader`.

Known limits, stated plainly because they affect what a user will see:

* Scanned PDFs with no text layer come back empty. There is no OCR fallback inside the PDF
  path — the page would have to be rasterised first, which needs a PDF renderer Bern does
  not ship. Extract the page as an image and use the handwriting path instead.
* Encrypted PDFs are refused rather than guessed at.
* The built-in PDF reader handles simple text operators (``Tj``, ``TJ``, ``'``, ``"``) and
  line positioning. It does not do CID/Type0 font mapping, so PDFs using custom glyph
  encodings produce garbled output; ``pypdf`` handles those correctly.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

#: Extensions routed to the handwriting recogniser.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

#: Extensions read as plain text.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".rst"}


class ExtractionError(RuntimeError):
    """Raised when a file cannot be turned into text."""


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------

def _pdf_unescape(literal: bytes) -> str:
    """Decode a PDF string literal, resolving the escapes the spec defines.

    PDF strings are raw bytes inside parentheses. ``\\(`` ``\\)`` ``\\\\`` are literal
    characters, ``\\n`` and friends are control characters, and ``\\ddd`` is an octal byte.
    """
    out = bytearray()
    index = 0
    while index < len(literal):
        byte = literal[index]
        if byte != 0x5C:  # backslash
            out.append(byte)
            index += 1
            continue
        index += 1
        if index >= len(literal):
            break
        escape = literal[index]
        mapping = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12,
                   0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}
        if escape in mapping:
            out.append(mapping[escape])
            index += 1
        elif 0x30 <= escape <= 0x37:  # octal, up to three digits
            digits = bytes([escape])
            index += 1
            for _ in range(2):
                if index < len(literal) and 0x30 <= literal[index] <= 0x37:
                    digits += bytes([literal[index]])
                    index += 1
                else:
                    break
            out.append(int(digits, 8) & 0xFF)
        elif escape in (0x0A, 0x0D):  # escaped newline = line continuation
            index += 1
        else:
            out.append(escape)
            index += 1

    raw = bytes(out)
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    return raw.decode("latin-1", errors="replace")


_STRING_RE = re.compile(rb"\((?:\\.|[^\\()])*\)", re.DOTALL)


def _pdf_text_from_content(content: bytes) -> str:
    """Pull readable text out of a PDF content stream.

    Content streams are postfix operator sequences. The ones that carry text are
    ``Tj`` (show string), ``TJ`` (show array of strings and kerning), ``'`` and ``"``
    (move to next line, then show). Line advances come from ``Td``/``TD``/``T*``/``Tm``;
    a negative vertical move is treated as a newline, which is what keeps paragraphs from
    collapsing into one wall of text.
    """
    pieces: list[str] = []
    for match in re.finditer(rb"(\[.*?\]|\((?:\\.|[^\\()])*\))\s*(Tj|TJ|'|\")|(-?[\d.]+)\s+(-?[\d.]+)\s+(Td|TD|Tm|T\*)",
                             content, re.DOTALL):
        array_or_string, operator, _x, y, move = match.groups()
        if operator:
            if operator in (b"'", b'"'):
                pieces.append("\n")
            if array_or_string.startswith(b"["):
                for literal in _STRING_RE.findall(array_or_string):
                    pieces.append(_pdf_unescape(literal[1:-1]))
            elif array_or_string.startswith(b"("):
                pieces.append(_pdf_unescape(array_or_string[1:-1]))
        elif move:
            try:
                if move in (b"Td", b"TD", b"T*") and float(y) < -0.5:
                    pieces.append("\n")
                elif move == b"Tm":
                    pieces.append("\n")
            except ValueError:
                pass
    return "".join(pieces)


def _extract_pdf_builtin(data: bytes) -> str:
    """Extract text from a PDF using only :mod:`zlib`.

    Walks every ``stream ... endstream`` block, inflates the FlateDecode ones, and reads
    their text operators. Streams that are images or fonts are skipped by sniffing the
    decoded bytes for text operators rather than by parsing the object dictionary.
    """
    if b"/Encrypt" in data:
        raise ExtractionError(
            "This PDF is encrypted. Decrypt it first, or export it as an image and use the "
            "handwriting endpoint."
        )

    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n?(.*?)endstream", data, re.DOTALL):
        blob = match.group(1)
        candidates = [blob]
        try:
            candidates.append(zlib.decompress(blob))
        except zlib.error:
            pass
        for candidate in candidates:
            if b"Tj" in candidate or b"TJ" in candidate:
                text = _pdf_text_from_content(candidate)
                if text.strip():
                    chunks.append(text)
    return "\n".join(chunks)


def _extract_xlsx(data: bytes) -> str:
    """Read a spreadsheet's cell values, one line per row.

    A ``.xlsx`` is a zip of XML. Text lives in ``xl/sharedStrings.xml`` and the cells in
    ``xl/worksheets/sheet*.xml`` hold indices into it, so both have to be read to recover what a
    person sees. Stdlib only, matching :func:`_extract_docx` - a spreadsheet is a common thing to
    hand an assistant, and failing on it was the plainest gap in what files Bern accepts.
    """
    import xml.etree.ElementTree as ET

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.iter():
                    if item.tag.endswith("}si"):
                        shared.append("".join(t.text or "" for t in item.iter()
                                              if t.tag.endswith("}t")))
            sheets = sorted(n for n in names
                            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            if not sheets:
                raise ExtractionError("This spreadsheet has no worksheets.")
            lines: list[str] = []
            for sheet in sheets:
                root = ET.fromstring(archive.read(sheet))
                for row in root.iter():
                    if not row.tag.endswith("}row"):
                        continue
                    cells: list[str] = []
                    for cell in row:
                        if not cell.tag.endswith("}c"):
                            continue
                        kind = cell.get("t")
                        value = ""
                        for child in cell:
                            if child.tag.endswith("}v") and child.text:
                                value = child.text
                            elif child.tag.endswith("}is"):
                                value = "".join(t.text or "" for t in child.iter()
                                                if t.tag.endswith("}t"))
                        if kind == "s" and value.isdigit() and int(value) < len(shared):
                            value = shared[int(value)]
                        if value:
                            cells.append(value)
                    if cells:
                        lines.append("\t".join(cells))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ExtractionError(f"Not a readable .xlsx file: {exc}") from exc

    if not lines:
        raise ExtractionError("This spreadsheet has no text in it.")
    return "\n".join(lines)


def _extract_pptx(data: bytes) -> str:
    """Read slide text, one block per slide.

    A ``.pptx`` is a zip of XML with one part per slide. Text sits in ``<a:t>`` runs inside the
    shapes; slide order comes from the file name, since ``ppt/slides/slide10.xml`` must sort
    after ``slide9.xml`` and would not as plain strings.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            slides = [n for n in archive.namelist()
                      if _re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
            if not slides:
                raise ExtractionError("This presentation has no slides.")
            slides.sort(key=lambda n: int(_re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
            blocks: list[str] = []
            for index, slide in enumerate(slides, start=1):
                root = ET.fromstring(archive.read(slide))
                runs = [node.text.strip() for node in root.iter()
                        if node.tag.endswith("}t") and node.text and node.text.strip()]
                if runs:
                    blocks.append(f"Slide {index}: " + " ".join(runs))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ExtractionError(f"Not a readable .pptx file: {exc}") from exc

    if not blocks:
        raise ExtractionError("This presentation has no text in it.")
    return "\n\n".join(blocks)


def _extract_pdf(data: bytes) -> tuple[str, str]:
    """Return ``(text, engine_name)``, preferring ``pypdf`` when it is importable."""
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("This PDF is encrypted.")
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages), "pypdf"
    except ImportError:
        return _extract_pdf_builtin(data), "builtin"
    except ExtractionError:
        raise
    except Exception as exc:  # malformed PDF that pypdf chokes on
        try:
            return _extract_pdf_builtin(data), "builtin"
        except Exception:
            raise ExtractionError(f"Could not read the PDF: {exc}") from exc


# --------------------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------------------

def _extract_docx(data: bytes) -> str:
    """Read ``word/document.xml`` and turn paragraphs into lines.

    A ``.docx`` is a zip. The visible text lives in ``<w:t>`` runs, grouped by ``<w:p>``
    paragraphs and ``<w:tbl>`` tables. Namespaces are matched by suffix so the code keeps
    working for documents produced by LibreOffice, Google Docs, or Word alike.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ExtractionError(f"Not a readable .docx file: {exc}") from exc

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ExtractionError(f"Corrupt document XML: {exc}") from exc

    lines: list[str] = []
    for paragraph in root.iter():
        if not paragraph.tag.endswith("}p"):
            continue
        text = "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))
        if text.strip():
            lines.append(text.strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------

def extract_text(data: bytes, filename: str = "") -> dict[str, Any]:
    """Convert a file's bytes into plain text.

    Args:
        data: Raw file contents.
        filename: Used only for its extension. Pass the real name when possible.

    Returns:
        ``{"text", "source_type", "engine", "characters", "lines"}``. ``source_type`` is
        one of ``text``, ``pdf``, ``docx`` or ``image``.

    Raises:
        ExtractionError: for unsupported types, encrypted PDFs, or unreadable files.
    """
    suffix = Path(filename).suffix.lower() if filename else ""

    if suffix in IMAGE_SUFFIXES:
        from bern.handwriting import HandwritingReader

        result = HandwritingReader().read_image(data)
        text = result.get("text", "")
        return _summary(text, "image", f"handwriting:{result.get('backend', '?')}")

    if suffix == ".pdf" or data[:5] == b"%PDF-":
        text, engine = _extract_pdf(data)
        return _summary(text, "pdf", engine)

    if suffix in (".docx", ".doc"):
        return _summary(_extract_docx(data), "docx", "zipfile")

    if suffix in (".xlsx", ".xlsm"):
        return _summary(_extract_xlsx(data), "xlsx", "zipfile")

    if suffix == ".pptx":
        return _summary(_extract_pptx(data), "pptx", "zipfile")

    if suffix in TEXT_SUFFIXES or not suffix:
        return _summary(_decode_text(data), "text", "decoded")

    raise ExtractionError(
        f"Unsupported file type {suffix!r}. Supported: PDF, DOCX, XLSX, PPTX, "
        f"{', '.join(sorted(IMAGE_SUFFIXES))}, {', '.join(sorted(TEXT_SUFFIXES))}."
    )


def extract_path(path: Path | str) -> dict[str, Any]:
    """Read a file from disk and extract its text."""
    path = Path(path)
    return extract_text(path.read_bytes(), path.name)


def _decode_text(data: bytes) -> str:
    """Decode bytes as UTF-8, falling back through common encodings.

    Filipino text sometimes arrives in Latin-1 from old systems, and windows-1252 creeps in
    from spreadsheet exports. Trying them in order beats raising on a stray byte.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _summary(text: str, source_type: str, engine: str) -> dict[str, Any]:
    """Normalise an extracted document into the shape the rest of Bern expects."""
    cleaned = _tidy(text)
    return {
        "text": cleaned,
        "source_type": source_type,
        "engine": engine,
        "characters": len(cleaned),
        "lines": len([line for line in cleaned.splitlines() if line.strip()]),
        "words": len(cleaned.split()),
    }


def _tidy(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure.

    Runs of spaces collapse, but blank lines are preserved because paragraph boundaries are
    what :mod:`bern.study` uses to segment topics.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
