# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Turn a person's exported library into a list of items.

One reader per format, each taking text that is already in memory, so the
command line (files on disk) and the page (files from the browser) share them:

    Obsidian vault      a folder of .md notes
    Are.na channel      the channel's JSON (exported, or fetched by slug);
                        any JSON list of {title, description, url} reads too
    bookmarks           the Netscape bookmarks .html every browser exports
                        (also Pocket, Pinboard, Raindrop)
    CSV                 any sheet with a title column (Goodreads, Zotero,
                        Readwise, a spreadsheet of your own)
    plain list          .txt / a single .md: one work per line

Standard library only. Nothing here touches the network except
`fetch_arena_channel`, which asks Are.na for a public channel.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple

# The dissonance axis: the three ways one work can stand against another.
DISSONANCE = ("contests", "complicates", "refutes")

MAX_TEXT_CHARS = 4000


@dataclass
class Item:
    """One thing in a person's library."""

    title: str
    text: str = ""
    url: str = ""
    source: str = ""
    # positions this person recorded: (axis, the title it stands against)
    stances: List[Tuple[str, str]] = field(default_factory=list)

    def document(self) -> str:
        return f"{self.title}\n{self.text}".strip()[:MAX_TEXT_CHARS]


class UnreadableSource(ValueError):
    """The file is not one of the formats above."""


# ─── Obsidian ───────────────────────────────────────────────────────────

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_INLINE_FIELD = re.compile(
    r"^\s*(?:[-*]\s*)?(contests|complicates|refutes)\s*::\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def _split_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """A note's YAML frontmatter, read without a YAML library: scalars,
    `[a, b]` lists and `- a` lists, which is all a stance needs."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: Dict[str, object] = {}
    key = None
    for line in text[3:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s+-\s*(.+)$", line)
        if item and key:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(_unquote(item.group(1)))
            continue
        pair = re.match(r"^([A-Za-z0-9_\- ]+):\s*(.*)$", line)
        if not pair:
            continue
        key = pair.group(1).strip().lower()
        value = pair.group(2).strip()
        if not value:
            meta[key] = []
        elif value.startswith("[") and value.endswith("]") and not value.startswith("[["):
            meta[key] = [_unquote(v) for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = _unquote(value)
    body = text[end + 4 :]
    return meta, body.lstrip("\n")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def _link_targets(value: str) -> List[str]:
    """The titles a stance names: `[[A]], [[B]]`, or bare text."""
    links = [m.strip() for m in _WIKILINK.findall(value)]
    if links:
        return links
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_obsidian(files: Iterable[Tuple[str, str]]) -> List[Item]:
    """`files` is (path, text) for each note. A note records a position with
    frontmatter (`contests: [[Other Note]]`) or an inline field
    (`contests:: [[Other Note]]`); the same for `complicates` and `refutes`."""
    items: List[Item] = []
    for path, text in files:
        if not path.lower().endswith((".md", ".markdown")):
            continue
        parts = path.replace("\\", "/").split("/")
        if any(p.startswith(".") for p in parts):  # .obsidian, .trash
            continue
        meta, body = _split_frontmatter(text)
        title = str(meta.get("title") or "") or os.path.splitext(parts[-1])[0]
        stances: List[Tuple[str, str]] = []
        for axis in DISSONANCE:
            value = meta.get(axis)
            values = value if isinstance(value, list) else [value] if value else []
            for v in values:
                stances += [(axis, t) for t in _link_targets(str(v))]
        for axis, value in _INLINE_FIELD.findall(body):
            stances += [(axis.lower(), t) for t in _link_targets(value)]
        plain = _WIKILINK.sub(lambda m: m.group(1), body)
        url = str(meta.get("url") or meta.get("source") or "")
        items.append(Item(title=title.strip(), text=plain.strip(), url=url, source="obsidian", stances=stances))
    return items


# ─── Are.na ─────────────────────────────────────────────────────────────


def parse_arena(text: str) -> List[Item]:
    """An Are.na channel as its API returns it: `{title, contents: [...]}`."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnreadableSource(f"not JSON: {exc}") from exc
    origin = "are.na"
    if isinstance(data, dict) and isinstance(data.get("contents"), list):
        blocks = data["contents"]
    elif isinstance(data, list):  # a plain list of {title, description, url}
        blocks, origin = data, "json"
    else:
        raise UnreadableSource("JSON without a `contents` list or a top-level list")
    items: List[Item] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        title = (
            block.get("title")
            or block.get("generated_title")
            or source.get("title")
            or block.get("name")
            or ""
        )
        body = " ".join(
            str(block.get(k) or "") for k in ("description", "content", "text", "notes", "author")
        ).strip()
        url = str(source.get("url") or block.get("url") or "")
        if not title and body:
            title = body[:80]
        if not title:
            continue
        items.append(Item(title=str(title).strip(), text=body, url=url, source=origin))
    return items


_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,200}$")


def fetch_arena_channel(slug_or_url: str, timeout: float = 20.0) -> str:
    """The JSON of a public Are.na channel, by its slug or its page address.
    The one network call in this repository; it sends the slug and nothing
    else."""
    slug = slug_or_url.strip().rstrip("/").split("/")[-1]
    if not _SLUG.match(slug):
        raise UnreadableSource("that does not look like an Are.na channel")
    url = f"https://api.are.na/v2/channels/{urllib.parse.quote(slug)}?per=100"
    request = urllib.request.Request(url, headers={"User-Agent": "cyph-tension-playground"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
        return response.read().decode("utf-8", errors="replace")


# ─── bookmarks ──────────────────────────────────────────────────────────


class _Bookmarks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: List[Item] = []
        self._href: Optional[str] = None
        self._text: List[str] = []
        self._tags = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            found = dict(attrs)
            self._href = found.get("href") or ""
            self._tags = (found.get("tags") or "").replace(",", " ")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            title = " ".join("".join(self._text).split())
            if title and self._href.startswith(("http://", "https://")):
                self.items.append(Item(title=title, text=self._tags, url=self._href, source="bookmarks"))
            self._href = None


def parse_bookmarks(text: str) -> List[Item]:
    parser = _Bookmarks()
    parser.feed(text)
    return parser.items


# ─── CSV ────────────────────────────────────────────────────────────────

_TITLE_COLUMNS = ("title", "book title", "name", "work")
_TEXT_COLUMNS = (
    "author", "authors", "author l-f", "my review", "review", "notes", "note", "highlight",
    "abstract", "description", "summary", "tags", "bookshelves", "publication title",
)
_URL_COLUMNS = ("url", "link", "source url")


def parse_csv(text: str) -> List[Item]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise UnreadableSource("an empty sheet")
    columns = {name.strip().lower(): name for name in reader.fieldnames if name}
    title_column = next((columns[c] for c in _TITLE_COLUMNS if c in columns), None)
    if title_column is None:
        raise UnreadableSource("no title column (looked for: " + ", ".join(_TITLE_COLUMNS) + ")")
    items: List[Item] = []
    for row in reader:
        title = (row.get(title_column) or "").strip()
        if not title:
            continue
        body = " ".join((row.get(columns[c]) or "").strip() for c in _TEXT_COLUMNS if c in columns)
        url = next(((row.get(columns[c]) or "").strip() for c in _URL_COLUMNS if c in columns), "")
        items.append(Item(title=title, text=" ".join(body.split()), url=url, source="csv"))
    return items


# ─── a plain list ───────────────────────────────────────────────────────

_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def parse_lines(text: str) -> List[Item]:
    items: List[Item] = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", raw).strip()
        if not line or line.startswith("#"):
            continue
        link = _MD_LINK.search(line)
        if link:
            rest = _MD_LINK.sub("", line).strip(" -–—:")
            items.append(Item(title=link.group(1).strip(), text=rest, url=link.group(2), source="list"))
        else:
            items.append(Item(title=line, source="list"))
    return items


# ─── choosing a reader ──────────────────────────────────────────────────


def parse_file(path: str, text: str) -> List[Item]:
    """One file, read by what it is."""
    lower = path.lower()
    if lower.endswith(".json"):
        return parse_arena(text)
    if lower.endswith((".html", ".htm")):
        return parse_bookmarks(text)
    if lower.endswith(".csv"):
        return parse_csv(text)
    if lower.endswith((".txt", ".md", ".markdown")):
        return parse_lines(text)
    raise UnreadableSource(f"{os.path.basename(path)}: not a format this reads")


def parse_files(files: List[Tuple[str, str]]) -> List[Item]:
    """Everything one person brought. Several notes together are a vault;
    anything else is read file by file."""
    notes = [(p, t) for p, t in files if p.lower().endswith((".md", ".markdown"))]
    others = [(p, t) for p, t in files if (p, t) not in notes]
    items: List[Item] = []
    if len(notes) > 1:
        items += parse_obsidian(notes)
    else:
        others = notes + others
    for path, text in others:
        if os.path.basename(path).startswith("."):
            continue
        items += parse_file(path, text)
    return items


def load_path(path: str) -> List[Item]:
    """A folder (a vault) or a single export, from disk."""
    if os.path.isdir(path):
        files: List[Tuple[str, str]] = []
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(names):
                if name.startswith("."):
                    continue
                full = os.path.join(root, name)
                if name.lower().endswith((".md", ".markdown", ".json", ".html", ".htm", ".csv", ".txt")):
                    with open(full, encoding="utf-8", errors="replace") as handle:
                        files.append((os.path.relpath(full, path), handle.read()))
        notes = [f for f in files if f[0].lower().endswith((".md", ".markdown"))]
        if len(notes) == 1:  # a folder holding one note is still a vault
            return parse_obsidian(notes) + parse_files([f for f in files if f not in notes])
        return parse_files(files)
    with open(path, encoding="utf-8", errors="replace") as handle:
        return parse_file(path, handle.read())
