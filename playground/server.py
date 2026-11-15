# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The playground as a page on your own machine.

Standard library only. It listens on 127.0.0.1, keeps everything in memory
for as long as it runs, and writes nothing to disk. The libraries you load
never leave the machine; the one outbound call is to Are.na, and only when
you ask for a channel by name.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

import tension
from playground import embed as embedding
from playground import importers, rank

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
MAX_BODY = 64 * 1024 * 1024


class Room:
    """Who is here and which lines between works are on record."""

    def __init__(self, embedder) -> None:
        self.lock = threading.Lock()
        self.embedder = embedder
        self.people: Dict[str, List[importers.Item]] = {}
        self.relations_text = ""
        self.relations: List[rank.Relation] = []

    def add(self, name: str, items: List[importers.Item]) -> None:
        self.people.setdefault(name, []).extend(items)

    def set_relations(self, text: str) -> None:
        self.relations = rank.parse_relations(text)
        self.relations_text = text

    def load_examples(self) -> None:
        folder = os.path.join(EXAMPLES, "people")
        for entry in sorted(os.listdir(folder)):
            if entry.startswith("."):
                continue
            self.people[entry.split(".")[0]] = importers.load_path(os.path.join(folder, entry))
        with open(os.path.join(EXAMPLES, "relations.tsv"), encoding="utf-8") as handle:
            self.set_relations(handle.read())

    def state(self) -> dict:
        return {
            "people": [
                {
                    "name": name,
                    "items": len(items),
                    "sources": dict(Counter(i.source for i in items)),
                    "recorded": sum(len(i.stances) for i in items),
                }
                for name, items in self.people.items()
            ],
            "relations": len(self.relations),
            "relations_text": self.relations_text,
            "backend": tension.BACKEND,
            "embedder": self.embedder.name,
            "floor": tension.I_FLOOR,
        }


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def make_handler(room: Room):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tension-playground"

        def log_message(self, *args) -> None:  # quiet
            return

        # ─── plumbing ───────────────────────────────────────────────

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _local(self) -> None:
            """Only this machine's own page may talk to it."""
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                raise Refused(403, "this page only answers on 127.0.0.1")

        def _body(self) -> dict:
            # a header a form on another site cannot set
            if self.headers.get("X-Playground") != "1":
                raise Refused(403, "not from the playground page")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise Refused(413, "that is more than this page will read at once")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                raise Refused(400, "not JSON") from exc
            if not isinstance(data, dict):
                raise Refused(400, "not an object")
            return data

        def _guard(self, work) -> None:
            try:
                self._local()
                with room.lock:
                    work()
            except Refused as exc:
                self._json({"error": str(exc)}, exc.status)
            except (importers.UnreadableSource, ValueError) as exc:
                self._json({"error": str(exc)}, 422)
            except KeyError as exc:
                self._json({"error": f"nobody here called {exc.args[0]}"}, 404)
            except (urllib.error.URLError, TimeoutError) as exc:
                self._json({"error": f"could not reach Are.na: {exc}"}, 502)

        # ─── routes ─────────────────────────────────────────────────

        def do_GET(self) -> None:  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            if url.path == "/":
                with open(os.path.join(HERE, "page.html"), "rb") as handle:
                    return self._send(200, handle.read(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._guard(lambda: self._json(room.state()))
            if url.path == "/api/rank":
                def work() -> None:
                    me = (query.get("me") or [""])[0]
                    topic = (query.get("topic") or [""])[0]
                    self._json(rank.rank(room.people, me, room.relations, topic, room.embedder))
                return self._guard(work)
            if url.path == "/api/group":
                def work() -> None:
                    members = [m for m in (query.get("members") or [""])[0].split(",") if m]
                    topic = (query.get("topic") or [""])[0]
                    self._json(rank.group(room.people, members, room.relations, topic, room.embedder))
                return self._guard(work)
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urllib.parse.urlparse(self.path).path

            def name_of(data: dict) -> str:
                name = " ".join(str(data.get("name") or "").split())[:60]
                if not name:
                    raise Refused(422, "a person needs a name")
                return name

            def person() -> None:
                data = self._body()
                files = [
                    (str(f.get("path") or ""), str(f.get("text") or ""))
                    for f in data.get("files") or []
                    if isinstance(f, dict)
                ]
                if not files:
                    raise Refused(422, "no files")
                items = importers.parse_files(files)
                if not items:
                    raise Refused(422, "nothing readable in those files")
                room.add(name_of(data), items)
                self._json(room.state())

            def arena() -> None:
                data = self._body()
                name = name_of(data)
                items = importers.parse_arena(importers.fetch_arena_channel(str(data.get("channel") or "")))
                if not items:
                    raise Refused(422, "that channel has nothing readable in it")
                room.add(name, items)
                self._json(room.state())

            def remove() -> None:
                room.people.pop(str(self._body().get("name") or ""), None)
                self._json(room.state())

            def relations() -> None:
                room.set_relations(str(self._body().get("text") or ""))
                self._json(room.state())

            def examples() -> None:
                self._body()
                room.load_examples()
                self._json(room.state())

            def reset() -> None:
                self._body()
                room.people.clear()
                room.set_relations("")
                self._json(room.state())

            routes = {
                "/api/person": person,
                "/api/arena": arena,
                "/api/remove": remove,
                "/api/relations": relations,
                "/api/examples": examples,
                "/api/reset": reset,
            }
            if path not in routes:
                return self._json({"error": "not found"}, 404)
            self._guard(routes[path])

    return Handler


def serve(port: int = 0, embedder_name: str = "lexical", lexical_band=None) -> None:
    embedder = embedding.make(embedder_name)
    if lexical_band and embedder_name == "lexical":
        embedder.band = tuple(lexical_band)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Room(embedder)))
    print(f"tension playground: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    print(f"model served by: {tension.BACKEND}   vectors: {embedder.name}   (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
