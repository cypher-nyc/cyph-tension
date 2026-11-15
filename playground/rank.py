# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Gather the model's inputs from imported libraries, then ask the model.

For one person against each of the others:

    shared ground   the cosine between the means of their two libraries,
                    read through the band measured for the source of the
                    vectors (`playground/band.py`); with a topic, also the
                    lower of their two cosines to it
    capacity        how much each has read, saturating: what the model's
                    absorptive-capacity term reads (off unless κ is set)
    divergence      evidence only: lines on the dissonance axis (contests /
                    complicates / refutes) between a work one of them holds
                    and a work the other holds. The lines come from a
                    relations file and from positions a person recorded in
                    their own notes. Never from how far apart the vectors are.

Lines have a direction. A line whose source the other person holds and whose
target I hold contests me; the reverse is a line I bring. The mutual score
counts both; the two directed scores count one each.

The arithmetic is the model's (`tension.pair_intelligibility`, `absorb_with`,
`line_strength_with`, `pair_novelty`, `pair_tension`, `car_score`,
`group_floor`); nothing here scores anything itself.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import tension
from playground import band as banding
from playground import embed as embedding
from playground.importers import DISSONANCE, Item

LENS_CHALLENGE = "challenge"        # shared ground and a line on record
LENS_GROUND_ONLY = "ground only"    # shared ground, nothing on record
LENS_BELOW_FLOOR = "below floor"    # not enough shared ground to score
LENS_UNKNOWN = "unknown"            # nothing to read for this person

DEFAULT_STRENGTH = 1.0
# Assessors whose strengths for one line differ by this much or more
# disagree about it: the line is kept, at their median, and marked.
DISPUTE_SPREAD = 0.5
# The number of works at which a reader's capacity reaches 1 − 1/e.
CAPACITY_WORKS = 50.0


@dataclass
class Relation:
    """One line: `source` stands against `target` on `axis`, this strongly.
    `by` is who says so and `cite` where; several people can assess the same
    line, and all of them are kept."""

    source: str
    axis: str
    target: str
    strength: float = DEFAULT_STRENGTH
    origin: str = "relations"
    by: str = ""
    cite: str = ""
    assessments: List[float] = field(default_factory=list)

    @property
    def disputed(self) -> bool:
        return len(self.assessments) > 1 and max(self.assessments) - min(self.assessments) >= DISPUTE_SPREAD


def key(title: str) -> str:
    """A title reduced to what two libraries would agree on."""
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(w for w in text.split() if w not in {"the", "a", "an"})


def parse_relations(text: str) -> List[Relation]:
    """One line per assessment, tab-separated (or ` | `):

        source work <tab> axis <tab> target work [<tab> strength [<tab> by [<tab> cite]]]

    The same source, axis and target may appear more than once, assessed by
    different people. They become one line whose strength is the median of
    the assessments; a strength of 0 is an assessor saying the line is not
    there. Lines on other axes are kept out; `#` starts a comment."""
    grouped: Dict[Tuple[str, str, str], Relation] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in (line.split("\t") if "\t" in line else line.split("|"))]
        if len(parts) < 3:
            raise ValueError(f"relations line {number}: expected source, axis, target")
        axis = parts[1].lower()
        if axis not in DISSONANCE:
            continue
        try:
            strength = float(parts[3]) if len(parts) > 3 and parts[3] else DEFAULT_STRENGTH
        except ValueError as exc:
            raise ValueError(f"relations line {number}: strength is not a number") from exc
        if math.isnan(strength):
            raise ValueError(f"relations line {number}: strength is not a number")
        strength = max(0.0, strength)
        by = parts[4] if len(parts) > 4 else ""
        cite = parts[5] if len(parts) > 5 else ""
        identity = (key(parts[0]), axis, key(parts[2]))
        if identity not in grouped:
            grouped[identity] = Relation(parts[0], axis, parts[2], strength, by=by, cite=cite)
        relation = grouped[identity]
        relation.assessments.append(strength)
        if by and by not in relation.by.split(", "):
            relation.by = ", ".join(filter(None, [relation.by, by])) if relation.by != by else by
        relation.cite = relation.cite or cite
    for relation in grouped.values():
        relation.strength = statistics.median(relation.assessments)
    return list(grouped.values())


def capacity(items: Sequence[Item]) -> float:
    """What a reader can take in, from how much they have read: the stock of
    knowledge Nooteboom et al. use as their proxy, saturating."""
    return 1.0 - math.exp(-len(items) / CAPACITY_WORKS)


def _holds(keys: Sequence[str], work: str) -> Optional[int]:
    """Which of a person's items is this work, if any: the same title, or a
    title that contains it (a note called "a theory of justice, part two")."""
    wanted = key(work)
    if not wanted:
        return None
    for index, item_key in enumerate(keys):
        if item_key == wanted:
            return index
    if len(wanted) >= 8:
        for index, item_key in enumerate(keys):
            if f" {wanted} " in f" {item_key} ":
                return index
    return None


def _recorded(items: Sequence[Item]) -> List[Relation]:
    return [
        Relation(item.title, axis, target, DEFAULT_STRENGTH, origin="recorded", assessments=[DEFAULT_STRENGTH])
        for item in items
        for axis, target in item.stances
        if axis in DISSONANCE
    ]


def _lines(mine: Sequence[Item], theirs: Sequence[Item], relations: Sequence[Relation]) -> List[dict]:
    """Every line between a work of mine and a work of theirs. `toward` says
    whom it contests: "me" when they hold the source and I hold the target."""
    my_keys = [key(i.title) for i in mine]
    their_keys = [key(i.title) for i in theirs]
    found: Dict[Tuple[str, str, str, str], dict] = {}
    for relation in list(relations) + _recorded(mine) + _recorded(theirs):
        if relation.strength <= 0:
            continue
        for toward in ("them", "me"):
            source_keys, target_keys = (my_keys, their_keys) if toward == "them" else (their_keys, my_keys)
            i, j = _holds(source_keys, relation.source), _holds(target_keys, relation.target)
            if i is None or j is None or source_keys[i] == target_keys[j]:
                continue  # the same work on both sides is ground, not a line
            yours, theirs_title = (mine[i].title, theirs[j].title) if toward == "them" else (mine[j].title, theirs[i].title)
            identity = (source_keys[i], target_keys[j], relation.axis, toward)
            if identity not in found:
                found[identity] = {
                    "yours": yours,
                    "theirs": theirs_title,
                    "axis": relation.axis,
                    "toward": toward,
                    "strength": relation.strength,
                    "from": relation.origin,
                    "by": relation.by,
                    "disputed": relation.disputed,
                    "against": target_keys[j],
                }
    return sorted(found.values(), key=lambda l: (-l["strength"], l["yours"], l["theirs"], l["toward"]))


def _strength(lines: Sequence[dict], rho: float) -> float:
    """Lines against the same work are grouped, and each group weighed by the
    model (`line_strength_with`); the groups add."""
    groups: Dict[str, List[float]] = {}
    for line in lines:
        groups.setdefault(line["against"], []).append(line["strength"])
    return sum(tension.line_strength_with(rho, strengths) for _, strengths in sorted(groups.items()))


def _shared(mine: Sequence[Item], theirs: Sequence[Item]) -> List[str]:
    their_keys = {key(i.title) for i in theirs}
    seen, out = set(), []
    for item in mine:
        k = key(item.title)
        if k and k in their_keys and k not in seen:
            seen.add(k)
            out.append(item.title)
    return out


class Room:
    """Vectors and readings for one set of people, made once."""

    def __init__(self, people: Dict[str, List[Item]], relations: Sequence[Relation] = (),
                 topic: Optional[str] = None, embedder=None, kappa: Optional[float] = None,
                 rho: Optional[float] = None) -> None:
        self.people = people
        self.relations = relations
        self.embedder = embedder or embedding.LexicalEmbedder()
        self.kappa = tension.KAPPA_ABSORB if kappa is None else kappa
        self.rho = tension.LINE_DEPENDENCE if rho is None else rho
        self.band = banding.band_for(self.embedder)
        self.topic = (topic or "").strip() or None
        documents: List[str] = []
        spans: Dict[str, Tuple[int, int]] = {}
        for name, items in people.items():
            start = len(documents)
            documents += [item.document() for item in items]
            spans[name] = (start, len(documents))
        if self.topic:
            documents.append(self.topic)
        vectors = self.embedder.embed(documents) if documents else []
        self.means = {name: embedding.mean(vectors[a:b]) for name, (a, b) in spans.items()}
        self.topic_vector = vectors[-1] if self.topic else None

    def _on_band(self, raw: float) -> float:
        return embedding.to_model_band(raw, self.band)

    def _to_topic(self, name: str) -> Optional[float]:
        if self.topic_vector is None or not self.means[name]:
            return None
        return embedding.cosine(self.means[name], self.topic_vector)

    def pair(self, me: str, them: str) -> dict:
        """Everything the model says about two people, from `me`'s side."""
        mine, theirs = self.people[me], self.people[them]
        row = {"name": them, "items": len(theirs), "score": None, "lens": LENS_UNKNOWN,
               "intelligibility": None, "novelty": None, "contested_by": None, "contests": None,
               "cos_people": None, "cos_topic": None, "contest_strength": 0.0,
               "capacity": round(capacity(theirs), 4), "lines": [], "shared_works": []}
        if not self.means[me] or not self.means[them]:
            return row
        raw_people = embedding.cosine(self.means[me], self.means[them])
        mine_topic, their_topic = self._to_topic(me), self._to_topic(them)
        raw_topic = min(mine_topic, their_topic) if mine_topic is not None and their_topic is not None else None
        ground = tension.pair_intelligibility(
            self._on_band(raw_people), self._on_band(raw_topic) if raw_topic is not None else None
        )
        # each reads the other through their own capacity; a table needs both
        ground_me = tension.absorb_with(self.kappa, ground, capacity(mine))
        ground_them = tension.absorb_with(self.kappa, ground, capacity(theirs))
        mutual_ground = min(ground_me, ground_them)
        lines = _lines(mine, theirs, self.relations)
        toward_me = [l for l in lines if l["toward"] == "me"]
        toward_them = [l for l in lines if l["toward"] == "them"]
        strength = _strength(toward_me, self.rho) + _strength(toward_them, self.rho)
        novelty = tension.pair_novelty(strength)
        score = tension.pair_tension(mutual_ground, novelty)
        if mutual_ground < tension.I_FLOOR:
            lens = LENS_BELOW_FLOOR
        elif score > 0:
            lens = LENS_CHALLENGE
        else:
            lens = LENS_GROUND_ONLY
        row.update(
            score=round(score, 4), lens=lens, intelligibility=round(mutual_ground, 4), novelty=round(novelty, 4),
            # what I stand to meet from them, and what they stand to meet from me
            contested_by=round(tension.pair_tension(ground_me, tension.pair_novelty(_strength(toward_me, self.rho))), 4),
            contests=round(tension.pair_tension(ground_them, tension.pair_novelty(_strength(toward_them, self.rho))), 4),
            cos_people=round(raw_people, 4),
            cos_topic=round(raw_topic, 4) if raw_topic is not None else None,
            contest_strength=round(strength, 4),
            lines=[{k: v for k, v in line.items() if k != "against"} for line in lines],
            shared_works=_shared(mine, theirs),
        )
        return row

    def about(self) -> dict:
        return {
            "topic": self.topic,
            "backend": tension.BACKEND,
            "embedder": self.embedder.name,
            "band": [round(self.band[0], 4), round(self.band[1], 4)],
            "floor": tension.I_FLOOR,
            "kappa": self.kappa,
            "rho": self.rho,
        }


def rank(people: Dict[str, List[Item]], me: str, relations: Sequence[Relation] = (),
         topic: Optional[str] = None, embedder=None, kappa: Optional[float] = None,
         rho: Optional[float] = None) -> dict:
    """Everyone else, ordered by productive tension with `me`."""
    if me not in people:
        raise KeyError(me)
    room = Room(people, relations, topic, embedder, kappa, rho)
    rows = [room.pair(me, name) for name in people if name != me]
    # by score; people nobody could read last. Among equal scores the one
    # with more shared ground comes first, then by name: an ordering for
    # reading the list, not part of the model.
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0), -(r["intelligibility"] or 0.0), r["name"]))
    return {"me": me, "capacity": round(capacity(people[me]), 4), **room.about(), "ranked": rows}


def group(people: Dict[str, List[Item]], members: Sequence[str], relations: Sequence[Relation] = (),
          topic: Optional[str] = None, embedder=None, kappa: Optional[float] = None,
          rho: Optional[float] = None) -> dict:
    """A table of several people: the mean of its pairs (what the model has
    always scored), its floor (the least well-met person's best pair), who
    each person's counterpart is, and how many different works are contested
    across it (three pairs contesting one book are one conversation)."""
    members = list(dict.fromkeys(members))
    for name in members:
        if name not in people:
            raise KeyError(name)
    if len(members) < 2:
        raise ValueError("a group needs at least two people")
    room = Room({name: people[name] for name in members}, relations, topic, embedder, kappa, rho)
    n = len(members)
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    pairs, contested = [], set()
    for i in range(n):
        for j in range(i + 1, n):
            row = room.pair(members[i], members[j])
            matrix[i][j] = matrix[j][i] = row["score"]
            pairs.append({"a": members[i], "b": members[j], "score": row["score"], "lens": row["lens"]})
            for line in row["lines"]:
                contested.add(key(line["yours"]) if line["toward"] == "me" else key(line["theirs"]))
    counterparts = []
    for i, name in enumerate(members):
        known = [(matrix[i][j], members[j]) for j in range(n) if j != i and matrix[i][j] is not None]
        best = max(known, key=lambda x: (x[0], x[1])) if known else None
        counterparts.append({"name": name, "best": best[0] if best else None,
                             "with": best[1] if best and best[0] > 0 else None})
    return {
        "members": members,
        **room.about(),
        "mean": tension.car_score([p["score"] for p in pairs]),
        "floor": tension.group_floor(matrix),
        "contested_works": len(contested),
        "counterparts": counterparts,
        "pairs": pairs,
    }
