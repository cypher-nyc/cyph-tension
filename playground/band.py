# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Where a source of vectors puts its cosines, measured.

The model reads a cosine through a band: `COS_LO` maps to no shared ground
and `COS_HI` to complete shared ground. That band is not a property of the
model. It belongs to the vectors (each embedding model puts its cosines
somewhere else) and to the thing compared (the mean of a whole library sits
closer to every other mean than two single documents do). So it is measured
for each source, on a reference that needs no labels:

    low end    two libraries about unrelated subjects. The reference shelves
               (examples/reference/shelves.json: bread, astronomy, contract
               law, ...) are unrelated by construction; the low end is the
               90th percentile of the cosines between their means, so nine
               unrelated pairs in ten read as no shared ground at all.
    high end   the same reader, twice. Each shelf is split in half every
               possible way; the high end is the median cosine between the
               means of the two halves.

A cosine from the source is then placed at the same relative position on the
model's own band before it is handed over (`embed.to_model_band`), so the
model itself is never retuned.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Sequence, Tuple

from playground import embed as embedding

SHELVES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "reference", "shelves.json")
LOW_PERCENTILE = 0.90
# A band narrower than this is not a measurement.
MIN_WIDTH = 0.1


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("nothing to measure")
    position = p * (len(ordered) - 1)
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def measure(embedder, shelves: Dict[str, List[str]] = None) -> dict:
    """The band this embedder's library means live in, with what it was
    measured on. Deterministic: no sampling."""
    if shelves is None:
        with open(SHELVES, encoding="utf-8") as handle:
            shelves = json.load(handle)
    names = list(shelves)
    documents: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for name in names:
        start = len(documents)
        documents += shelves[name]
        spans[name] = (start, len(documents))
    vectors = embedder.embed(documents)
    means = {name: embedding.mean(vectors[a:b]) for name, (a, b) in spans.items()}
    unrelated = [embedding.cosine(means[a], means[b]) for a, b in itertools.combinations(names, 2)]
    same: List[float] = []
    for name in names:
        own = vectors[spans[name][0] : spans[name][1]]
        indices = range(len(own))
        for half in itertools.combinations(indices, len(own) // 2):
            if 0 not in half:  # each split once
                continue
            other = [i for i in indices if i not in half]
            same.append(embedding.cosine(embedding.mean([own[i] for i in half]), embedding.mean([own[i] for i in other])))
    low, high = _percentile(unrelated, LOW_PERCENTILE), _percentile(same, 0.5)
    if high - low < MIN_WIDTH:
        raise ValueError("this source does not tell unrelated libraries from the same reader")
    return {
        "low": low,
        "high": high,
        "unrelated_pairs": len(unrelated),
        "unrelated_median": _percentile(unrelated, 0.5),
        "same_reader_splits": len(same),
        "shelves": len(names),
    }


_measured: Dict[str, dict] = {}


def band_for(embedder) -> Tuple[float, float]:
    """The embedder's band: the one it was given, else the measured one
    (measured once per source in a process)."""
    given = getattr(embedder, "band", None)
    if given:
        return tuple(given)
    if embedder.name not in _measured:
        _measured[embedder.name] = measure(embedder)
    found = _measured[embedder.name]
    return found["low"], found["high"]
