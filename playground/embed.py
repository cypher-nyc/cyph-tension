# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Vectors for the playground.

The model takes cosines between vectors and does not care where the vectors
come from. Two sources here:

    lexical   words only (TF-IDF over the libraries in the room). Needs
              nothing installed. Good for seeing the mechanics; it knows
              nothing about meaning, so two people who read the same
              question in different vocabularies look far apart.
    qwen      Qwen3-Embedding-0.6B (open weights, Apache-2.0) through
              sentence-transformers, run on your own machine. Needs
              `pip install sentence-transformers` and a one-time download.

Each source puts its cosines in a band of its own. For an embedding model
`playground/band.py` measures it; for word statistics it is set by hand. A cosine is re-expressed on the model's band (`COS_LO`..`COS_HI`)
before it is handed over (`to_model_band`), so the model itself is never
retuned here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from tension import COS_HI, COS_LO

Vector = Dict[object, float]

_WORD = re.compile(r"[a-zà-ÿ][a-zà-ÿ'-]{2,}")
_STOP = frozenset(
    """
    the and for that with this from have not are was were but his her its their they you your our
    one all can will would should could has had been into than then them these those there here what
    which who whom whose when where why how about above after again against also any because before
    being below between both does doing down during each few further more most other some such only
    own same too very just over under out off once nor may might must shall upon while http https www
    com org html notes note
    """.split()
)


def tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def cosine(a: Vector, b: Vector) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def mean(vectors: Sequence[Vector]) -> Vector:
    """A person's composite: the mean of their items' vectors."""
    total: Dict[object, float] = {}
    for vector in vectors:
        for k, v in vector.items():
            total[k] = total.get(k, 0.0) + v
    n = len(vectors)
    return {k: v / n for k, v in total.items()} if n else {}


def _unit(vector: Vector) -> Vector:
    norm = math.sqrt(sum(v * v for v in vector.values()))
    return {k: v / norm for k, v in vector.items()} if norm else {}


class LexicalEmbedder:
    """TF-IDF over the documents it is given. No model, no download."""

    name = "lexical"
    # Set by hand, on the shipped examples, and rough. Word statistics cannot
    # be measured the way an embedding model is (playground/band.py): a
    # reader's own short notes share almost no words with each other, so
    # "the same reader, twice" reads as unrelated. Yours to move
    # (`--lexical-band`).
    band: Tuple[float, float] = (0.05, 0.45)

    def embed(self, documents: Sequence[str]) -> List[Vector]:
        counted = [Counter(tokens(d)) for d in documents]
        n = len(counted)
        df: Counter = Counter()
        for counts in counted:
            df.update(counts.keys())
        out: List[Vector] = []
        for counts in counted:
            out.append(
                _unit({w: (1.0 + math.log(c)) * (math.log((1 + n) / (1 + df[w])) + 1.0) for w, c in counts.items()})
            )
        return out


class QwenEmbedder:
    """Qwen3-Embedding-0.6B on this machine."""

    name = "qwen"
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    band = None  # measured on the reference shelves (playground/band.py)

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("the qwen embedder needs: pip install sentence-transformers") from exc
        # On the CPU on purpose. On Apple's GPU backend (seen with torch 2.2 on
        # macOS) a batch of several texts comes back with wrong vectors while
        # one text at a time is right; the CPU gives the same vector either
        # way, and the same vector on every machine.
        self._model = SentenceTransformer(self.model_id, device="cpu")

    def embed(self, documents: Sequence[str]) -> List[Vector]:
        rows = self._model.encode(list(documents), normalize_embeddings=True)
        return [{i: float(v) for i, v in enumerate(row)} for row in rows]


def make(name: str):
    if name == "lexical":
        return LexicalEmbedder()
    if name == "qwen":
        return QwenEmbedder()
    raise ValueError(f"unknown embedder: {name}")


def to_model_band(raw: float, band: Tuple[float, float]) -> float:
    """A cosine from a source whose cosines live in `band`, placed at the
    same relative position on the model's band."""
    lo, hi = band
    if (lo, hi) == (COS_LO, COS_HI) or hi <= lo:
        return raw
    return COS_LO + (raw - lo) / (hi - lo) * (COS_HI - COS_LO)
