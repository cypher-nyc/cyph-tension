# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The productive-tension model, in Python.

    tension = gate(I) · I · N · (1 + λ·T)

The same model as `haskell/src/Cyph/Tension.hs`, operation for operation, so
the two agree to the last bit (`tests/test_parity.py`). The `_py_` functions
here are the reference; the public names at the bottom are served by the
compiled Haskell library when it has been built (`make -C haskell lib`) and
by the reference otherwise. `BACKEND` says which.

The reasoning and the research behind each term are in `docs/MODEL.md`.
Every function is pure: scalars in, a scalar out, no I/O.
"""

from __future__ import annotations

import functools
import math
from typing import Iterable, Optional

from tension import _haskell

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Mutual-intelligibility floor. Below this there is not enough shared
# language for divergence to be generative, so the score is ZERO rather
# than small — an admit ranking must never recommend someone on the
# strength of pure opposition (Bail et al. 2018).
I_FLOOR = 0.15

# Cosines between real document embeddings cluster in a band, not across
# [0, 1]. This band was set on bge-large-en-v1.5, where they sit in roughly
# [0.25, 0.85]; mapping that band to [0, 1] keeps the intelligibility
# term from saturating at ~0.5 for everyone. Values outside clamp.
COS_LO = 0.25
COS_HI = 0.85

# Intelligibility blends a semantic reading (cosine of the person's vector to the
# topic) with a behavioural one (weighted engagement depth over the
# topic's resource neighbourhood). Equal halves: the vector alone is
# fooled by adjacent-but-untouched territory, the graph alone misses
# someone whose relevant reading lives outside this neighbourhood.
W_VEC = 0.5
W_GRAPH = 0.5

# Novelty blends two kinds of divergence evidence. Reading divergence is
# common and weakly informative; a recorded stance (contests / complicates / refutes) is
# rarer and much stronger — it is a position the person actually took.
READING_W = 0.6
STANCE_W = 0.4

# Learning-progress bonus. Deliberately modest: trajectory MODULATES a
# score, it never creates one. A person moving toward the topic with no
# shared ground is still gated out by I_FLOOR.
LAMBDA_TRAJECTORY = 0.35

# Cosine-delta between a person's recent and older territory that
# saturates the trajectory bonus. Movement of ~0.15 cosine toward a
# topic across a season is already a pronounced shift in attention.
T_MAX = 0.15


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _py_normalize_cosine(cosine: float) -> float:
    """Map a raw embedding cosine onto [0, 1] across the band where real
    document similarities actually live."""
    if COS_HI <= COS_LO:
        return _clamp01(cosine)
    return _clamp01((float(cosine) - COS_LO) / (COS_HI - COS_LO))


def _py_compute_intelligibility(cos_to_anchor: float, depth_norm: float) -> float:
    """Shared ground between a person and a topic — Nooteboom's
    absorptive-capacity term, the factor that DECREASES with epistemic
    distance.

    Args:
        cos_to_anchor: cosine between the person's vector (the
            recency-weighted centroid of what they have explored) and the
            topic's stored vector. Raw cosine; normalized internally.
        depth_norm: their similarity-weighted engagement over the topic's
            resource neighbourhood, normalized to [0, 1] against the rest
            of the candidate pool.

    Returns [0, 1]. High = they can already speak this language.
    """
    return _clamp01(
        W_VEC * _py_normalize_cosine(cos_to_anchor) + W_GRAPH * _clamp01(depth_norm)
    )


def _py_compute_novelty(divergent_reading_ratio: float, stance_ratio: float) -> float:
    """Divergence WITHIN topic-relevant zones — Nooteboom's novelty term,
    the factor that INCREASES with epistemic distance.

    Both inputs are behavioural by design (Nemeth 2001): this measures a
    different route actually travelled through shared territory, not a
    distance inferred from embeddings. Manufacturing opposition from
    vector distance produces the devil's-advocate failure mode, where
    apparent disagreement bolsters rather than challenges.

    Args:
        divergent_reading_ratio: fraction of the topic-local resources the
            person engaged that sit OUTSIDE the topic's canonical core —
            they read around the same question from elsewhere.
        stance_ratio: the positions they recorded over this
            neighbourhood (COMPLICATES / CONTESTS / REFUTES), normalized
            against the candidate pool. Recorded, specific positions.

    Returns [0, 1]. Zero when there is no divergence evidence at all.
    """
    return _clamp01(
        READING_W * _clamp01(divergent_reading_ratio)
        + STANCE_W * _clamp01(stance_ratio)
    )


def _py_compute_trajectory(cos_recent: float, cos_older: float) -> float:
    """Learning progress toward the topic (Kidd et al. 2012; Ten et al.
    2021): how much closer a person's recent territory sits to the topic
    than their older territory did.

    Only motion TOWARD the topic is rewarded — drift away is not
    penalized, because a person moving on from a subject still knows it.
    Returns [0, 1], saturating at T_MAX of cosine movement.
    """
    delta = float(cos_recent) - float(cos_older)
    if delta <= 0 or T_MAX <= 0:
        return 0.0
    return _clamp01(delta / T_MAX)


def _py_compute_tension(
    intelligibility: float,
    novelty: float,
    trajectory: float = 0.0,
) -> float:
    """Productive tension between a person and a topic.

    The inverted-U is structural, not tuned: `intelligibility` falls with
    epistemic distance while `novelty` rises with it, so the product
    peaks at moderate distance (Nooteboom et al. 2007). Below I_FLOOR the
    result is hard zero — divergence without shared language entrenches
    rather than generates (Bail et al. 2018).

    Asymmetric by design: the arguments describe a person's standing
    relative to a topic, which is not the same question as the topic's
    standing relative to that person.

    Returns [0, 1]:
        0.0  no shared language, or no divergence evidence
        0.3  heavy overlap — likely an echo chamber
        0.5  the sweet spot for wrestling
        0.8  strong divergence with enough shared ground
        1.0  theoretical maximum; in practice may indicate incompatibility
    """
    intelligibility = _clamp01(intelligibility)
    if intelligibility < I_FLOOR:
        return 0.0
    novelty = _clamp01(novelty)
    bonus = 1.0 + LAMBDA_TRAJECTORY * _clamp01(trajectory)
    return _clamp01(intelligibility * novelty * bonus)


def _py_compute_suitedness(intelligibility: float, trajectory: float = 0.0) -> float:
    """How well-suited a person is to a conversation — depth on the topic
    plus live movement toward it, with NO divergence reward.

    The complement of `compute_tension`: tension finds who will push
    back, suitedness finds who can carry the thread. A host wants both
    kinds in the room, so they are scored separately rather than blended
    into one ranking.
    """
    intelligibility = _clamp01(intelligibility)
    if intelligibility < I_FLOOR:
        return 0.0
    return _clamp01(intelligibility * (1.0 + LAMBDA_TRAJECTORY * _clamp01(trajectory)))



# ─────────────────────────────────────────────────────────────────────
# A person against a person
# ─────────────────────────────────────────────────────────────────────

# Shared ground: how much comes from the two people's DNA vs. the topic.
W_PEOPLE = 0.5
W_TOPIC = 0.5


def _py_pair_intelligibility(cos_people: float, cos_topic: Optional[float] = None) -> float:
    """Shared ground between two people inside a topic.

    Args:
        cos_people: raw cosine between the two people's composite vectors.
        cos_topic: the LOWER of the two people's raw cosines to the topic's
            vector (both must speak the topic), or None when there is
            no topic.
    """
    people = _py_normalize_cosine(cos_people)
    if cos_topic is None:
        return people
    return _clamp01(W_PEOPLE * people + W_TOPIC * _py_normalize_cosine(cos_topic))


def _py_pair_novelty(contest_strength: float) -> float:
    """Divergence from recorded positions: summed strength of the contest-
    axis lines between the works that shaped each of them, saturating."""
    strength = max(0.0, float(contest_strength or 0.0))
    return _clamp01(1.0 - math.exp(-strength))


def _py_pair_tension(intelligibility: float, novelty: float) -> float:
    """The same tension curve with no trajectory bonus."""
    return _py_compute_tension(intelligibility, novelty, 0.0)


def _py_car_score(pair_scores: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over the occupants we could read; None when we could read none."""
    usable = [float(p) for p in pair_scores if p is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)




# ─────────────────────────────────────────────────────────────────────
# Absorptive capacity
# ─────────────────────────────────────────────────────────────────────

# How strongly accumulated knowledge widens what a person can take in.
# Zero until it is fitted against outcomes: at zero `absorb` changes
# nothing, and the model scores everyone on one curve.
KAPPA_ABSORB = 0.0


def _py_absorb_with(kappa: float, intelligibility: float, capacity: float) -> float:
    """Shared ground as a person of capacity `c` experiences it (Nooteboom's
    third hypothesis): `1 − (1 − I)^(1 + κ·c)`.

    It never lowers shared ground and leaves 0 and 1 where they are. With
    `I = 1 − d` against `N = d`, the peak of the product sits at
    `d* = (2 + κc)^(−1/(1 + κc))`, which moves outward as capacity grows.
    A negative or unreadable κ is no adjustment.
    """
    kappa = float(kappa)
    k = kappa if kappa > 0.0 else 0.0
    return _clamp01(1.0 - (1.0 - _clamp01(intelligibility)) ** (1.0 + k * _clamp01(capacity)))


def _py_absorb(intelligibility: float, capacity: float) -> float:
    """`absorb_with` at the model's own κ."""
    return _py_absorb_with(KAPPA_ABSORB, intelligibility, capacity)


# ─────────────────────────────────────────────────────────────────────
# Lines of disagreement that are not independent
# ─────────────────────────────────────────────────────────────────────

# How far several lines against the same work are one disagreement seen
# from several sides (1) or separate disagreements (0). Zero until fitted:
# every line counts in full, as it always has.
LINE_DEPENDENCE = 0.0


def _py_line_strength_with(rho: float, strengths: Iterable[float]) -> float:
    """The strength of all the lines that stand against one work:
    `max + (1 − ρ)·(sum − max)`. At ρ = 0 the plain sum; at ρ = 1 only the
    strongest line counts. Strengths below zero count as none."""
    total = 0.0
    strongest = 0.0
    for raw in strengths:
        x = float(raw)
        x = x if x > 0.0 else 0.0
        total = total + x
        strongest = x if x > strongest else strongest
    return strongest + (1.0 - _clamp01(rho)) * (total - strongest)


def _py_line_strength(strengths: Iterable[float]) -> float:
    """`line_strength_with` at the model's own ρ."""
    return _py_line_strength_with(LINE_DEPENDENCE, strengths)


# ─────────────────────────────────────────────────────────────────────
# Groups
# ─────────────────────────────────────────────────────────────────────


def _py_group_floor(matrix: Iterable[Iterable[Optional[float]]]) -> Optional[float]:
    """The least well-met person in a group.

    For each member, the best pair score they have with anyone else in it;
    the group's floor is the lowest of those. A group can have a fine mean
    and still hold someone with no counterpart at all: the mean dips, the
    floor goes to zero. `matrix[i][j]` is the pair score of members i and j,
    None where it could not be read (the diagonal is ignored). Members with
    no readable pair are left out; a group with none has no floor.
    """
    floor: Optional[float] = None
    for i, row in enumerate(matrix):
        best: Optional[float] = None
        for j, value in enumerate(row):
            if j == i or value is None:
                continue
            x = float(value)
            if math.isnan(x):
                continue
            best = x if best is None or x > best else best
        if best is None:
            continue
        floor = best if floor is None or best < floor else floor
    return floor


# ─────────────────────────────────────────────────────────────────────
# The public functions: Haskell when its library loads, else Python
# ─────────────────────────────────────────────────────────────────────

_HS = _haskell.load()
BACKEND = "haskell" if _HS is not None else "python"


def _serve(py, name):
    """The public function: the Haskell one when loaded (keeping the
    Python one's name and docstring), else the Python one itself."""
    if _HS is None:
        return py
    hs = getattr(_HS, name)

    @functools.wraps(py)
    def served(*args, **kwargs):
        return hs(*args, **kwargs)

    return served


normalize_cosine = _serve(_py_normalize_cosine, "normalize_cosine")
compute_intelligibility = _serve(_py_compute_intelligibility, "compute_intelligibility")
compute_novelty = _serve(_py_compute_novelty, "compute_novelty")
compute_trajectory = _serve(_py_compute_trajectory, "compute_trajectory")
compute_tension = _serve(_py_compute_tension, "compute_tension")
compute_suitedness = _serve(_py_compute_suitedness, "compute_suitedness")

pair_intelligibility = _serve(_py_pair_intelligibility, "pair_intelligibility")
pair_novelty = _serve(_py_pair_novelty, "pair_novelty")
pair_tension = _serve(_py_pair_tension, "pair_tension")
car_score = _serve(_py_car_score, "car_score")
absorb_with = _serve(_py_absorb_with, "absorb_with")
absorb = _serve(_py_absorb, "absorb")
line_strength_with = _serve(_py_line_strength_with, "line_strength_with")
line_strength = _serve(_py_line_strength, "line_strength")
group_floor = _serve(_py_group_floor, "group_floor")
