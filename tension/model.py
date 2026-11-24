# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The productive-tension model, from Python.

    tension = gate(I) · I · N · (1 + λ·T)

The model is written once, in Haskell (`haskell/src/Cyph/Tension.hs` and
`Cyph/Pair.hs`), where its claims are stated as properties and checked across
generated inputs (`make -C haskell test`). This module is its Python face: the
constants and the signatures, each function a call into the compiled library
(`tension/_haskell.py`; build it with `make -C haskell lib`). There is no
Python copy of the model and no fallback. `tests/test_golden.py` pins every
output to the bit against `tests/golden/tension.json`.

The reasoning and the research behind each term are in `docs/MODEL.md`.
Every function is pure: scalars in, a scalar out, no I/O.
"""

from __future__ import annotations

from typing import Iterable, Optional

from tension import _haskell

# The model is the compiled Haskell library; without it nothing here can score.
_HS = _haskell.require()
BACKEND = "haskell"

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




def normalize_cosine(cosine: float) -> float:
    """Map a raw embedding cosine onto [0, 1] across the band where real
    document similarities actually live."""
    return _HS.normalize_cosine(cosine)


def compute_intelligibility(cos_to_anchor: float, depth_norm: float) -> float:
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
    return _HS.compute_intelligibility(cos_to_anchor, depth_norm)


def compute_novelty(divergent_reading_ratio: float, stance_ratio: float) -> float:
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
    return _HS.compute_novelty(divergent_reading_ratio, stance_ratio)


def compute_trajectory(cos_recent: float, cos_older: float) -> float:
    """Learning progress toward the topic (Kidd et al. 2012; Ten et al.
    2021): how much closer a person's recent territory sits to the topic
    than their older territory did.

    Only motion TOWARD the topic is rewarded — drift away is not
    penalized, because a person moving on from a subject still knows it.
    Returns [0, 1], saturating at T_MAX of cosine movement.
    """
    return _HS.compute_trajectory(cos_recent, cos_older)


def compute_tension(
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
    return _HS.compute_tension(intelligibility, novelty, trajectory)


def compute_suitedness(intelligibility: float, trajectory: float = 0.0) -> float:
    """How well-suited a person is to a conversation — depth on the topic
    plus live movement toward it, with NO divergence reward.

    The complement of `compute_tension`: tension finds who will push
    back, suitedness finds who can carry the thread. A host wants both
    kinds in the room, so they are scored separately rather than blended
    into one ranking.
    """
    return _HS.compute_suitedness(intelligibility, trajectory)



# ─────────────────────────────────────────────────────────────────────
# A person against a person
# ─────────────────────────────────────────────────────────────────────

# Shared ground: how much comes from the two people's DNA vs. the topic.
W_PEOPLE = 0.5
W_TOPIC = 0.5


def pair_intelligibility(cos_people: float, cos_topic: Optional[float] = None) -> float:
    """Shared ground between two people inside a topic.

    Args:
        cos_people: raw cosine between the two people's composite vectors.
        cos_topic: the LOWER of the two people's raw cosines to the topic's
            vector (both must speak the topic), or None when there is
            no topic.
    """
    return _HS.pair_intelligibility(cos_people, cos_topic)


def pair_novelty(contest_strength: float) -> float:
    """Divergence from recorded positions: summed strength of the contest-
    axis lines between the works that shaped each of them, saturating."""
    return _HS.pair_novelty(contest_strength)


def pair_tension(intelligibility: float, novelty: float) -> float:
    """The same tension curve with no trajectory bonus."""
    return _HS.pair_tension(intelligibility, novelty)


def work_intelligibility(cos_to_work: float, depth: float) -> float:
    """Shared ground between a person and a work they have not met: half the
    cosine of their composite and the work's vector (banded), half their
    engagement in the work's neighbourhood (`depth`, in [0, 1])."""
    return _HS.work_intelligibility(cos_to_work, depth)


def work_novelty(contest_strength: float) -> float:
    """What the work could push against in them: the summed strength of the
    contest lines between it and the works that shaped them, as 1 − e^(−s).
    Graph evidence only."""
    return _HS.work_novelty(contest_strength)


def work_tension(cos_to_work: float, depth: float, contest_strength: float) -> float:
    """Productive tension between a person and a work: gate(I) · I · N with
    the two terms above. Zero below the floor, zero with no contest."""
    return _HS.work_tension(cos_to_work, depth, contest_strength)


def car_score(pair_scores: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over the occupants we could read; None when we could read none."""
    return _HS.car_score(pair_scores)




# ─────────────────────────────────────────────────────────────────────
# Absorptive capacity
# ─────────────────────────────────────────────────────────────────────

# How strongly accumulated knowledge widens what a person can take in.
# Zero until it is fitted against outcomes: at zero `absorb` changes
# nothing, and the model scores everyone on one curve.
KAPPA_ABSORB = 0.0


def absorb_with(kappa: float, intelligibility: float, capacity: float) -> float:
    """Shared ground as a person of capacity `c` experiences it (Nooteboom's
    third hypothesis): `1 − (1 − I)^(1 + κ·c)`.

    It never lowers shared ground and leaves 0 and 1 where they are. With
    `I = 1 − d` against `N = d`, the peak of the product sits at
    `d* = (2 + κc)^(−1/(1 + κc))`, which moves outward as capacity grows.
    A negative or unreadable κ is no adjustment.
    """
    return _HS.absorb_with(kappa, intelligibility, capacity)


def absorb(intelligibility: float, capacity: float) -> float:
    """`absorb_with` at the model's own κ."""
    return _HS.absorb(intelligibility, capacity)


# ─────────────────────────────────────────────────────────────────────
# Lines of disagreement that are not independent
# ─────────────────────────────────────────────────────────────────────

# How far several lines against the same work are one disagreement seen
# from several sides (1) or separate disagreements (0). Zero until fitted:
# every line counts in full, as it always has.
LINE_DEPENDENCE = 0.0


def line_strength_with(rho: float, strengths: Iterable[float]) -> float:
    """The strength of all the lines that stand against one work:
    `max + (1 − ρ)·(sum − max)`. At ρ = 0 the plain sum; at ρ = 1 only the
    strongest line counts. Strengths below zero count as none."""
    return _HS.line_strength_with(rho, strengths)


def line_strength(strengths: Iterable[float]) -> float:
    """`line_strength_with` at the model's own ρ."""
    return _HS.line_strength(strengths)


# ─────────────────────────────────────────────────────────────────────
# Contests reached through the graph (the inverted U, in hops)
# ─────────────────────────────────────────────────────────────────────

# How much of a contest survives each further hop between two works. A
# starting point, not fitted.
HOP_REACH = 0.7
# How fast what is new to a reader builds up with hops; at 0 only reach is
# left. A starting point, not fitted.
HOP_NOVELTY = 0.6
# The hop budget: a path longer than this is not evidence.
MAX_HOPS = 4


def hop_weight_with(reach: float, novelty: float, max_hops: int, hops: int) -> float:
    """The weight of a contest by how many hops of the graph it is reached
    through: `reach^(h − 1) · (1 − novelty^h)`, divided by its largest
    value within the budget, so the best distance weighs 1.

    Reach falls with hops and novelty rises, so their product rises to one
    peak and falls from it: the inverted U is in the formula. Outside
    `1..max_hops` the weight is 0. With `novelty = 0` it is plain decay
    from a direct line, and with `max_hops = 1` only direct lines count, at
    full weight: the model as it was before paths.
    """
    return _HS.hop_weight_with(reach, novelty, max_hops, hops)


def hop_weight(hops: int) -> float:
    """`hop_weight_with` at the model's own reach, novelty and budget."""
    return _HS.hop_weight(hops)


def path_contest_with(
    reach: float, novelty: float, max_hops: int, hops: int, dissonant: int, strength: float
) -> float:
    """What one path between two works adds to the strength of the contest
    between their readers.

    A path is a contest when an odd number of its edges are on the
    dissonance axis: a work that builds on what another refutes stands
    against it, while two works that refute the same third stand together.
    Its strength (the product of its edges', from the caller) is held to
    [0, 1], unreadable counting as none, and weighed by its length.
    """
    return _HS.path_contest_with(reach, novelty, max_hops, hops, dissonant, strength)


def path_contest(hops: int, dissonant: int, strength: float) -> float:
    """`path_contest_with` at the model's own reach, novelty and budget."""
    return _HS.path_contest(hops, dissonant, strength)


# ─────────────────────────────────────────────────────────────────────
# Groups
# ─────────────────────────────────────────────────────────────────────


def group_floor(matrix: Iterable[Iterable[Optional[float]]]) -> Optional[float]:
    """The least well-met person in a group.

    For each member, the best pair score they have with anyone else in it;
    the group's floor is the lowest of those. A group can have a fine mean
    and still hold someone with no counterpart at all: the mean dips, the
    floor goes to zero. `matrix[i][j]` is the pair score of members i and j,
    None where it could not be read (the diagonal is ignored). Members with
    no readable pair are left out; a group with none has no floor.
    """
    return _HS.group_floor(matrix)
