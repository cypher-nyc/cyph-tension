# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Fit the model's constants to what actually happened.

A record is one pair of people who sat together, the readings the model had
of them beforehand, and what came of it:

    {"ground": 0.62,          shared ground on the model's band, before capacity
     "strength": 1.4,         summed strength of the lines between them
     "capacity_a": 0.5, "capacity_b": 0.9,
     "outcome": {"revised": 1, "read_next": 1, "continued": 1}}

Three outcomes, in the order the model trusts them:

    revised      one of them changed a recorded position, or recorded a new
                 line, after the conversation. The claim the model makes.
    read_next    one of them went on to read something from the other's side.
    continued    they kept talking. Recorded, never fitted alone: a model
                 tuned to keep people talking is a feed.

The fit asks one thing of a setting of the constants: across all records,
do pairs that scored higher go on to the outcome more often? That is the
area under the ROC curve (the chance a pair that reached the outcome scored
above one that did not), which assumes no curve linking score to outcome.
The constants are searched on a grid using half the records and judged on
the other half; the gain over the current constants comes with a bootstrap
interval, and `adopt` is true only when that interval clears zero. Under
`MIN_RECORDS` it says nothing at all.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence

import tension

OUTCOMES = ("revised", "read_next", "continued")
MIN_RECORDS = 200
GRID = {
    "floor": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "kappa": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
}


class NotEnough(ValueError):
    """Too few records, or only one kind of outcome, to say anything."""


def score(record: dict, floor: float, kappa: float) -> float:
    """The pair score under a given floor and κ. The curve is the model's;
    only the two constants being fitted are passed in."""
    ground = float(record["ground"])
    mutual = min(
        tension.absorb_with(kappa, ground, float(record.get("capacity_a", 0.0))),
        tension.absorb_with(kappa, ground, float(record.get("capacity_b", 0.0))),
    )
    if mutual < floor:
        return 0.0
    return min(1.0, max(0.0, mutual * tension.pair_novelty(float(record["strength"]))))


def auc(scores: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """The chance a pair that reached the outcome scored above one that did
    not (ties count half). None when every pair went the same way."""
    ranked = sorted(zip(scores, outcomes))
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        rank_sum += mean_rank * sum(o for _, o in ranked[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def fit(records: List[dict], outcome: str = "revised", seed: int = 0, resamples: int = 200) -> Dict[str, object]:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome is one of {', '.join(OUTCOMES)}")
    if len(records) < MIN_RECORDS:
        raise NotEnough(f"{len(records)} records; the fit says nothing under {MIN_RECORDS}")
    labels = [1 if r.get("outcome", {}).get(outcome) else 0 for r in records]
    if sum(labels) in (0, len(labels)):
        raise NotEnough(f"every pair has the same {outcome!r} outcome; there is nothing to tell apart")

    def evaluate(rows: Sequence[int], floor: float, kappa: float) -> Optional[float]:
        return auc([score(records[i], floor, kappa) for i in rows], [labels[i] for i in rows])

    # Fit on one half, judge on the other: constants chosen on the records
    # they are then scored on always look better than they are.
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    train, test = order[: len(order) // 2], order[len(order) // 2 :]
    grid = list(itertools.product(GRID["floor"], GRID["kappa"]))
    scored = [(evaluate(train, f, k), f, k) for f, k in grid]
    scored = [x for x in scored if x[0] is not None]
    if not scored or evaluate(test, tension.I_FLOOR, tension.KAPPA_ABSORB) is None:
        raise NotEnough(f"too few pairs reached {outcome!r} to split the records and still tell them apart")
    _, best_floor, best_kappa = max(scored, key=lambda x: (x[0], -x[2], -x[1]))
    fitted = evaluate(test, best_floor, best_kappa)
    current = evaluate(test, tension.I_FLOOR, tension.KAPPA_ABSORB)

    draws = []
    for _ in range(resamples):
        sample = [test[rng.randrange(len(test))] for _ in test]
        a, b = evaluate(sample, best_floor, best_kappa), evaluate(sample, tension.I_FLOOR, tension.KAPPA_ABSORB)
        if a is not None and b is not None:
            draws.append(a - b)
    draws.sort()
    low, high = (draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]) if draws else (None, None)
    return {
        "outcome": outcome,
        "records": len(records),
        "reached_outcome": sum(labels),
        "fitted_on": len(train),
        "judged_on": len(test),
        "current": {"floor": tension.I_FLOOR, "kappa": tension.KAPPA_ABSORB, "auc": round(current, 4)},
        "fitted": {"floor": best_floor, "kappa": best_kappa, "auc": round(fitted, 4)},
        # how much better the fitted constants order pairs they were not fitted on, with a 95% interval
        "gain": {"auc": round(fitted - current, 4), "low": None if low is None else round(low, 4),
                 "high": None if high is None else round(high, 4)},
        "adopt": bool(low is not None and low > 0),
    }
