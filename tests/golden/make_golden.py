# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Freeze the model's outputs (`tests/golden/tension.json`).

The model is the Haskell library (`haskell/`). This writes what it returns on a
fixed set of inputs, to the bit, so `tests/test_golden.py` can show that a
change to the library or to the binding has not moved any score. Run it again
only when the model is meant to change, and say why in the commit:

    make -C haskell lib && python3 -m tests.golden.make_golden

The inputs are the edges by hand plus a seeded random spread either side of
[0, 1] (out-of-range values are real: a noisy cosine, a ratio over 1).
"""

import json
import random
from pathlib import Path

import tension

OUT = Path(__file__).with_name("tension.json")

EDGES = [
    -1.0, -1e-9, 0.0, 1e-12, 0.1499999, 0.15, 0.1500001, 0.25, 0.5,
    0.85, 0.999999, 1.0, 1.0000001, 2.0,
]
SEED = 20261121
N = 400


def bits(x):
    """A float as its exact hex form, so the file compares to the bit."""
    return None if x is None else float(x).hex()


def cases():
    rng = random.Random(SEED)
    r = lambda: rng.uniform(-0.5, 1.5)  # noqa: E731
    one = [(e,) for e in EDGES] + [(r(),) for _ in range(N)]
    two = [(a, b) for a in EDGES for b in EDGES] + [(r(), r()) for _ in range(N)]
    three = [(a, b, c) for a in EDGES[::3] for b in EDGES[::3] for c in EDGES[::3]] + [
        (r(), r(), r()) for _ in range(N)
    ]
    hops = range(-1, 7)
    lists = [[], [0.0], [0.5], [1.0, 1.0], *[[r() for _ in range(rng.randint(1, 8))] for _ in range(60)]]

    def matrix(n):
        m = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                m[i][j] = m[j][i] = rng.choice([None, rng.random()])
        return m

    def run(fn, args):
        return [[list(a), bits(getattr(tension, fn)(*a))] for a in args]

    return {
        "normalize_cosine": run("normalize_cosine", one),
        "compute_intelligibility": run("compute_intelligibility", two),
        "compute_novelty": run("compute_novelty", two),
        "compute_trajectory": run("compute_trajectory", two),
        "compute_tension": run("compute_tension", three),
        "compute_suitedness": run("compute_suitedness", two),
        "pair_intelligibility": run("pair_intelligibility", two + [(a[0], None) for a in one]),
        "pair_novelty": run("pair_novelty", one + [(rng.uniform(0, 12),) for _ in range(100)]),
        "pair_tension": run("pair_tension", two),
        "absorb_with": run("absorb_with", [(k, a, b) for k in (0.0, 0.5, 1.0, 3.0) for a, b in two[::4]]),
        "absorb": run("absorb", two),
        "line_strength_with": run("line_strength_with", [(rho, xs) for rho in (-0.5, 0.0, 0.3, 1.0, 1.5) for xs in lists]),
        "line_strength": run("line_strength", [(xs,) for xs in lists]),
        "hop_weight": run("hop_weight", [(h,) for h in hops]),
        "hop_weight_with": run("hop_weight_with", [
            (rh, nv, mh, h)
            for rh in (0.0, 0.5, 0.7, 1.0)
            for nv in (0.0, 0.3, 0.6, 1.0)
            for mh in (-1, 1, 2, 4, 6)
            for h in hops
        ]),
        "path_contest": run("path_contest", [
            (h, d, s) for h in hops for d in range(-1, 5) for s in (-0.5, 0.0, 0.3, 1.0, 1.5)
        ]),
        "path_contest_with": run("path_contest_with", [
            (rh, nv, mh, h, d, s)
            for rh in (0.0, 0.7, 1.0)
            for nv in (0.0, 0.6, 1.0)
            for mh in (1, 4)
            for h in hops
            for d in range(0, 4)
            for s in (0.0, 0.5, 1.0)
        ]),
        "car_score": run("car_score", [([None],), ([0.2, None, 0.6],)] + [(xs,) for xs in lists]),
        "group_floor": run("group_floor", [(matrix(rng.randint(1, 6)),) for _ in range(200)]),
    }


def main():
    out = {"constants": dict(tension.model._HS.constants), **cases()}
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{OUT.name}: {sum(len(v) for k, v in out.items() if k != 'constants')} outputs")


if __name__ == "__main__":
    main()
