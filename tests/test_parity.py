# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The Haskell model and the Python model are one model: the same constants and
the same value, to the bit, on every input. The edges by hand, and many
random inputs either side of [0, 1] (out-of-range values are real: a noisy
cosine, a ratio over 1).

Skipped when the library has not been built (`make -C haskell lib`).
"""

import math
import random
import unittest

from tension import _haskell, model

try:
    HS = _haskell.open_library()
except (OSError, AttributeError):
    HS = None

N = 20_000
EDGES = [
    -1.0, -1e-9, 0.0, 1e-12, 0.1499999, 0.15, 0.1500001, 0.25, 0.5,
    0.85, 0.999999, 1.0, 1.0000001, 2.0, float("inf"), float("-inf"), float("nan"),
]


def same(a, b):
    """Equal to the bit, NaN equal to NaN, None equal to None."""
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def cases(arity, seed):
    rng = random.Random(seed)
    if arity == 1:
        for e in EDGES:
            yield (e,)
    else:
        for e in EDGES:
            for f in EDGES[:: max(1, len(EDGES) // 6)]:
                yield (e, f) + (0.5,) * (arity - 2)
                yield (f, e) + (0.5,) * (arity - 2)
    for _ in range(N):
        yield tuple(rng.uniform(-0.5, 1.5) for _ in range(arity))


@unittest.skipIf(HS is None, "Haskell library not built (make -C haskell lib)")
class Parity(unittest.TestCase):
    def check(self, name, py, arity, seed):
        hs = getattr(HS, name)
        for args in cases(arity, seed):
            a, b = py(*args), hs(*args)
            self.assertTrue(same(a, b), f"{name}{args}: python {a!r}, haskell {b!r}")

    def test_constants(self):
        for name, value in HS.constants.items():
            self.assertEqual(getattr(model, name), value, name)

    def test_normalize_cosine(self):
        self.check("normalize_cosine", model._py_normalize_cosine, 1, 1)

    def test_intelligibility(self):
        self.check("compute_intelligibility", model._py_compute_intelligibility, 2, 2)

    def test_novelty(self):
        self.check("compute_novelty", model._py_compute_novelty, 2, 3)

    def test_trajectory(self):
        self.check("compute_trajectory", model._py_compute_trajectory, 2, 4)

    def test_tension(self):
        self.check("compute_tension", model._py_compute_tension, 3, 5)

    def test_suitedness(self):
        self.check("compute_suitedness", model._py_compute_suitedness, 2, 6)

    def test_pair_intelligibility(self):
        self.check("pair_intelligibility", model._py_pair_intelligibility, 2, 7)
        for (c,) in cases(1, 8):
            self.assertTrue(same(model._py_pair_intelligibility(c, None), HS.pair_intelligibility(c, None)))

    def test_pair_novelty(self):
        self.check("pair_novelty", model._py_pair_novelty, 1, 9)
        rng = random.Random(10)
        for _ in range(N):
            s = rng.uniform(0, 12)
            self.assertTrue(same(model._py_pair_novelty(s), HS.pair_novelty(s)))

    def test_pair_tension(self):
        self.check("pair_tension", model._py_pair_tension, 2, 11)

    def test_car_score(self):
        rng = random.Random(12)
        self.assertIsNone(HS.car_score([]))
        self.assertIsNone(HS.car_score([None, None]))
        for _ in range(2_000):
            scores = [rng.choice([None, rng.random()]) for _ in range(rng.randint(0, 8))]
            self.assertTrue(same(model._py_car_score(scores), HS.car_score(scores)), scores)

    def test_absorb(self):
        self.check("absorb_with", model._py_absorb_with, 3, 13)
        self.check("absorb", model._py_absorb, 2, 14)
        rng = random.Random(15)
        for _ in range(N):
            args = (rng.uniform(0, 6), rng.random(), rng.random())
            self.assertTrue(same(model._py_absorb_with(*args), HS.absorb_with(*args)), args)

    def test_line_strength(self):
        rng = random.Random(16)
        self.assertEqual(HS.line_strength_with(0.5, []), model._py_line_strength_with(0.5, []))
        for _ in range(5_000):
            rho = rng.uniform(-0.5, 1.5)
            xs = [rng.uniform(-0.5, 2.0) for _ in range(rng.randint(0, 9))]
            self.assertTrue(same(model._py_line_strength_with(rho, xs), HS.line_strength_with(rho, xs)), (rho, xs))
            self.assertTrue(same(model._py_line_strength(xs), HS.line_strength(xs)), xs)

    def test_group_floor(self):
        rng = random.Random(17)
        for _ in range(3_000):
            n = rng.randint(1, 6)
            m = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    m[i][j] = m[j][i] = rng.choice([None, rng.random()])
            self.assertTrue(same(model._py_group_floor(m), HS.group_floor(m)), m)



if __name__ == "__main__":
    unittest.main()
