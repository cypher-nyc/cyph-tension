# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The claims the model makes, by example, on whichever backend is serving.
(`haskell/test/Properties.hs` states the same claims as laws.)"""

import unittest

import tension


class Model(unittest.TestCase):
    def test_zero_below_the_floor_whatever_the_divergence(self):
        self.assertEqual(tension.compute_tension(tension.I_FLOOR - 0.01, 1.0, 1.0), 0.0)

    def test_zero_without_evidence_of_divergence(self):
        self.assertEqual(tension.compute_tension(0.9, 0.0, 1.0), 0.0)

    def test_inverted_u(self):
        # as two people move apart, ground falls and divergence rises:
        # the product peaks strictly inside
        scores = [tension.compute_tension(1.0 - d, d) for d in (i / 20 for i in range(21))]
        peak = scores.index(max(scores))
        self.assertTrue(0 < peak < 20)
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[-1], 0.0)

    def test_trajectory_modulates_and_never_creates(self):
        self.assertGreater(tension.compute_tension(0.6, 0.5, 1.0), tension.compute_tension(0.6, 0.5, 0.0))
        self.assertEqual(tension.compute_tension(0.05, 0.5, 1.0), 0.0)

    def test_drift_away_is_not_penalised(self):
        self.assertEqual(tension.compute_trajectory(0.4, 0.6), 0.0)

    def test_pair_novelty_saturates(self):
        self.assertEqual(tension.pair_novelty(0.0), 0.0)
        self.assertLess(tension.pair_novelty(1.0), tension.pair_novelty(3.0))
        self.assertLessEqual(tension.pair_novelty(50.0), 1.0)

    def test_pair_ground_without_a_topic_is_the_people_half(self):
        self.assertEqual(tension.pair_intelligibility(0.55), tension.normalize_cosine(0.55))

    def test_group_mean_ignores_the_unreadable(self):
        self.assertEqual(tension.car_score([0.2, None, 0.4]), 0.3)
        self.assertIsNone(tension.car_score([None]))

    def test_scores_stay_in_the_unit_interval(self):
        for i in (-1.0, 0.0, 0.3, 1.0, 7.0):
            for n in (-1.0, 0.0, 0.6, 1.0, 7.0):
                self.assertTrue(0.0 <= tension.compute_tension(i, n, 2.0) <= 1.0)


if __name__ == "__main__":
    unittest.main()
