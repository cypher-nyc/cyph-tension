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


class WorkTensionTests(unittest.TestCase):
    """A person against a work they have not met (2026-11-23)."""

    def test_no_ground_no_score_however_contested(self):
        self.assertEqual(tension.work_tension(0.0, 0.0, 10.0), 0.0)

    def test_no_contest_no_score_however_close(self):
        self.assertEqual(tension.work_tension(0.9, 1.0, 0.0), 0.0)

    def test_close_and_contested_scores(self):
        t = tension.work_tension(0.7, 0.6, 1.0)
        self.assertGreater(t, 0.0)
        self.assertLessEqual(t, 1.0)

    def test_on_the_models_one_curve(self):
        for c, d, s in [(0.5, 0.2, 0.3), (0.8, 0.9, 2.0), (0.3, 0.0, 1.0)]:
            self.assertEqual(
                tension.work_tension(c, d, s),
                tension.compute_tension(tension.compute_intelligibility(c, d), tension.pair_novelty(s), 0.0),
            )

    def test_more_engagement_never_lowers_it(self):
        self.assertGreaterEqual(tension.work_tension(0.6, 0.8, 1.0), tension.work_tension(0.6, 0.2, 1.0))


class PairScore(unittest.TestCase):
    """The hardened pair score (Cyph.Pair), through the library."""

    SAMPLE = [i / 100 for i in range(10, 91)]

    def test_the_middle_ground_scores(self):
        # the middle ground's readings (I = 0.50, s = 1.86), here with five items each
        t = tension.pair_score(self.SAMPLE, 5, 5, 0.50, 1.8575)
        self.assertIsNotNone(t)
        self.assertGreater(t, 0.0)

    def test_a_small_library_is_not_scored(self):
        self.assertIsNone(tension.pair_score(self.SAMPLE, tension.MIN_LIBRARY - 1, 400, 0.5, 1.0))

    def test_symmetric_and_bounded(self):
        for na, nb, cos, s in [(5, 400, 0.3, 0.5), (12, 40, 0.7, 2.0), (500, 500, 0.12, 3.0)]:
            ab = tension.pair_score(self.SAMPLE, na, nb, cos, s)
            ba = tension.pair_score(self.SAMPLE, nb, na, cos, s)
            self.assertEqual(ab, ba)
            self.assertTrue(0.0 <= ab <= 1.0)

    def test_engagement_depth_lifts_ground(self):
        light = tension.pair_ground(self.SAMPLE, 5, 5, 0.4)
        deep = tension.pair_ground(self.SAMPLE, 500, 500, 0.4)
        self.assertGreater(deep, light)
        self.assertEqual(tension.reader_capacity(tension.N_REF), 1.0)

    def test_an_unusable_sample_is_an_error(self):
        with self.assertRaises(ValueError):
            tension.pair_score([0.4, 0.4], 10, 10, 0.4, 1.0)

    def test_table_mean_and_lower_bound(self):
        mean, lower = tension.table_score([0.2, None, 0.6])
        self.assertAlmostEqual(mean, 0.4)
        self.assertAlmostEqual(lower, 0.2)
        self.assertEqual(tension.table_score([0.3]), (0.3, None))
        self.assertEqual(tension.table_score([]), (None, None))

