# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The importers and the ranking, on the libraries in examples/."""

import math
import os
import random
import unittest

import tension
from playground import band, calibrate, embed, importers, rank
from playground.importers import Item

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEOPLE = os.path.join(ROOT, "examples", "people")


def room():
    people = {e.split(".")[0]: importers.load_path(os.path.join(PEOPLE, e)) for e in sorted(os.listdir(PEOPLE))}
    with open(os.path.join(ROOT, "examples", "relations.tsv"), encoding="utf-8") as handle:
        return people, rank.parse_relations(handle.read())


class Importers(unittest.TestCase):
    def test_every_format_reads(self):
        people, _ = room()
        self.assertEqual({n: len(i) for n, i in people.items()},
                         {"ada": 6, "ben": 6, "cleo": 6, "dev": 6, "eze": 6, "fay": 6})
        self.assertEqual({people[n][0].source for n in people},
                         {"obsidian", "are.na", "bookmarks", "csv", "list", "json"})

    def test_obsidian_records_a_stance_inline_and_in_frontmatter(self):
        notes = [
            ("a.md", "---\ntitle: First\ncontests: \"[[Second]]\"\nrefutes:\n  - Third\n---\nbody [[Second|alias]]"),
            ("b.md", "complicates:: [[Fourth]], [[Fifth#part]]\n"),
            (".obsidian/x.md", "ignored"),
        ]
        items = importers.parse_obsidian(notes)
        self.assertEqual([i.title for i in items], ["First", "b"])
        self.assertEqual(items[0].stances, [("contests", "Second"), ("refutes", "Third")])
        self.assertEqual(items[1].stances, [("complicates", "Fourth"), ("complicates", "Fifth")])
        self.assertIn("Second", items[0].text)

    def test_bookmarks_keep_only_links(self):
        items = importers.parse_bookmarks('<DL><DT><A HREF="https://e.org/a" TAGS="x,y">A  title</A><DT><A HREF="place:sort=8">menu</A></DL>')
        self.assertEqual([(i.title, i.url, i.text) for i in items], [("A title", "https://e.org/a", "x y")])

    def test_csv_needs_a_title_column(self):
        self.assertEqual(importers.parse_csv("Title,Author\nLeviathan,Hobbes\n,\n")[0].text, "Hobbes")
        with self.assertRaises(importers.UnreadableSource):
            importers.parse_csv("a,b\n1,2\n")

    def test_a_list_reads_bullets_and_links(self):
        items = importers.parse_lines("# heading\n- One\n2. [Two](https://e.org/2) a note\n\n")
        self.assertEqual([(i.title, i.url, i.text) for i in items], [("One", "", ""), ("Two", "https://e.org/2", "a note")])

    def test_unknown_files_are_refused(self):
        with self.assertRaises(importers.UnreadableSource):
            importers.parse_file("photo.png", "")
        with self.assertRaises(importers.UnreadableSource):
            importers.fetch_arena_channel("https://example.com/../etc?x=1")


class Ranking(unittest.TestCase):
    def setUp(self):
        self.people, self.relations = room()

    def by_name(self, me, **kwargs):
        result = rank.rank(self.people, me, self.relations, **kwargs)
        return {row["name"]: row for row in result["ranked"]}, result

    def test_lenses(self):
        rows, result = self.by_name("ada")
        self.assertEqual(rows["ben"]["lens"], rank.LENS_CHALLENGE)
        self.assertEqual(rows["dev"]["lens"], rank.LENS_CHALLENGE)
        self.assertEqual(rows["fay"]["lens"], rank.LENS_GROUND_ONLY)   # shared ground, nothing on record
        self.assertEqual(rows["eze"]["lens"], rank.LENS_BELOW_FLOOR)   # bread and gardens
        self.assertEqual(result["ranked"][0]["lens"], rank.LENS_CHALLENGE)
        self.assertEqual(result["backend"], tension.BACKEND)

    def test_scores_come_from_the_model(self):
        rows, _ = self.by_name("ada")
        for row in rows.values():
            expected = tension.pair_tension(row["intelligibility"], row["novelty"])
            self.assertAlmostEqual(row["score"], expected, places=3)

    def test_no_relations_no_challenge(self):
        # divergence is never read off vector distance: with nothing on
        # record nobody is a challenge, however far apart they sit
        for items in self.people.values():
            for item in items:
                item.stances = []
        result = rank.rank(self.people, "ada", [])
        self.assertTrue(all(row["score"] == 0.0 for row in result["ranked"]))
        self.assertNotIn(rank.LENS_CHALLENGE, {row["lens"] for row in result["ranked"]})

    def test_a_recorded_stance_is_evidence_on_its_own(self):
        rows = {r["name"]: r for r in rank.rank(self.people, "ada", [])["ranked"]}
        self.assertEqual([(l["yours"], l["axis"], l["theirs"], l["toward"], l["from"]) for l in rows["ben"]["lines"]],
                         [("A Theory of Justice", "contests", "Anarchy, State, and Utopia", "them", "recorded")])

    def test_a_line_is_counted_once_each_way(self):
        rows, _ = self.by_name("ada")
        lines = [(l["yours"], l["theirs"], l["axis"], l["toward"]) for l in rows["ben"]["lines"]]
        self.assertEqual(len(lines), len(set(lines)))

    def test_lines_have_a_direction(self):
        # Nozick's book contests Rawls's: ben, who holds it, contests ada
        rows = {r["name"]: r for r in rank.rank(self.people, "ada", self.relations)["ranked"]}
        toward_ada = [l for l in rows["ben"]["lines"] if l["toward"] == "me"]
        self.assertIn(("A Theory of Justice", "Anarchy, State, and Utopia"), [(l["yours"], l["theirs"]) for l in toward_ada])
        # and the two directed scores are the same pair seen from each side
        back = {r["name"]: r for r in rank.rank(self.people, "ben", self.relations)["ranked"]}
        self.assertEqual(rows["ben"]["contested_by"], back["ada"]["contests"])
        self.assertEqual(rows["ben"]["contests"], back["ada"]["contested_by"])
        self.assertEqual(rows["ben"]["score"], back["ada"]["score"])   # the mutual score is mutual

    def test_capacity_widens_ground_only_when_kappa_is_set(self):
        plain = {r["name"]: r for r in rank.rank(self.people, "ada", self.relations)["ranked"]}
        wide = {r["name"]: r for r in rank.rank(self.people, "ada", self.relations, kappa=3.0)["ranked"]}
        self.assertEqual(plain["ben"]["intelligibility"], rank.rank(self.people, "ada", self.relations, kappa=0.0)["ranked"][1]["intelligibility"])
        self.assertGreater(wide["ben"]["intelligibility"], plain["ben"]["intelligibility"])
        self.assertEqual(wide["eze"]["intelligibility"], 0.0)   # capacity cannot make ground out of none
        self.assertAlmostEqual(rank.capacity(self.people["ada"]), 1 - 2.718281828459045 ** (-6 / 50), places=9)

    def test_dependent_lines_count_for_less(self):
        apart = {r["name"]: r for r in rank.rank(self.people, "ada", self.relations, rho=0.0)["ranked"]}
        as_one = {r["name"]: r for r in rank.rank(self.people, "ada", self.relations, rho=1.0)["ranked"]}
        # dev brings two works against the same book of ada's
        self.assertLess(as_one["dev"]["contest_strength"], apart["dev"]["contest_strength"])
        self.assertAlmostEqual(apart["dev"]["contest_strength"], 0.8 + 0.5 + 0.6, places=4)
        self.assertAlmostEqual(as_one["dev"]["contest_strength"], 0.8 + 0.6, places=4)

    def test_someone_with_nothing_is_unknown_and_last(self):
        self.people["nobody"] = []
        result = rank.rank(self.people, "ada", self.relations)
        self.assertEqual(result["ranked"][-1]["lens"], rank.LENS_UNKNOWN)
        self.assertIsNone(result["ranked"][-1]["score"])

    def test_several_assessors_of_one_line(self):
        parsed = rank.parse_relations(
            "Alpha | contests | Beta | 0.9 | first | p. 12\n"
            "alpha | contests | the beta | 0.2 | second\n"
            "Alpha | contests | Beta | 0.7 | third\n"
            "Gamma | refutes | Delta | 0.8 | first\nGamma | refutes | Delta | 0 | second\n"
        )
        self.assertEqual(len(parsed), 2)
        line = parsed[0]
        self.assertEqual((line.strength, line.assessments, line.by, line.cite), (0.7, [0.9, 0.2, 0.7], "first, second, third", "p. 12"))
        self.assertTrue(line.disputed)                     # 0.9 against 0.2
        self.assertEqual(parsed[1].strength, 0.4)          # one says it is there, one says it is not
        people = {"x": [Item("Alpha", "shared words here")], "y": [Item("Beta", "shared words here")]}
        row = rank.rank(people, "x", parsed)["ranked"][0]
        self.assertTrue(row["lines"][0]["disputed"])
        self.assertEqual(row["lines"][0]["strength"], 0.7)

    def test_a_group(self):
        table = rank.group(self.people, ["ada", "ben", "dev"], self.relations)
        self.assertEqual(len(table["pairs"]), 3)
        self.assertGreater(table["floor"], 0)
        self.assertAlmostEqual(table["mean"], round(sum(p["score"] for p in table["pairs"]) / 3, 4), places=4)
        # swap in someone with no counterpart: the mean dips, the floor goes to zero
        cold = rank.group(self.people, ["ada", "ben", "eze"], self.relations)
        self.assertGreater(cold["mean"], 0)
        self.assertEqual(cold["floor"], 0.0)
        self.assertEqual({c["name"]: c["with"] for c in cold["counterparts"]}, {"ada": "ben", "ben": "ada", "eze": None})
        with self.assertRaises(ValueError):
            rank.group(self.people, ["ada"], self.relations)

    def test_relations_file(self):
        parsed = rank.parse_relations("# c\nA | contests | B | 0.5\nA\tbuilds on\tB\nA\trefutes\tC\n")
        self.assertEqual([(r.source, r.axis, r.target, r.strength) for r in parsed],
                         [("A", "contests", "B", 0.5), ("A", "refutes", "C", 1.0)])
        with self.assertRaises(ValueError):
            rank.parse_relations("only two | fields")


class Band(unittest.TestCase):
    class Topical:
        """A stand-in source: one axis per shelf, plus a component every
        text shares (what makes library means sit close together)."""

        name, band = "topical", None

        def __init__(self, shared=1.0):
            self.shared, self.axes = shared, {}

        def embed(self, documents):
            import json
            with open(band.SHELVES, encoding="utf-8") as handle:
                shelves = json.load(handle)
            where = {text: shelf for shelf, texts in shelves.items() for text in texts}
            rng = random.Random(7)
            out = []
            for text in documents:
                v = {"shared": self.shared, where[text]: 1.0}
                v.update({("noise", i): rng.gauss(0, 0.15) for i in range(6)})
                out.append(v)
            return out

    def test_the_band_is_measured_between_unrelated_and_the_same_reader(self):
        found = band.measure(self.Topical())
        self.assertEqual((found["shelves"], found["unrelated_pairs"], found["same_reader_splits"]), (8, 28, 80))
        self.assertLess(found["unrelated_median"], found["low"] + 1e-12)
        self.assertLess(found["low"], found["high"])
        # the shared component lifts the whole band well above where single
        # documents sit, which is what a fixed band gets wrong
        self.assertGreater(found["low"], 0.4)

    def test_a_source_that_cannot_tell_them_apart_is_refused(self):
        with self.assertRaises(ValueError):
            band.measure(self.Topical(shared=400.0))

    def test_a_band_set_by_hand_is_kept(self):
        self.assertEqual(band.band_for(embed.LexicalEmbedder()), (0.05, 0.45))
        self.assertAlmostEqual(embed.to_model_band(0.25, (0.05, 0.45)), 0.25 + 0.5 * 0.6)


class Calibration(unittest.TestCase):
    @staticmethod
    def records(n, floor, kappa, seed):
        """Pairs whose chance of the outcome follows the model's own curve
        under a known floor and κ, with noise."""
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            r = {"ground": rng.random(), "strength": rng.uniform(0, 3),
                 "capacity_a": rng.random(), "capacity_b": rng.random()}
            p = 0.05 + 0.9 * calibrate.score(r, floor, kappa)
            r["outcome"] = {"revised": int(rng.random() < p), "continued": 1}
            out.append(r)
        return out

    def test_it_recovers_constants_it_was_given(self):
        result = calibrate.fit(self.records(8000, floor=0.30, kappa=2.0, seed=1), resamples=60)
        self.assertEqual(result["fitted"]["floor"], 0.30)
        self.assertGreaterEqual(result["fitted"]["kappa"], 1.0)
        self.assertGreater(result["gain"]["auc"], 0)
        self.assertTrue(result["adopt"])

    def test_it_does_not_move_constants_that_were_right(self):
        result = calibrate.fit(self.records(4000, floor=0.15, kappa=0.0, seed=2), resamples=60)
        self.assertLess(result["gain"]["auc"], 0.01)
        self.assertFalse(result["adopt"])

    def test_it_says_nothing_on_too_little(self):
        with self.assertRaises(calibrate.NotEnough):
            calibrate.fit(self.records(50, 0.15, 0.0, 3))
        with self.assertRaises(calibrate.NotEnough):
            calibrate.fit(self.records(300, 0.15, 0.0, 4), outcome="continued")   # everyone kept talking

    def test_auc(self):
        self.assertEqual(calibrate.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(calibrate.auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(calibrate.auc([0.1, 0.2], [1, 1]))


if __name__ == "__main__":
    unittest.main()
