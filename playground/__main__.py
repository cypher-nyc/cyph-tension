# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
    python -m playground                      the page, on a free local port
    python -m playground rank --examples --me ada
    python -m playground rank --person ada=~/vault --person ben=channel.json \\
        --relations relations.tsv --me ada [--topic "..."] [--json]
    python -m playground group --examples --members ada,ben,dev
    python -m playground band --embedder qwen      the band a source's cosines live in
    python -m playground calibrate outcomes.jsonl  fit the constants to what happened
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from playground import embed as embedding
from playground import band as banding
from playground import calibrate, importers, rank, server


def _band(value: str):
    lo, hi = (float(v) for v in value.split(","))
    if not lo < hi:
        raise argparse.ArgumentTypeError("the band is low,high")
    return lo, hi


def _add_vector_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedder", choices=("lexical", "qwen"), default="lexical")
    parser.add_argument("--lexical-band", type=_band, metavar="LOW,HIGH",
                        help="where lexical cosines live (default 0.05,0.45)")


def _load(args):
    people = {}
    relations_text = []
    if args.examples:
        folder = os.path.join(server.EXAMPLES, "people")
        for entry in sorted(os.listdir(folder)):
            if not entry.startswith("."):
                people[entry.split(".")[0]] = importers.load_path(os.path.join(folder, entry))
        with open(os.path.join(server.EXAMPLES, "relations.tsv"), encoding="utf-8") as handle:
            relations_text.append(handle.read())
    for spec in args.person or []:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--person takes name=path, not {spec!r}")
        people.setdefault(name, []).extend(importers.load_path(os.path.expanduser(path)))
    for path in args.relations or []:
        with open(os.path.expanduser(path), encoding="utf-8") as handle:
            relations_text.append(handle.read())
    return people, rank.parse_relations("\n".join(relations_text))


def _embedder(args):
    embedder = embedding.make(args.embedder)
    if args.lexical_band and args.embedder == "lexical":
        embedder.band = args.lexical_band
    return embedder


def _show(v):
    return "   -" if v is None else f"{v:.3f}"


def _rank(args) -> int:
    people, relations = _load(args)
    if args.me not in people:
        print(f"nobody called {args.me!r}; here: {', '.join(people) or 'no one'}", file=sys.stderr)
        return 2
    result = rank.rank(people, args.me, relations, args.topic, _embedder(args), args.kappa, args.rho)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"for {result['me']}   model: {result['backend']}   vectors: {result['embedder']}")
    print(f"{'':2}{'name':<12}{'lens':<13}{'tension':>8}{'ground':>8}{'diverge':>9}{'contested by':>14}{'contests':>10}")
    for row in result["ranked"]:
        print(f"{'':2}{row['name']:<12}{row['lens']:<13}{_show(row['score']):>8}{_show(row['intelligibility']):>8}"
              f"{_show(row['novelty']):>9}{_show(row['contested_by']):>14}{_show(row['contests']):>10}")
        for line in row["lines"]:
            a, b = (line["theirs"], line["yours"]) if line["toward"] == "me" else (line["yours"], line["theirs"])
            who = "theirs" if line["toward"] == "me" else "yours"
            mark = ", disputed" if line["disputed"] else ""
            print(f"{'':6}{a}  {line['axis']}  {b}  ({who}; {line['strength']:g}, {line['from']}{mark})")
    return 0


def _group(args) -> int:
    people, relations = _load(args)
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    missing = [m for m in members if m not in people]
    if missing:
        print(f"nobody called {', '.join(missing)}; here: {', '.join(people) or 'no one'}", file=sys.stderr)
        return 2
    result = rank.group(people, members, relations, args.topic, _embedder(args), args.kappa, args.rho)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{', '.join(result['members'])}   model: {result['backend']}   vectors: {result['embedder']}")
    print(f"  mean of pairs {_show(result['mean'])}   floor {_show(result['floor'])}   works contested {result['contested_works']}")
    for pair in result["pairs"]:
        print(f"{'':4}{pair['a']} and {pair['b']:<10}{pair['lens']:<13}{_show(pair['score'])}")
    for person in result["counterparts"]:
        print(f"{'':4}{person['name']}: " + (f"best met by {person['with']}" if person["with"] else "no counterpart here"))
    return 0


def _band(args) -> int:
    embedder = embedding.make(args.embedder)
    if embedder.band:
        print(f"{embedder.name}: set by hand at {embedder.band[0]:g} to {embedder.band[1]:g} (see playground/embed.py)")
        return 0
    found = banding.measure(embedder)
    print(f"{embedder.name}: {found['low']:.3f} to {found['high']:.3f}")
    print(f"  unrelated libraries: {found['unrelated_pairs']} pairs, median {found['unrelated_median']:.3f}, "
          f"{int(banding.LOW_PERCENTILE * 100)}th percentile {found['low']:.3f}")
    print(f"  the same reader, twice: {found['same_reader_splits']} splits, median {found['high']:.3f}")
    return 0


def _calibrate(args) -> int:
    with open(os.path.expanduser(args.outcomes), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    try:
        result = calibrate.fit(records, outcome=args.outcome)
    except calibrate.NotEnough as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m playground", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=0, help="default: any free port")
    _add_vector_flags(parser)
    commands = parser.add_subparsers(dest="command")

    def sources(sub):
        sub.add_argument("--person", action="append", metavar="NAME=PATH")
        sub.add_argument("--relations", action="append", metavar="FILE")
        sub.add_argument("--examples", action="store_true", help="the libraries in examples/")
        sub.add_argument("--topic")
        sub.add_argument("--json", action="store_true")
        sub.add_argument("--kappa", type=float, help="absorptive capacity (default: the model's, 0)")
        sub.add_argument("--rho", type=float, help="dependence of lines against one work (default: the model's, 0)")
        _add_vector_flags(sub)

    ranker = commands.add_parser("rank", help="rank everyone else for one person")
    ranker.add_argument("--me", required=True)
    sources(ranker)
    grouper = commands.add_parser("group", help="score a table of several people")
    grouper.add_argument("--members", required=True, metavar="A,B,C")
    sources(grouper)
    bander = commands.add_parser("band", help="measure where a source puts its cosines")
    _add_vector_flags(bander)
    fitter = commands.add_parser("calibrate", help="fit the constants to recorded outcomes")
    fitter.add_argument("outcomes", metavar="FILE.jsonl")
    fitter.add_argument("--outcome", choices=calibrate.OUTCOMES, default=calibrate.OUTCOMES[0])
    args = parser.parse_args(argv)
    handlers = {"rank": _rank, "group": _group, "band": _band, "calibrate": _calibrate}
    if args.command in handlers:
        return handlers[args.command](args)
    server.serve(args.port, args.embedder, args.lexical_band)
    return 0


if __name__ == "__main__":
    sys.exit(main())
