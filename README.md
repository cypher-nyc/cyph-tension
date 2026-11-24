# tension

A small model of who should talk to whom.

Most matching puts similar people together. This one looks for **productive
tension**: two people with enough shared ground to understand each other, and
a recorded disagreement between the works that shaped them.

```
tension = gate(I) · I · N · (1 + λ·T)

  I   shared ground             falls as two readers grow apart
  N   evidenced divergence      rises as they grow apart
  T   movement toward a topic   a modest bonus, never a score of its own
```

Because `I` falls with distance and `N` rises with it, the product peaks in
the middle. Below a floor of shared ground the score is zero, not small:
opposition without common language entrenches people. Divergence is only ever
read from evidence (one work contests another; a reader wrote down that they
disagree), never from how far apart two vectors are.

The model is written once, in Haskell (`haskell/src/Cyph/Tension.hs` and
`Pair.hs`, no dependencies), compiled into a small C library. Python calls it
(`tension/`); there is no second copy of the model to drift from the first. It
was written for [cyph](https://cyph.city). It uses no language model.

The reasoning, the research behind each term, every constant and the open
questions are in [docs/MODEL.md](docs/MODEL.md). That document is where to
start if you want to contest the model.

## try it

Python 3.10 or later, and GHC (9.10, base only; [ghcup](https://www.haskell.org/ghcup/)
installs it) to build the model once.

```bash
make -C haskell lib
python3 -m playground
```

It prints a local address. Open it, choose **load the examples**, and click a
name to see everyone else ranked for that person. Then bring your own:

| Source | What to give it |
|---|---|
| Obsidian | the vault folder (drop it on the page, or `--person you=~/vault`) |
| Are.na | a channel's name or address (fetched from Are.na), or its exported `.json` |
| Browser bookmarks, Pocket, Pinboard, Raindrop | the exported bookmarks `.html` |
| Goodreads, Zotero, Readwise, any spreadsheet | a `.csv` with a title column |
| Anything else | a `.txt` or `.md` list, one work per line, or a `.json` list of `{title, description, url}` |

One person can bring several sources; add them under the same name.

Everything runs on your machine. The page listens on `127.0.0.1` only, keeps
the libraries in memory while it runs and writes nothing to disk. The one
outbound call is to Are.na, and only when you ask for a channel by name.

The same thing on the command line:

```bash
python3 -m playground rank --examples --me ada
python3 -m playground rank \
  --person ada=~/vault --person ben=channel.json --person cleo=bookmarks.html \
  --relations relations.tsv --me ada
```

```
for ada   model: haskell   vectors: lexical
  name        lens          tension  ground  diverge  contested by  contests
  dev         challenge       0.552   0.650    0.850         0.472     0.293
      Liberalism and the Limits of Justice  contests  A Theory of Justice  (theirs; 0.8, relations)
      Justice, Gender, and the Family  complicates  A Theory of Justice  (yours; 0.6, relations)
      After Virtue  complicates  A Theory of Justice  (theirs; 0.5, relations)
  ben         challenge       0.365   0.394    0.926         0.249     0.315
      Anarchy, State, and Utopia  contests  A Theory of Justice  (theirs; 1, relations)
      A Theory of Justice  contests  Anarchy, State, and Utopia  (yours; 1, recorded)
      Justice, Gender, and the Family  complicates  A Theory of Justice  (yours; 0.6, relations)
  cleo        challenge       0.097   0.215    0.451         0.000     0.097
      Justice, Gender, and the Family  complicates  A Theory of Justice  (yours; 0.6, relations)
  fay         ground only     0.000   0.472    0.000         0.000     0.000
  eze         below floor     0.000   0.000    0.000         0.000     0.000
```

### the four readings

| Lens | Meaning |
|---|---|
| challenge | shared ground, and a line of disagreement on record between your works |
| ground only | shared ground, nothing on record yet. The model will not invent a disagreement |
| below floor | not enough shared ground to score |
| unknown | nothing to read for this person |

### where divergence comes from

Two places, both of them evidence:

1. **A relations file.** One line per assessment: a work, then `contests`,
   `complicates` or `refutes`, then the work it stands against, then a
   strength, and optionally who says so and where. Several people can assess
   the same line; the median counts, and a wide spread marks the line
   disputed. See [examples/relations.tsv](examples/relations.tsv).
2. **Your own notes.** In an Obsidian note, `contests:: [[Another Note]]`
   (or the same key in frontmatter) records a position you took.

A line counts between two people when one of them holds the first work and
the other holds the second. Works are matched by title.

### where shared ground comes from

The cosine between two people's vectors, where a person's vector is the mean
of their items' vectors. The playground has two ways to make vectors:

- `lexical` (default): word statistics only. Nothing to install, and good
  enough to see how the model behaves. It knows nothing about meaning, so
  treat its numbers as a demonstration.
- `--embedder qwen`: Qwen3-Embedding-0.6B, an open-weights model, on your own
  machine (`pip install sentence-transformers`, then a download of about
  1.2 GB on first use).

Each source puts its cosines somewhere different, so the range that counts
as "no shared ground" to "the same reader" is measured for the embedding
model on a set of reference shelves (`python3 -m playground band --embedder
qwen`), not assumed. See "The band" in [docs/MODEL.md](docs/MODEL.md).

### more than a ranking

```bash
python3 -m playground group --examples --members ada,ben,eze   # a table: its mean, its floor, who is left out
python3 -m playground rank --examples --me ada --kappa 1.5     # let capacity widen shared ground
python3 -m playground calibrate outcomes.jsonl                 # fit the constants to what happened
```

Each row of a ranking carries two directed scores beside the mutual one:
how far that person contests you, and how far you contest them.

## build and test

```bash
make -C haskell lib                       # the compiled model, which Python calls
make -C haskell test                      # the model's claims as laws (GHC 9.10, base only)
python3 -m unittest discover -s tests     # every output to the bit, the importers, the ranking
```

A service calls the model through its Python face: `pip install
"cyph-tension @ git+https://github.com/cypher-nyc/cyph-tension@<commit>"` (or
`pip install -e .` in a checkout) gives `import tension`; the library itself
is built with `make -C haskell lib` and found through `CYPH_TENSION_LIB`.
`tension.work_tension` scores a person against a work they have not met
(`docs/MODEL.md`, "What to put in front of someone").

`tests/golden/tension.json` holds what the model returns on 9,901 inputs, to
the bit; a change that moves any of them fails. When the model is meant to
change, regenerate it (`python3 -m tests.golden.make_golden`) and say why.

## what is here

| Path | What it is |
|---|---|
| `haskell/src/Cyph/Tension.hs` | the model |
| `haskell/src/Cyph/TensionFFI.hs` | the model as a C library, for any language |
| `haskell/test/` | the model's laws and facts, each law checked over thousands of generated inputs |
| `tension/` | the model's Python face: its constants and signatures, each a call into the library |
| `playground/` | importers, the gathering of inputs, the page and the command line |
| `examples/` | six small invented libraries, a relations file, and the reference shelves the band is measured on |
| `docs/MODEL.md` | the model in writing |

## working on the model together

The model is a set of claims about people, and each one can be wrong. The
most useful contributions are not code:

- a case where the ranking is plainly wrong, with the libraries that show it;
- evidence for or against a constant (the floor, the weights, the bonus);
- a better account of what counts as a recorded disagreement;
- relations between works in a field you know.

Open an issue with your case. A change to the model itself goes in
`haskell/src/Cyph/`, and the property tests have to hold.

## licence

[Mozilla Public License 2.0](LICENSE). You can use the model inside anything,
open or closed. If you change these files and distribute the result, the
changed files stay open under the same licence.
