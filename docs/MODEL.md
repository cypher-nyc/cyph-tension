
Tension is not equal to disagreement. Tension is the potential for
two people, or a person and a topic, to change each other's minds.

```
tension = gate(I) · I · N · (1 + λ·T)
```

| Term | Name | What it measures | Moves with distance |
|---|---|---|---|
| `I` | intelligibility | shared ground | falls |
| `N` | novelty | divergence, from evidence | rises |
| `T` | trajectory | recent movement toward the topic | (a bonus) |
| `gate` | the floor | `0` when `I < I_FLOOR`, else `1` | |

All inputs and outputs are in `[0, 1]`. 

## Inspiration

Five bodies of research point at the same structure. Useful exchange peaks at
a middle distance, and the curve is the product of two opposing forces, not
one similarity score.

1. **Optimal cognitive distance.** Nooteboom, Vanhaverbeke, Duysters, Gilsing
   & van den Oord (2007), *Research Policy* 36(7). Innovation between
   partners is an inverted U in cognitive distance, and it decomposes into
   absorptive capacity (falls with distance) times novelty (rises with it).
   This gives the model its form.

2. **The Goldilocks effect and learning progress.** Kidd, Piantadosi & Aslin
   (2012), *PLOS ONE* 7(5); Ten, Kaushik, Oudeyer & Gottlieb (2021), *Nature
   Communications* 12. Attention peaks at intermediate predictability, and
   people explore by learning progress more than by raw novelty. A person
   moving toward a topic outranks a static expert of equal depth.

3. **Authentic dissent.** Nemeth, Brown & Rogers (2001), *European Journal
   of Social Psychology* 31(6). Positions people really hold stimulate
   better thinking. Role-played devil's advocacy does not. Novelty is computed from evidence of a real
   divergent path and **never from vector distance**.

4. **Diversity under a shared language.** Hong & Page (2004), *PNAS*
   101(46). Diverse groups can outperform high-ability homogeneous ones,
   given a shared problem language. This is why a caller should surface
   complementary people, not several copies of the top scorer.

5. **Backfire.** Bail et al. (2018), *PNAS* 115(37). Exposure to opposing
   views without shared ground increased polarisation. 

## Who to bring into a room

```
I = clamp( W_VEC · band(cos(person, topic)) + W_GRAPH · depth )
N = clamp( READING_W · divergent_reading + STANCE_W · stance )
T = clamp( (cos_recent − cos_older) / T_MAX )      zero when moving away
tension     = gate(I) · I · N · (1 + λ·T)
suitedness  = gate(I) · I · (1 + λ·T)              who can carry the thread
```

- `depth`: how much of the topic's neighbourhood the person has engaged,
  against the rest of the pool.
- `divergent_reading`: the share of what they read around the topic that
  sits outside its canonical core.
- `stance`: positions they recorded (contests, complicates, refutes) in this
  neighbourhood, against the pool.
- `band(c)` maps a raw cosine from `[COS_LO, COS_HI]` onto `[0, 1]`.

Tension finds who will push back. Suitedness finds who can champion the
conversation. We score both.

## What to put in front of someone

A work (a book, a cyph, a route) the person has not met yet, scored against
everything that shaped them.

```
I = clamp( W_VEC · band(cos(person, work)) + W_GRAPH · depth )
N = 1 − e^(−s)
tension = gate(I) · I · N
```

- `I` is the person-against-a-topic ground with the work as the anchor:
  `depth` is the person's share of activity in the work's neighbourhood.
- `s` is the summed strength of the lines on the dissonance axis between the
  work and the works that shaped the person, directly or through the graph
  (weighted by hops), as between two people.
- A work the person shares no ground with scores zero however strongly it is
  contested; a work nothing contests scores zero however close it sits.
- No trajectory term: it ranks what to offer, not where someone is heading.

## Who sits together

```
I = band(cos(a, b))                                          with no topic
I = clamp( W_PEOPLE · band(cos(a, b)) + W_TOPIC · band(min(cos(a, t), cos(b, t))) )
N = 1 − e^(−s)
tension = gate(I) · I · N
```

- `s` is the summed strength of the lines on the dissonance axis between a
  work that shaped one person and a work that shaped the other.
- The topic half takes the lower of the two cosines: both people have to
  speak the topic.
- No trajectory term. We measure movement toward a topic, which is
  a different question from who should share a table.
- A group's score is the mean of the pairs that could be read; its floor
  is the least well-met person's best pair.

### The pair score, hardened (`Cyph.Pair`)

The same two opposing terms, with shared ground read against a measured
distribution, a smooth gate, and each cypher's library counted.

```
n       = works explored, saved, annotated + cyphs attended, online and in person
c       = min(c_a, c_b),   c_x = min(1, log(1 + n_x) / log(1 + n_ref))
I       = percentile of cos(a, b) among random pairs, per embedding model
I'      = 1 − (1 − I)^(1 + κ·c)
g       = σ((I' − θ) / τ)
w(h)    = reach^(h − 1) · (1 − novelty^h), scaled so its peak is 1; 0 past H hops
s       = Σ over contest paths p of strength_p · w(h_p)
N       = 1 − e^(−s / s₀)
tension = g · I' · N
table   = mean of its pairs, and mean − λ·σ
```

- A library is what a person explored, saved and annotated, and the cyphs
  they attended. A cyph held in person is a cyph in its own right and counts
  exactly as one held online: one toward `n`, and its topic and works in the
  vector shared ground is read from.
- A cypher with fewer than `MIN_LIBRARY` items is not scored.
- A contest path is a path through the graph with an odd number of edges on
  the dissonance axis (contests, complicates, refutes, opposes). An even
  number is agreement and adds nothing to `s`. Walks stop at `H` hops.
- `c` (engagement depth) is how deep the lighter cypher goes, with diminishing
  returns, capped at 1. With `κ > 0`, deeply engaged pairs find footing with less
  overlap (Nooteboom's third hypothesis: the best distance rises with what
  one already knows).

| Constant | Value | How it was set |
|---|---|---|
| `θ` (`I_FLOOR`) | 0.15 | Bail et al.; not fitted |
| `τ` | 0.02 | the gate's width; a starting point |
| `κ` (`KAPPA_ABSORB`) | 1.0 | a rule, not a fit: the largest value at which the most engaged pair's floor stays at or above half of `θ` |
| `s₀` | 1.0 | a starting point |
| `n_ref` | 500 | a starting point; to be set from measured library sizes |
| `λ` | 1.0 | a starting point |
| `H`, reach, novelty | 4, 0.7, 0.6 | starting points; the weight peaks at two hops |
| `MIN_LIBRARY` | 5 | a starting point |

How κ was chosen. With no recorded outcomes to fit against, six values were
run through the model on the illustrative curve (`I = 1 − d`, `N = d`) and
on 4,000 generated pairs:

| κ | best distance, c = 1 | most ground added | raw floor, c = 1 | top tenth unchanged |
|---|---|---|---|---|
| 0 | 0.50 | 0 | 0.150 | 100% |
| 0.5 | 0.54 | 0.15 | 0.103 | 93% |
| 1 | 0.58 | 0.25 | 0.078 | 88% |
| 1.5 | 0.61 | 0.33 | 0.063 | 84% |
| 2 | 0.63 | 0.39 | 0.053 | 82% |
| 3 | 0.67 | 0.47 | 0.040 | 77% |

At 1 the best distance moves outward for deeply engaged pairs while the floor
that guards against backfire still shuts out the bottom 7.8% of pairs. Past
it, engagement depth alone lets pairs under half the floor. `playground
calibrate` replaces this with a fit once 200 seated pairs have outcomes.

## The constants

These are the tuning surface. The shape of the curve is held by the property
tests, so changing a constant cannot quietly deform it.

| Constant | Value | Why |
|---|---|---|
| `I_FLOOR` | 0.15 | below this, no score at all (Bail et al.) |
| `COS_LO`, `COS_HI` | 0.25, 0.85 | the band real document cosines sat in on bge-large-en-v1.5 |
| `W_VEC`, `W_GRAPH` | 0.5, 0.5 | the vector alone is fooled by nearby territory a person never touched; engagement alone misses relevant reading elsewhere |
| `READING_W`, `STANCE_W` | 0.6, 0.4 | reading divergence is common and weak; a recorded stance is rare and strong |
| `LAMBDA_TRAJECTORY` | 0.35 | modest on purpose: trajectory modulates a score and never creates one |
| `T_MAX` | 0.15 | cosine movement toward a topic at which the bonus saturates |
| `W_PEOPLE`, `W_TOPIC` | 0.5, 0.5 | ground from the two people against ground from the topic |
| `KAPPA_ABSORB` | 1.0 | how far engagement depth widens shared ground; set by a rule (see the pair score), not fitted |
| `LINE_DEPENDENCE` | 0 | how far lines against one work are one disagreement; zero until fitted |