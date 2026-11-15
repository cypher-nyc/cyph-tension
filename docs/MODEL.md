
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
| `KAPPA_ABSORB` | 0 | how far capacity widens shared ground; zero until fitted |
| `LINE_DEPENDENCE` | 0 | how far lines against one work are one disagreement; zero until fitted |