-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0. If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.

-- | The productive-tension model.
--
-- > tension = gate(I) · I · N · (1 + λ·T)
--
-- Shared ground (I) falls as two readers grow apart while evidenced
-- divergence (N) rises, so their product peaks at a moderate distance; below
-- a floor of shared ground the score is zero. The reasoning and the research
-- behind each term are in @docs/MODEL.md@; this module states the model and
-- nothing else.
--
-- Every function is total and pure: scalars in, a scalar in [0, 1] out.
-- This is the only implementation: Python calls it through the C library
-- (@tension/@), and @tests/test_golden.py@ pins its outputs to the bit. The
-- order of operations below was set to match the retired Python reference
-- exactly, which is why some comments still name Python's behaviour.
module Cyph.Tension
  ( -- * Constants (the tuning surface; Python reads these names too)
    iFloor
  , cosLo
  , cosHi
  , wVec
  , wGraph
  , readingW
  , stanceW
  , lambdaTrajectory
  , tMax
  , wPeople
  , wTopic
    -- * A person against a topic (who to bring into a room)
  , clamp01
  , normalizeCosine
  , computeIntelligibility
  , computeNovelty
  , computeTrajectory
  , computeTension
  , computeSuitedness
    -- * A person against a person (who should sit together)
  , pairIntelligibility
  , pairNovelty
  , pairTension
  , carMean
    -- * Absorptive capacity (the best distance rises with what one already knows)
  , kappaAbsorb
  , absorbWith
  , absorb
    -- * Lines of disagreement that are not independent
  , lineDependence
  , lineStrengthWith
  , lineStrength
    -- * Groups (nobody left without a counterpart)
  , groupFloor
    -- * Contests reached through the graph (the inverted U, in hops)
  , hopReach
  , hopNovelty
  , maxHops
  , hopWeightWith
  , hopWeight
  , pathContestWith
  , pathContest
  ) where

-- | Mutual-intelligibility floor: below it the score is zero, not small
-- (Bail et al. 2018).
iFloor :: Double
iFloor = 0.15

-- | The band cosines between real document embeddings live in; mapped to
-- [0, 1].
cosLo, cosHi :: Double
cosLo = 0.25
cosHi = 0.85

-- | Intelligibility: half the vector reading, half the graph reading.
wVec, wGraph :: Double
wVec = 0.5
wGraph = 0.5

-- | Novelty: reading divergence (common, weak) and recorded stance (rare,
-- strong).
readingW, stanceW :: Double
readingW = 0.6
stanceW = 0.4

-- | Learning-progress bonus: it modulates a score, it never creates one.
lambdaTrajectory :: Double
lambdaTrajectory = 0.35

-- | Cosine movement toward a topic that saturates the trajectory bonus.
tMax :: Double
tMax = 0.15

-- | Pairs: shared ground from the two people versus from the topic.
wPeople, wTopic :: Double
wPeople = 0.5
wTopic = 0.5

-- | Clamp to [0, 1]. NaN clamps to 1, exactly as Python's
-- @max(0.0, min(1.0, nan))@ does (Python's min/max keep the first argument
-- unless the next compares strictly smaller/larger, and NaN never does).
clamp01 :: Double -> Double
clamp01 v = max0 (min1 v)
  where
    min1 x = if x < 1.0 then x else 1.0
    max0 x = if x > 0.0 then x else 0.0

-- | A raw embedding cosine onto [0, 1] across the band real similarities
-- live in.
normalizeCosine :: Double -> Double
normalizeCosine c
  | cosHi <= cosLo = clamp01 c
  | otherwise = clamp01 ((c - cosLo) / (cosHi - cosLo))

-- | Shared ground with a topic: the term that falls with epistemic distance.
computeIntelligibility :: Double -> Double -> Double
computeIntelligibility cosToAnchor depthNorm =
  clamp01 (wVec * normalizeCosine cosToAnchor + wGraph * clamp01 depthNorm)

-- | Evidenced divergence within the topic: the term that rises with
-- distance. Behavioural inputs only, never vector distance (Nemeth 2001).
computeNovelty :: Double -> Double -> Double
computeNovelty divergentReadingRatio stanceRatio =
  clamp01 (readingW * clamp01 divergentReadingRatio + stanceW * clamp01 stanceRatio)

-- | Learning progress toward the topic; drift away is not penalised.
computeTrajectory :: Double -> Double -> Double
computeTrajectory cosRecent cosOlder
  -- the same test as Python's @delta <= 0@, NaN falling through as there
  | delta <= 0 || tMax <= 0 = 0.0
  | otherwise = clamp01 (delta / tMax)
  where
    delta = cosRecent - cosOlder

-- | Productive tension: the inverted-U is the product of two opposing terms,
-- gated to hard zero below the intelligibility floor.
computeTension :: Double -> Double -> Double -> Double
computeTension intelligibility novelty trajectory
  | i < iFloor = 0.0
  | otherwise = clamp01 (i * n * bonus)
  where
    i = clamp01 intelligibility
    n = clamp01 novelty
    bonus = 1.0 + lambdaTrajectory * clamp01 trajectory

-- | Who can carry the thread: depth plus movement, no divergence reward.
computeSuitedness :: Double -> Double -> Double
computeSuitedness intelligibility trajectory
  | i < iFloor = 0.0
  | otherwise = clamp01 (i * (1.0 + lambdaTrajectory * clamp01 trajectory))
  where
    i = clamp01 intelligibility

-- | Shared ground between two people inside a topic. 'Nothing' when there
-- is no topic: the people half alone.
pairIntelligibility :: Double -> Maybe Double -> Double
pairIntelligibility cosPeople cosTopic =
  case cosTopic of
    Nothing -> people
    Just t -> clamp01 (wPeople * people + wTopic * normalizeCosine t)
  where
    people = normalizeCosine cosPeople

-- | Divergence from recorded positions, saturating as 1 − e^(−s).
pairNovelty :: Double -> Double
pairNovelty contestStrength = clamp01 (1.0 - exp (negate strength))
  where
    strength = if contestStrength > 0.0 then contestStrength else 0.0

-- | The same curve with no trajectory bonus.
pairTension :: Double -> Double -> Double
pairTension intelligibility novelty = computeTension intelligibility novelty 0.0

-- | The mean of the pair scores we could read; 'Nothing' when none. (The
-- Python wrapper rounds it to 4 places.)
carMean :: [Double] -> Maybe Double
carMean [] = Nothing
carMean xs = Just (sum' xs / fromIntegral (length xs))
  where
    -- left-to-right, as Python's sum() adds
    sum' = foldl (+) 0.0

-- | How strongly accumulated knowledge widens what a person can take in.
-- Zero until it is fitted against outcomes: at zero 'absorb' changes
-- nothing, and the model scores everyone on one curve.
kappaAbsorb :: Double
kappaAbsorb = 0.0

-- | Shared ground as a person of capacity @c@ experiences it (Nooteboom's
-- third hypothesis): @1 − (1 − I)^(1 + κ·c)@. It never lowers shared
-- ground, it leaves 0 and 1 where they are, and with @I = 1 − d@ against
-- @N = d@ the peak of the product sits at @d* = (2 + κc)^(−1/(1 + κc))@,
-- which moves outward as capacity grows. A negative or unreadable κ is no
-- adjustment.
absorbWith :: Double -> Double -> Double -> Double
absorbWith kappa intelligibility capacity =
  clamp01 (1.0 - (1.0 - clamp01 intelligibility) ** (1.0 + k * clamp01 capacity))
  where
    k = if kappa > 0.0 then kappa else 0.0

-- | 'absorbWith' at the model's own κ.
absorb :: Double -> Double -> Double
absorb = absorbWith kappaAbsorb

-- | How far several lines against the same work are one disagreement seen
-- from several sides (1) or separate disagreements (0). Zero until fitted:
-- every line counts in full, as it always has.
lineDependence :: Double
lineDependence = 0.0

-- | The strength of all the lines that stand against one work:
-- @max + (1 − ρ)·(sum − max)@. At ρ = 0 the plain sum; at ρ = 1 only the
-- strongest line counts. Strengths below zero (and unreadable ones) count
-- as none.
lineStrengthWith :: Double -> [Double] -> Double
lineStrengthWith rho strengths = strongest + (1.0 - clamp01 rho) * (total - strongest)
  where
    kept = [if x > 0.0 then x else 0.0 | x <- strengths]
    -- left-to-right, as the Python adds and compares
    total = foldl (+) 0.0 kept
    strongest = foldl (\m x -> if x > m then x else m) 0.0 kept

-- | 'lineStrengthWith' at the model's own ρ.
lineStrength :: [Double] -> Double
lineStrength = lineStrengthWith lineDependence

-- | The least well-met person in a group. For each member, the best pair
-- score they have with anyone else in it; the group's floor is the lowest
-- of those. A group can have a fine mean and still hold someone with no
-- counterpart at all: the mean dips, the floor goes to zero. Rows are
-- members, @Nothing@ a pair that could not be read (and the diagonal);
-- members with no readable pair are left out, and a group with none has no
-- floor.
groupFloor :: [[Maybe Double]] -> Maybe Double
groupFloor rows =
  case [b | Just b <- zipWith best [0 :: Int ..] rows] of
    [] -> Nothing
    (b : bs) -> Just (foldl (\m x -> if x < m then x else m) b bs)
  where
    best i row =
      case [x | (j, Just x) <- zip [0 :: Int ..] row, j /= i] of
        [] -> Nothing
        (x : xs) -> Just (foldl (\m y -> if y > m then y else m) x xs)

-- | How much of a contest survives each further hop between two works:
-- what a reader can still take in falls with distance. A starting point,
-- not fitted.
hopReach :: Double
hopReach = 0.7

-- | How fast what is new to a reader builds up with hops: a contest
-- between neighbouring works is the least new. At 0 every distance is as
-- new as any other and only reach is left. A starting point, not fitted.
hopNovelty :: Double
hopNovelty = 0.6

-- | The hop budget: a path longer than this is not evidence.
maxHops :: Int
maxHops = 4

-- | The weight of a contest by how many hops of the graph it is reached
-- through: @reach^(h − 1) · (1 − novelty^h)@, divided by its largest value
-- within the budget, so the best distance weighs 1. Reach falls with hops
-- and novelty rises, so their product rises to one peak and falls from it
-- (Nooteboom's two opposing terms, measured on the graph instead of
-- assumed): the inverted U is in the formula. Outside @1..limit@ the
-- weight is 0. With @novelty = 0@ it is plain decay from a direct line,
-- and with @limit = 1@ only direct lines count, at full weight: the model
-- as it was before paths.
hopWeightWith :: Double -> Double -> Int -> Int -> Double
hopWeightWith reach novelty limit hops
  | hops < 1 || hops > budget = 0.0
  | peak > 0.0 = clamp01 (raw hops / peak)
  | otherwise = 0.0
  where
    budget = if limit < 64 then limit else 64
    r = clamp01 reach
    v = clamp01 novelty
    raw :: Int -> Double
    raw k = r ** fromIntegral (k - 1) * (1.0 - v ** fromIntegral k)
    -- left-to-right over the budget, as the Python does
    peak = foldl (\m k -> let x = raw k in if x > m then x else m) 0.0 [1 .. budget]

-- | 'hopWeightWith' at the model's own reach, novelty and budget.
hopWeight :: Int -> Double
hopWeight = hopWeightWith hopReach hopNovelty maxHops

-- | What one path between two works adds to the strength of the contest
-- between their readers. A path is a contest when an odd number of its
-- edges are on the dissonance axis: a work that builds on what another
-- refutes stands against it, while two works that refute the same third
-- stand together (structural balance; Cartwright & Harary 1956). Its
-- strength (the product of its edges', from the caller) is held to [0, 1],
-- unreadable counting as none, and weighed by its length.
pathContestWith :: Double -> Double -> Int -> Int -> Int -> Double -> Double
pathContestWith reach novelty limit hops dissonant strength
  | dissonant < 1 || even dissonant = 0.0
  | otherwise = s * hopWeightWith reach novelty limit hops
  where
    s = if strength > 0.0 then (if strength < 1.0 then strength else 1.0) else 0.0

-- | 'pathContestWith' at the model's own reach, novelty and budget.
pathContest :: Int -> Int -> Double -> Double
pathContest = pathContestWith hopReach hopNovelty maxHops
