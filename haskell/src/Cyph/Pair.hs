-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0. If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.

-- | The pair and table score, hardened.
--
-- > c       = min c_a c_b,        c_x = min 1 (log (1 + n_x) / log (1 + nRef))
-- > I       = percentile of cos(a, b) among random pairs, for the embedding model
-- > I'      = 1 − (1 − I)^(1 + κ·c)
-- > g       = sigmoid ((I' − floorI) / τ)
-- > N       = 1 − e^(−s / s0),  s = Σ over the contest paths between the two
-- >                             libraries of weight · strength · w(hops),
-- >                             each work's total capped
-- > w(h)    = reach^(h − 1) · (1 − novelty^h), scaled so its peak is 1
-- > tension = g · I' · N                  (÷ T*(k) when normalizePeak)
--
-- The evidence is paths, not only direct lines: the graph is traversed
-- between the two libraries (by the caller: this module is pure and scores
-- the snapshot it is given), so two readers with no work in common and no
-- line between their works are still read, through what their works build
-- on and stand against. A path is a contest on an odd number of dissonant
-- edges, and its weight peaks at a middle number of hops
-- ('Cyph.Tension.hopWeightWith'): the inverted U is in the formula. A
-- direct line is a path of one hop.
--
-- Against "Cyph.Tension"'s 'Cyph.Tension.pairTension' this replaces the hard
-- floor with a smooth gate (no jump at the floor), reads shared ground as a
-- percentile against a measured random-pair distribution instead of a fixed
-- cosine band, weights and caps the lines between works, and refuses to
-- score a reader whose library is too small to say anything. A table reports
-- its mean and a lower bound (mean − λ·σ) instead of its minimum.
--
-- "Cyph.Tension" is untouched: the FFI still serves the original functions.
module Cyph.Pair
  ( -- * Values in [0, 1]
    UnitInterval
  , unitInterval
  , clampUnit
  , unUnit
    -- * Configuration
  , Config (..)
  , TypeWeights (..)
  , defaultConfig
  , validateConfig
    -- * Shared ground, calibrated per embedding model
  , Calibration
  , calibration
  , calModelId
  , Calibrations
  , calibrations
  , calibrationFor
  , percentile
    -- * Libraries and the lines between their works
  , WorkId
  , Library
  , library
  , librarySize
  , LinkType (..)
  , Link (..)
  , Hop (..)
  , Path (..)
  , linkPath
    -- * The terms
  , readerCapacity
  , pairCapacity
  , absorbed
  , gate
  , peakHeight
  , pathContestStrength
  , pathKinStrength
  , contestStrength
  , kinStrength
  , divergenceStrength
  , divergence
  , ground
  , tensionFrom
    -- * Scores
  , ScoreError (..)
  , scorePaths
  , scorePair
  , TableScore (..)
  , scoreTable
  ) where

import Cyph.Tension (hopNovelty, hopReach, hopWeightWith, iFloor, kappaAbsorb, maxHops, pathContestWith)
import Data.List (group, sort, sortBy)
import Data.Ord (comparing)

-- ─── values in [0, 1] ──────────────────────────────────────────────────

-- | A number known to lie in [0, 1]. Built only through 'unitInterval'
-- (which refuses anything else) or 'clampUnit' (which pulls it in).
newtype UnitInterval = UnitInterval Double
  deriving (Eq, Ord, Show)

-- | 'Just' when the value is in [0, 1]; NaN is not.
unitInterval :: Double -> Maybe UnitInterval
unitInterval x
  | x >= 0 && x <= 1 = Just (UnitInterval x)
  | otherwise = Nothing

-- | Clamp into [0, 1]. NaN reads as 0: an unreadable value scores nothing.
clampUnit :: Double -> UnitInterval
clampUnit x
  | isNaN x = UnitInterval 0
  | x < 0 = UnitInterval 0
  | x > 1 = UnitInterval 1
  | otherwise = UnitInterval x

unUnit :: UnitInterval -> Double
unUnit (UnitInterval x) = x

-- ─── configuration ─────────────────────────────────────────────────────

-- | How much each kind of line counts.
data TypeWeights = TypeWeights
  { wContests :: Double
  , wComplicates :: Double
  , wRefutes :: Double
  }
  deriving (Eq, Show)

data Config = Config
  { kappa :: Double
  -- ^ how far capacity widens shared ground (κ)
  , floorI :: Double
  -- ^ where the gate is half open
  , tau :: Double
  -- ^ the gate's width: smaller is sharper
  , s0 :: Double
  -- ^ line strength at which divergence reaches 1 − 1/e
  , nRef :: Int
  -- ^ library size at which capacity saturates
  , lambda :: Double
  -- ^ how many standard deviations a table's lower bound sits under its mean
  , normalizePeak :: Bool
  -- ^ divide by the curve's own peak height, so every capacity peaks at 1
  , minLibrarySize :: Int
  -- ^ below this many works a reader is not scored
  , embeddingModelId :: String
  -- ^ which calibration reads the cosines
  , typeWeights :: TypeWeights
  , workCap :: Double
  -- ^ the most the paths standing against any one work can add to @s@
  , reach :: Double
  -- ^ how much of a contest survives each further hop
  , noveltyRate :: Double
  -- ^ how fast what is new builds up with hops (0: only reach is left)
  , hopBudget :: Int
  -- ^ a path longer than this is not evidence (1: direct lines only)
  , graphGround :: Double
  -- ^ how much of shared ground is read from kinship paths in the graph,
  -- against the embedding percentile (0: the percentile alone)
  }
  deriving (Eq, Show)

-- | The model's defaults for an embedding model. κ and @floorI@ are the
-- values "Cyph.Tension" already holds ('kappaAbsorb', 'iFloor'); @s0 = 1@
-- keeps @N = 1 − e^(−s)@. The rest (@nRef@, @lambda@, @minLibrarySize@,
-- @workCap@, equal type weights) are starting points, not fitted; so are
-- the hop terms, which are "Cyph.Tension"'s ('hopReach', 'hopNovelty',
-- 'maxHops'). @graphGround@ is 0 until fitted: shared ground stays the
-- percentile.
defaultConfig :: String -> Config
defaultConfig modelId =
  Config
    { kappa = kappaAbsorb
    , floorI = iFloor
    , tau = 0.02
    , s0 = 1.0
    , nRef = 500
    , lambda = 1.0
    , normalizePeak = False
    , minLibrarySize = 5
    , embeddingModelId = modelId
    , typeWeights = TypeWeights 1.0 1.0 1.0
    , workCap = 1.0
    , reach = hopReach
    , noveltyRate = hopNovelty
    , hopBudget = maxHops
    , graphGround = 0.0
    }

-- | Every field in its range, or the first one that is not.
validateConfig :: Config -> Either ScoreError Config
validateConfig cfg
  | not (finite (kappa cfg) && kappa cfg >= 0) = bad "kappa must be finite and >= 0"
  | not (floorI cfg > 0 && floorI cfg < 1) = bad "floorI must be in (0, 1)"
  | not (finite (tau cfg) && tau cfg > 0) = bad "tau must be finite and > 0"
  | not (finite (s0 cfg) && s0 cfg > 0) = bad "s0 must be finite and > 0"
  | nRef cfg < 1 = bad "nRef must be >= 1"
  | not (finite (lambda cfg) && lambda cfg >= 0) = bad "lambda must be finite and >= 0"
  | minLibrarySize cfg < 0 = bad "minLibrarySize must be >= 0"
  | not (finite (workCap cfg) && workCap cfg > 0) = bad "workCap must be finite and > 0"
  | not (all (\w -> finite w && w >= 0) [wContests tw, wComplicates tw, wRefutes tw]) =
      bad "type weights must be finite and >= 0"
  | not (reach cfg >= 0 && reach cfg <= 1) = bad "reach must be in [0, 1]"
  | not (noveltyRate cfg >= 0 && noveltyRate cfg < 1) = bad "noveltyRate must be in [0, 1)"
  | hopBudget cfg < 1 = bad "hopBudget must be >= 1"
  | not (graphGround cfg >= 0 && graphGround cfg <= 1) = bad "graphGround must be in [0, 1]"
  | otherwise = Right cfg
  where
    tw = typeWeights cfg
    bad = Left . InvalidConfig

finite :: Double -> Bool
finite x = not (isNaN x || isInfinite x)

-- ─── calibration ───────────────────────────────────────────────────────

-- | Cosines between random pairs of libraries, measured for one embedding
-- model: the distribution a pair's cosine is read against.
data Calibration = Calibration
  { calModelId :: String
  , calSorted :: [Double]
  }
  deriving (Eq, Show)

-- | A calibration from a sample of random-pair cosines. Needs at least two
-- distinct finite values.
calibration :: String -> [Double] -> Either ScoreError Calibration
calibration modelId sample
  | length (group xs) < 2 = Left (InvalidCalibration modelId "needs at least two distinct finite cosines")
  | otherwise = Right (Calibration modelId xs)
  where
    xs = sort (filter finite sample)

-- | One calibration per embedding model.
newtype Calibrations = Calibrations [Calibration]
  deriving (Eq, Show)

calibrations :: [Calibration] -> Either ScoreError Calibrations
calibrations cs =
  case [m | (m : _ : _) <- group (sort (map calModelId cs))] of
    [] -> Right (Calibrations cs)
    (m : _) -> Left (InvalidCalibration m "more than one calibration for this model")

-- | The calibration for the config's embedding model; an error when there
-- is none, never a silent fallback.
calibrationFor :: Config -> Calibrations -> Either ScoreError Calibration
calibrationFor cfg (Calibrations cs) =
  case [c | c <- cs, calModelId c == embeddingModelId cfg] of
    (c : _) -> Right c
    [] -> Left (NoCalibration (embeddingModelId cfg))

-- | Where a cosine falls among the random pairs: 0 at or below the lowest,
-- 1 at or above the highest, linear between neighbouring samples (so it is
-- continuous and non-decreasing in the cosine wherever the sample has no
-- ties).
percentile :: Calibration -> Double -> UnitInterval
percentile (Calibration _ xs) x = clampUnit (go 0 xs)
  where
    m = fromIntegral (length xs - 1) :: Double
    go :: Int -> [Double] -> Double
    go i (a : rest@(b : _))
      | i == 0 && x <= a = 0
      | x < b = (fromIntegral i + (x - a) / (b - a)) / m
      | otherwise = go (i + 1) rest
    go _ _ = 1

-- ─── libraries and lines ───────────────────────────────────────────────

type WorkId = String

-- | The distinct works a reader has.
newtype Library = Library [WorkId]
  deriving (Eq, Show)

library :: [WorkId] -> Library
library ws = Library [w | (w : _) <- group (sort ws)]

librarySize :: Library -> Int
librarySize (Library ws) = length ws

data LinkType = Contests | Complicates | Refutes
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | A line from one work to another (@linkFrom@ stands against @linkTo@),
-- with how sure its assessment is.
data Link = Link
  { linkFrom :: WorkId
  , linkTo :: WorkId
  , linkType :: LinkType
  , linkConfidence :: Double
  }
  deriving (Eq, Ord, Show)

-- | One edge of a path through the graph: on the dissonance axis (with its
-- kind) or on the lineage axis (builds on, extends, echoes…), each with
-- how sure it is.
data Hop
  = Contest LinkType Double
  | Kin Double
  deriving (Eq, Ord, Show)

-- | A path through the graph from one work to another: what a traversal
-- between two libraries returns. The works in between are the graph's
-- business; the score needs the two ends and the edges.
data Path = Path
  { pathFrom :: WorkId
  , pathTo :: WorkId
  , pathHops :: [Hop]
  }
  deriving (Eq, Ord, Show)

-- | A direct line is a path of one hop.
linkPath :: Link -> Path
linkPath l = Path (linkFrom l) (linkTo l) [Contest (linkType l) (linkConfidence l)]

-- ─── the terms ─────────────────────────────────────────────────────────

-- | How much one reader has to draw on: @min 1 (log (1 + n) / log (1 + nRef))@.
readerCapacity :: Config -> Int -> Double
readerCapacity cfg n =
  min 1 (log (1 + fromIntegral (max 0 n)) / log (1 + fromIntegral (max 1 (nRef cfg))))

-- | A pair can use only as much as its lesser reader brings.
pairCapacity :: Config -> Int -> Int -> Double
pairCapacity cfg na nb = min (readerCapacity cfg na) (readerCapacity cfg nb)

-- | Shared ground as the pair can use it: @1 − (1 − I)^(1 + κ·c)@. At
-- @κ·c = 0@ it returns @I@ itself (not @1 − (1 − I)@, which rounds).
absorbed :: Config -> Double -> UnitInterval -> UnitInterval
absorbed cfg c i
  | kc == 0 = i
  | otherwise = clampUnit (1 - (1 - unUnit i) ** (1 + kc))
  where
    kc = kappa cfg * max 0 (min 1 c)

-- | How open the gate is at a given shared ground: half open at @floorI@,
-- closing smoothly below it over a width of about @τ@.
gate :: Config -> UnitInterval -> Double
gate cfg i' = 1 / (1 + exp (negate ((unUnit i' - floorI cfg) / tau cfg)))

-- | The height of @d · (1 − d^k)@ at its peak @d* = (1 / (k + 1))^(1/k)@:
-- the most @I' · N@ reaches on the illustrative curve @I = 1 − d@, @N = d@.
peakHeight :: Double -> Double
peakHeight k = d * (1 - d ** k)
  where
    d = (1 / (k + 1)) ** (1 / k)

conf01 :: Double -> Double
conf01 x = if finite x then max 0 (min 1 x) else 0

-- | A path's own strength (the product of its edges' confidences) and how
-- many of its edges are dissonant.
pathShape :: Path -> (Double, Int)
pathShape p = (product (map hopConf (pathHops p)), length [() | Contest _ _ <- pathHops p])
  where
    hopConf (Contest _ c) = conf01 c
    hopConf (Kin c) = conf01 c

-- | What one path adds to the contest between two readers: 0 unless an
-- odd number of its edges are dissonant; else its strength, times the
-- weights of its dissonant edges' kinds, times the weight of its length.
pathContestStrength :: Config -> Path -> Double
pathContestStrength cfg p =
  product [weight t | Contest t _ <- pathHops p]
    * pathContestWith (reach cfg) (noveltyRate cfg) (hopBudget cfg) (length (pathHops p)) dissonant strength
  where
    (strength, dissonant) = pathShape p
    weight t = case t of
      Contests -> wContests (typeWeights cfg)
      Complicates -> wComplicates (typeWeights cfg)
      Refutes -> wRefutes (typeWeights cfg)

-- | What one path adds to the kinship between two readers: 0 unless an
-- even number of its edges (none included) are dissonant; else its
-- strength, losing 'reach' with each further hop. No novelty term:
-- kinship is closeness.
pathKinStrength :: Config -> Path -> Double
pathKinStrength cfg p
  | odd dissonant = 0
  | otherwise = strength * hopWeightWith (reach cfg) 0 (hopBudget cfg) (length (pathHops p))
  where
    (strength, dissonant) = pathShape p

-- | Evidence summed over the paths between two libraries. Only paths with
-- one end in each library count; the same path given twice (same ends,
-- same edges in kind) counts once, at its strongest; the paths arriving
-- at any one work add at most @workCap@. Summed in a fixed order, so it
-- is exactly symmetric in the two libraries.
evidence :: (Path -> Double) -> Config -> Library -> Library -> [Path] -> Double
evidence worth cfg (Library as) (Library bs) paths =
  foldl' (+) 0 (map capped (groupOn (pathTo . fst) deduped))
  where
    inA w = w `elem` as
    inB w = w `elem` bs
    across p =
      pathFrom p /= pathTo p
        && not (null (pathHops p))
        && ((inA (pathFrom p) && inB (pathTo p)) || (inB (pathFrom p) && inA (pathTo p)))
    kind h = case h of
      Contest t _ -> Just t
      Kin _ -> Nothing
    key (p, _) = (pathTo p, pathFrom p, map kind (pathHops p))
    -- by key, the strongest copy first; keep it
    deduped =
      (\gs -> [x | (x : _) <- gs])
        . groupOn key
        . sortBy (comparing key <> comparing (negate . snd))
        $ [(p, worth p) | p <- paths, across p]
    capped ps = min (workCap cfg) (foldl' (+) 0 (map snd ps))

-- | The strength of the contest between two libraries, from the paths
-- between their works.
contestStrength :: Config -> Library -> Library -> [Path] -> Double
contestStrength cfg = evidence (pathContestStrength cfg) cfg

-- | The strength of the kinship between two libraries, from the paths
-- between their works.
kinStrength :: Config -> Library -> Library -> [Path] -> Double
kinStrength cfg = evidence (pathKinStrength cfg) cfg

-- | 'contestStrength' over direct lines alone, each a path of one hop.
divergenceStrength :: Config -> Library -> Library -> [Link] -> Double
divergenceStrength cfg a b = contestStrength cfg a b . map linkPath

groupOn :: Eq k => (a -> k) -> [a] -> [[a]]
groupOn _ [] = []
groupOn f (x : xs) = (x : same) : groupOn f rest
  where
    (same, rest) = span ((== f x) . f) xs

-- | Divergence from line strength: @1 − e^(−s / s0)@, non-decreasing in @s@.
divergence :: Config -> Double -> UnitInterval
divergence cfg s = clampUnit (1 - exp (negate (max 0 s / s0 cfg)))

-- | Shared ground from the embedding percentile and the kinship the graph
-- records: @(1 − w)·p + w·(1 − e^(−kin / s0))@, @w = graphGround@. At
-- @w = 0@ the percentile itself.
ground :: Config -> UnitInterval -> Double -> UnitInterval
ground cfg p kin
  | w == 0 = p
  | otherwise = clampUnit ((1 - w) * unUnit p + w * (1 - exp (negate (max 0 kin / s0 cfg))))
  where
    w = graphGround cfg

-- | Tension from capacity, shared ground and divergence:
-- @g(I') · I' · N@, divided by the peak height when @normalizePeak@.
tensionFrom :: Config -> Double -> UnitInterval -> UnitInterval -> UnitInterval
tensionFrom cfg c i n
  | normalizePeak cfg = clampUnit (raw / peakHeight k)
  | otherwise = clampUnit raw
  where
    i' = absorbed cfg c i
    k = 1 + kappa cfg * max 0 (min 1 c)
    raw = gate cfg i' * unUnit i' * unUnit n

-- ─── scores ────────────────────────────────────────────────────────────

data ScoreError
  = NoCalibration String
  | InvalidCalibration String String
  | InvalidConfig String
  | UnreadableCosine Double
  deriving (Eq, Show)

-- | The tension between two readers: their libraries, the raw cosine
-- between them, and the paths the graph holds between their works (a
-- snapshot: the same paths always give the same score). 'Left' for a
-- config or calibration that cannot be used (whatever the readers);
-- @Right Nothing@ when either library is smaller than @minLibrarySize@.
scorePaths :: Config -> Calibrations -> Library -> Library -> Double -> [Path] -> Either ScoreError (Maybe UnitInterval)
scorePaths cfg0 cals a b cosAB paths = do
  cfg <- validateConfig cfg0
  cal <- calibrationFor cfg cals
  if not (finite cosAB)
    then Left (UnreadableCosine cosAB)
    else
      if librarySize a < minLibrarySize cfg || librarySize b < minLibrarySize cfg
        then Right Nothing
        else
          let c = pairCapacity cfg (librarySize a) (librarySize b)
              i = ground cfg (percentile cal cosAB) (kinStrength cfg a b paths)
              n = divergence cfg (contestStrength cfg a b paths)
           in Right (Just (tensionFrom cfg c i n))

-- | 'scorePaths' over direct lines alone.
scorePair :: Config -> Calibrations -> Library -> Library -> Double -> [Link] -> Either ScoreError (Maybe UnitInterval)
scorePair cfg cals a b cosAB = scorePaths cfg cals a b cosAB . map linkPath

-- | A table: the mean of its scorable pairs and a lower bound under it.
data TableScore = TableScore
  { tableMean :: UnitInterval
  , tableLower :: UnitInterval
  -- ^ @max 0 (mean − λ·σ)@, σ the spread of the table's pairs
  }
  deriving (Eq, Show)

-- | 'Nothing' with fewer than two scorable pairs.
scoreTable :: Config -> [Maybe UnitInterval] -> Maybe TableScore
scoreTable cfg pairs =
  case [unUnit x | Just x <- pairs] of
    xs@(_ : _ : _) ->
      let n = fromIntegral (length xs)
          mean = foldl' (+) 0 xs / n
          sd = sqrt (foldl' (+) 0 [(x - mean) * (x - mean) | x <- xs] / n)
       in Just (TableScore (clampUnit mean) (clampUnit (mean - lambda cfg * sd)))
    _ -> Nothing
