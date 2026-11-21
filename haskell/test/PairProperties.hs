-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0. If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.

-- | Property and golden tests for "Cyph.Pair".
--
-- Base only, no packages (QuickCheck-style laws over a splitmix64
-- generator, as in @Properties.hs@): a failure prints its seed and its
-- input and reproduces exactly.
--
-- > pair-properties              run the laws and the golden checks
-- > pair-properties --figure13   print the figure's curve (the golden file)
module Main (main) where

import Cyph.Pair
import Data.Bits (shiftR, xor)
import Data.List (maximumBy)
import Data.Ord (comparing)
import Data.Word (Word64)
import Numeric (showFFloat)
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)

-- ─── a tiny generator ──────────────────────────────────────────────────

newtype Seed = Seed Word64

next64 :: Seed -> (Word64, Seed)
next64 (Seed s) =
  let s' = s + 0x9e3779b97f4a7c15
      z1 = (s' `xor` (s' `shiftR` 30)) * 0xbf58476d1ce4e5b9
      z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94d049bb133111eb
   in (z2 `xor` (z2 `shiftR` 31), Seed s')

newtype Gen a = Gen (Seed -> (a, Seed))

runGen :: Gen a -> Seed -> (a, Seed)
runGen (Gen f) = f

instance Functor Gen where
  fmap f (Gen g) = Gen (\s -> let (a, s') = g s in (f a, s'))

instance Applicative Gen where
  pure a = Gen (\s -> (a, s))
  Gen f <*> Gen g = Gen (\s -> let (h, s1) = f s; (a, s2) = g s1 in (h a, s2))

instance Monad Gen where
  Gen g >>= k = Gen (\s -> let (a, s1) = g s in runGen (k a) s1)

-- | A double uniformly in [lo, hi].
double :: Double -> Double -> Gen Double
double lo hi = Gen $ \s ->
  let (w, s') = next64 s
      u = fromIntegral (w `shiftR` 11) / 9007199254740992.0
   in (lo + u * (hi - lo), s')

-- | An int uniformly in [lo, hi].
int :: Int -> Int -> Gen Int
int lo hi = do
  u <- double 0 1
  pure (min hi (lo + floor (u * fromIntegral (hi - lo + 1))))

bool :: Gen Bool
bool = (< 0.5) <$> double 0 1

listOf :: Int -> Int -> Gen a -> Gen [a]
listOf lo hi g = do
  n <- int lo hi
  mapM (const g) [1 .. n]

elements :: [a] -> Gen a
elements xs = (xs !!) <$> int 0 (length xs - 1)

unit' :: Gen UnitInterval
unit' = clampUnit <$> double 0 1

-- ─── the harness ───────────────────────────────────────────────────────

forAll :: Show a => String -> Int -> Gen a -> (a -> Maybe String) -> Int -> IO Bool
forAll name trials g check seed = go 0 (Seed (fromIntegral seed))
  where
    go i s
      | i >= trials = do
          putStrLn ("  ok    " ++ name ++ "  (" ++ show trials ++ " cases)")
          pure True
      | otherwise =
          let (x, s') = runGen g s
           in case check x of
                Nothing -> go (i + 1) s'
                Just why -> do
                  putStrLn ("  FAIL  " ++ name ++ "  seed=" ++ show seed ++ " case=" ++ show i)
                  putStrLn ("        input " ++ show x ++ ": " ++ why)
                  pure False

expect :: Bool -> String -> Maybe String
expect ok why = if ok then Nothing else Just why

-- ─── generated inputs ──────────────────────────────────────────────────

modelId :: String
modelId = "test-embedding"

genConfig :: Gen Config
genConfig = do
  k <- elements [0, 0, 0.5, 1, 2, 3]
  t <- double 0.002 0.1
  s <- double 0.2 3
  nr <- int 1 1000
  l <- double 0 3
  np <- bool
  m <- int 0 6
  ws <- mapM (const (double 0 2)) [1 :: Int, 2, 3]
  cap <- double 0.2 3
  rch <- double 0 1
  nov <- double 0 0.95
  budget <- int 1 6
  gg <- elements [0, 0, 0.3, 0.5, 1]
  let [wc, wm, wr] = ws
  pure
    (defaultConfig modelId)
      { kappa = k
      , tau = t
      , s0 = s
      , nRef = nr
      , lambda = l
      , normalizePeak = np
      , minLibrarySize = m
      , typeWeights = TypeWeights wc wm wr
      , workCap = cap
      , reach = rch
      , noveltyRate = nov
      , hopBudget = budget
      , graphGround = gg
      }

universe :: [WorkId]
universe = ["w" ++ show i | i <- [0 :: Int .. 14]]

genLibrary :: Gen Library
genLibrary = library <$> listOf 0 12 (elements universe)

genLink :: Gen Link
genLink = Link <$> elements universe <*> elements universe <*> elements [minBound .. maxBound] <*> double (-0.2) 1.2

genHop :: Gen Hop
genHop = do
  kin <- bool
  c <- double (-0.2) 1.2
  if kin then pure (Kin c) else (\t -> Contest t c) <$> elements [minBound .. maxBound]

genPath :: Gen Path
genPath = Path <$> elements universe <*> elements universe <*> listOf 0 6 genHop

-- | Random-pair cosines from a skewed, tied-in-places sample.
genCalibrations :: Gen Calibrations
genCalibrations = do
  xs <- listOf 2 60 (double 0.1 0.9)
  let sample = 0.1 : 0.9 : map (\x -> fromIntegral (round (x * 200) :: Int) / 200) xs
  pure (either (error . show) id (calibration modelId sample >>= \c -> calibrations [c]))

data PairCase = PairCase Config Library Library Double [Link]
  deriving (Show)

data PathCase = PathCase Config Library Library Double [Path]
  deriving (Show)

pathGen :: Gen (PathCase, Shown Calibrations)
pathGen = do
  (PairCase cfg a b c _, cals) <- genPair
  ps <- listOf 0 25 genPath
  pure (PathCase cfg a b c ps, Shown cals)

scoreP :: Calibrations -> PathCase -> Either ScoreError (Maybe UnitInterval)
scoreP cals (PathCase cfg a b c ps) = scorePaths cfg cals a b c ps

-- | One contest reached through @h@ hops: kinship edges, then the contest.
reached :: Int -> Path
reached h = Path "a0" "b0" (replicate (h - 1) (Kin 1) ++ [Contest Contests 1])

-- Calibrations has no useful Show of its own here; it travels beside the case
genPair :: Gen (PairCase, Calibrations)
genPair = do
  cfg <- genConfig
  a <- genLibrary
  b <- genLibrary
  c <- double 0 1
  ls <- listOf 0 25 genLink
  cals <- genCalibrations
  pure (PairCase cfg a b c ls, cals)

newtype Shown a = Shown a

instance Show (Shown a) where
  show _ = "<calibrations>"

pairGen :: Gen (PairCase, Shown Calibrations)
pairGen = fmap (fmap Shown) genPair

score :: Calibrations -> PairCase -> Either ScoreError (Maybe UnitInterval)
score cals (PairCase cfg a b c ls) = scorePair cfg cals a b c ls

-- ─── the laws ──────────────────────────────────────────────────────────

laws :: [Int -> IO Bool]
laws =
  [ forAll
      "tension lands in [0, 1] (the terms and the pair score)"
      5000
      ((,,,) <$> genConfig <*> double (-0.5) 1.5 <*> unit' <*> unit')
      ( \(cfg, c, i, n) ->
          let t = unUnit (tensionFrom cfg c i n)
           in expect (t >= 0 && t <= 1) ("tension " ++ show t)
      )
  , forAll
      "a pair's score is in [0, 1], or Nothing"
      5000
      pairGen
      ( \(pc, Shown cals) -> case score cals pc of
          Right (Just t) -> expect (unUnit t >= 0 && unUnit t <= 1) ("score " ++ show t)
          Right Nothing -> Nothing
          Left e -> Just ("error " ++ show e)
      )
  , forAll
      "symmetric: score a b == score b a, exactly"
      5000
      pairGen
      ( \(PairCase cfg a b c ls, Shown cals) ->
          let ab = score cals (PairCase cfg a b c ls)
              ba = score cals (PairCase cfg b a c ls)
           in expect (ab == ba) (show ab ++ " /= " ++ show ba)
      )
  , forAll
      "κ·c = 0 leaves shared ground as it is (I' == I)"
      5000
      ((,,,) <$> genConfig <*> double 0 1 <*> unit' <*> bool)
      ( \(cfg, c, i, zeroKappa) ->
          let (cfg', c') = if zeroKappa then (cfg {kappa = 0}, c) else (cfg, 0)
           in expect (absorbed cfg' c' i == i) (show (absorbed cfg' c' i) ++ " /= " ++ show i)
      )
  , forAll
      "capacity never lowers shared ground, and leaves 0 and 1 where they are"
      5000
      ((,,) <$> genConfig <*> double 0 1 <*> unit')
      ( \(cfg, c, i) ->
          let i' = absorbed cfg c i
           in expect
                ( i' >= i
                    && absorbed cfg c (clampUnit 0) == clampUnit 0
                    && absorbed cfg c (clampUnit 1) == clampUnit 1
                )
                (show i' ++ " from " ++ show i)
      )
  , forAll
      "continuous near the floor: no step as I' crosses floorI"
      5000
      ((,,) <$> genConfig <*> double (-5) 5 <*> unit')
      ( \(cfg0, z, n) ->
          let cfg = cfg0 {kappa = 0}
              x = floorI cfg + z * tau cfg
              h = 1e-7
              t v = unUnit (tensionFrom cfg 0 (clampUnit v) n)
              -- |d/dI' (g · I' · N)| <= 1 / (4τ) + 1, scaled by the peak divisor
              lip = (1 / (4 * tau cfg) + 1) / (if normalizePeak cfg then peakHeight 1 else 1)
              -- a random step near the floor, and a step straddling it exactly
              jump = max (abs (t (x + h) - t x)) (abs (t (floorI cfg + h / 2) - t (floorI cfg - h / 2)))
           in expect (jump <= h * lip * 1.0001 + 1e-13) ("jump " ++ show jump ++ " near I' = " ++ show x)
      )
  , forAll
      "divergence is non-decreasing in s"
      5000
      ((,,) <$> genConfig <*> double (-1) 10 <*> double 0 5)
      ( \(cfg, s, ds) ->
          expect (divergence cfg s <= divergence cfg (s + ds)) (show s ++ " +" ++ show ds)
      )
  , forAll
      "tension is non-decreasing in divergence"
      5000
      ((,,,) <$> genConfig <*> double 0 1 <*> unit' <*> ((,) <$> unit' <*> unit'))
      ( \(cfg, c, i, (n1, n2)) ->
          let (lo, hi) = (min n1 n2, max n1 n2)
           in expect (tensionFrom cfg c i lo <= tensionFrom cfg c i hi) "fell as N rose"
      )
  , forAll
      "one more line never lowers s (dedup keeps the strongest, caps are caps)"
      5000
      ((,) <$> pairGen <*> genLink)
      ( \((PairCase cfg a b _ ls, _), l) ->
          let before = divergenceStrength cfg a b ls
              after = divergenceStrength cfg a b (l : ls)
           in expect (after >= before) (show after ++ " < " ++ show before)
      )
  , forAll
      "a repeated line counts once"
      5000
      pairGen
      ( \(PairCase cfg a b _ ls, _) ->
          expect (divergenceStrength cfg a b (ls ++ ls) == divergenceStrength cfg a b ls) "doubled"
      )
  , forAll
      "no work adds more than workCap"
      5000
      pairGen
      ( \(PairCase cfg a b _ ls, _) ->
          let s = divergenceStrength cfg a b ls
              targets = length universe
           in expect (s <= workCap cfg * fromIntegral targets + 1e-9) ("s " ++ show s)
      )
  , forAll
      "a library under minLibrarySize is not scored"
      5000
      pairGen
      ( \(pc@(PairCase cfg a b _ _), Shown cals) ->
          let small = librarySize a < minLibrarySize cfg || librarySize b < minLibrarySize cfg
           in case score cals pc of
                Right Nothing -> expect small "Nothing for libraries big enough"
                Right (Just _) -> expect (not small) "scored a library that is too small"
                Left e -> Just (show e)
      )
  , forAll
      "over paths: the score is in [0, 1] or Nothing, and symmetric, exactly"
      5000
      pathGen
      ( \(pc@(PathCase cfg a b c ps), Shown cals) ->
          let ab = scoreP cals pc
              ba = scoreP cals (PathCase cfg b a c ps)
           in case ab of
                Left e -> Just ("error " ++ show e)
                Right r -> expect (ab == ba && maybe True (\t -> unUnit t >= 0 && unUnit t <= 1) r) (show ab ++ " vs " ++ show ba)
      )
  , forAll
      "one more path never lowers the contest or the kinship; the same path twice counts once"
      5000
      ((,) <$> pathGen <*> genPath)
      ( \((PathCase cfg a b _ ps, _), q) ->
          expect
            ( contestStrength cfg a b (q : ps) >= contestStrength cfg a b ps
                && kinStrength cfg a b (q : ps) >= kinStrength cfg a b ps
                && contestStrength cfg a b (ps ++ ps) == contestStrength cfg a b ps
            )
            "evidence fell, or doubled"
      )
  , forAll
      "a path is a contest on an odd number of dissonant edges and kinship on an even one, never both"
      5000
      ((,) <$> genConfig <*> genPath)
      ( \(cfg, q) ->
          let dis = length [() | Contest _ _ <- pathHops q]
              ct = pathContestStrength cfg q
              kn = pathKinStrength cfg q
           in expect ((odd dis || ct == 0) && (even dis || kn == 0) && ct >= 0 && kn >= 0) (show (dis, ct, kn))
      )
  , forAll
      "a direct line is a path of one hop: links and their paths score the same"
      5000
      pairGen
      ( \(pc@(PairCase cfg a b c ls), Shown cals) ->
          expect (score cals pc == scoreP cals (PathCase cfg a b c (map linkPath ls))) "links and paths disagree"
      )
  , forAll
      "the inverted U is in the model: a contest's worth rises to one peak in hops and falls from it"
      5000
      genConfig
      ( \cfg ->
          let ws = map (pathContestStrength cfg . reached) [1 .. hopBudget cfg + 1]
              steps = zipWith (-) (drop 1 ws) ws
              after = dropWhile (<= 1e-12) (dropWhile (>= -1e-12) steps)
           in expect (null after && last ws == 0) ("not one peak, or evidence past the budget: " ++ show ws)
      )
  , forAll
      "graph ground: off, shared ground is the percentile; on, more kinship never lowers it"
      5000
      ((,,,) <$> genConfig <*> unit' <*> double 0 5 <*> double 0 5)
      ( \(cfg, i, k, dk) ->
          expect
            ( ground cfg {graphGround = 0} i k == i
                && ground cfg i k <= ground cfg i (k + dk)
            )
            "ground"
      )
  , forAll
      "a table: lower <= mean, both in [0, 1]; Nothing under two pairs"
      5000
      ((,) <$> genConfig <*> listOf 0 8 (elements [Nothing, Just ()] >>= traverse (const unit')))
      ( \(cfg, ps) ->
          case (scoreTable cfg ps, length [() | Just _ <- ps]) of
            (Nothing, k) -> expect (k < 2) ("no table score with " ++ show k ++ " pairs")
            (Just (TableScore m lo), k) -> expect (k >= 2 && lo <= m) (show (m, lo))
      )
  ]

-- ─── fixed facts ───────────────────────────────────────────────────────

facts :: [(String, Bool)]
facts =
  [ ("an unknown embedding model is an error, not a fallback", noCal == Left (NoCalibration "other-model"))
  , ("defaults validate", validateConfig (defaultConfig modelId) == Right (defaultConfig modelId))
  , ("tau = 0 is refused", isLeft (validateConfig (defaultConfig modelId) {tau = 0}))
  , ("a calibration needs two distinct cosines", isLeft (calibration modelId [0.4, 0.4]))
  , ("two calibrations for one model are refused", isLeft (calibrations [cal, cal]))
  , ("percentile: 0 at the bottom, 1 at the top, mid-rank in between", map (unUnit . percentile cal) [0, 0.2, 0.3, 0.4, 1] == [0, 0, 0.5, 1, 1])
  , ("unitInterval refuses what is outside [0, 1]", map unitInterval [-0.1, 1.1, 0 / 0] == [Nothing, Nothing, Nothing])
  , ("peak height at k = 1 is 1/4", peakHeight 1 == 0.25)
  , ("at the defaults a contest two hops away outweighs a direct one, and three hops", worth 2 > worth 1 && worth 2 > worth 3 && worth 3 > worth 4)
  , ("past the hop budget a path is not evidence", worth 5 == 0)
  , ("two works that refute the same third are kin, not a contest", pathContestStrength dflt enemyOfEnemy == 0 && pathKinStrength dflt enemyOfEnemy > 0)
  , ("readers with no path between them: no contest, so no tension", noPaths == Right (Just (clampUnit 0)))
  , ("readers with no work in common and no direct line still score, through a two-hop contest", fmap (fmap ((> 0) . unUnit)) twoHop == Right (Just True))
  , ("lower = mean − λσ: {0.2, 0.6}, λ = 1 → 0.4 and 0.2", tbl == Just (0.4, 0.2))
  ]
  where
    dflt = defaultConfig modelId
    worth = pathContestStrength dflt . reached
    enemyOfEnemy = Path "a0" "b0" [Contest Refutes 1, Contest Refutes 1]
    libA = library ["a" ++ show i | i <- [0 :: Int .. 5]]
    libB = library ["b" ++ show i | i <- [0 :: Int .. 5]]
    noPaths = scorePaths dflt cals libA libB 0.35 []
    twoHop = scorePaths dflt cals libA libB 0.35 [Path "a0" "b0" [Kin 0.9, Contest Refutes 0.8]]
    cal = either (error . show) id (calibration modelId [0.2, 0.3, 0.4])
    cals = either (error . show) id (calibrations [cal])
    noCal = scorePair (defaultConfig "other-model") cals (library []) (library []) 0.3 []
    isLeft = either (const True) (const False)
    tbl =
      fmap
        (\(TableScore m lo) -> (round4 (unUnit m), round4 (unUnit lo)))
        (scoreTable (defaultConfig modelId) [Just (clampUnit 0.2), Nothing, Just (clampUnit 0.6)])
    round4 x = fromIntegral (round (x * 10000) :: Int) / 10000 :: Double

-- ─── figure 13 ─────────────────────────────────────────────────────────

-- | The figure's illustrative curve: @I = 1 − d@, @N = d@, @normalizePeak@
-- off, at κ·c = 0, 1, 2 (κ = 0, 1, 2 at full capacity).
curve :: Double -> Double -> Double -> Double
curve t kc d =
  unUnit (tensionFrom (defaultConfig modelId) {kappa = kc, tau = t, normalizePeak = False} 1 (clampUnit (1 - d)) (clampUnit d))

-- | Peak and cutoff of the curve to two places, found on a grid of 10⁻⁵.
-- The cutoff is where the gate is half shut (tension falls below half the
-- ungated value).
peakAndCutoff :: Double -> Double -> (Double, Double)
peakAndCutoff t kc = (round2 peak, round2 cutoff)
  where
    grid = [fromIntegral i / 100000 | i <- [0 :: Int .. 100000]]
    peak = fst (maximumBy (comparing snd) [(d, curve t kc d) | d <- grid])
    ungated d = (1 - d ** (1 + kc)) * d
    cutoff = last [d | d <- grid, ungated d > 0, curve t kc d >= 0.5 * ungated d]
    round2 x = fromIntegral (round (x * 100) :: Int) / 100

-- | The figure as published: a contest's weight by the hops it is reached
-- through, at the model's own reach, novelty and budget, with the two
-- opposing terms it is the product of. One row per hop, two past the
-- budget.
figureTsv :: String
figureTsv =
  unlines
    ( "hops\treach\tnovelty\tweight"
        : [ show h ++ concatMap (\x -> '\t' : fmt x) [reach cfg ** fromIntegral (h - 1), 1 - noveltyRate cfg ** fromIntegral h, pathContestStrength cfg (reached h)]
          | h <- [1 .. hopBudget cfg + 2]
          ]
    )
  where
    cfg = defaultConfig modelId
    fmt x = showFFloat (Just 6) x ""

goldenPath :: FilePath
goldenPath = "test/golden/figure13.tsv"

golden :: IO [(String, Bool)]
golden = do
  stored <- readFile goldenPath
  let sharp = [peakAndCutoff 1e-6 kc | kc <- [0, 1, 2]]
  pure
    [ ( "the gate against I = 1 − d, N = d, τ → 0: peaks 0.50 / 0.58 / 0.63, cutoffs 0.85 / 0.92 / 0.95 — got " ++ show sharp
      , sharp == [(0.50, 0.85), (0.58, 0.92), (0.63, 0.95)]
      )
    , ("figure 13, as drawn (a contest's weight by hops): the model's numbers match " ++ goldenPath, stored == figureTsv)
    ]

-- ─── main ──────────────────────────────────────────────────────────────

report :: (String, Bool) -> IO Bool
report (name, ok) = do
  putStrLn ((if ok then "  ok    " else "  FAIL  ") ++ name)
  pure ok

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--figure13"] -> putStr figureTsv
    _ -> do
      putStrLn "Cyph.Pair properties"
      rs <- mapM (\(k, l) -> l (2000 + k)) (zip [0 ..] laws)
      fs <- mapM report facts
      gs <- golden >>= mapM report
      let results = rs ++ fs ++ gs
          failed = length (filter not results)
      putStrLn (show (length results - failed) ++ "/" ++ show (length results) ++ " properties hold")
      if failed == 0 then exitSuccess else exitFailure
