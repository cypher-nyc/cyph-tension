-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0. If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.

-- | Property tests for "Cyph.Tension": the research claims the model makes,
-- each checked across thousands of generated inputs rather than a handful of
-- examples.
--
-- Base only, no packages: inputs come from a small deterministic generator
-- (splitmix64), so a failure prints its seed and reproduces exactly.
module Main (main) where

import Cyph.Tension
import Data.Bits (shiftR, xor)
import Data.Word (Word64)
import System.Exit (exitFailure, exitSuccess)

-- ─── a tiny generator ──────────────────────────────────────────────────

newtype Gen = Gen Word64

next64 :: Gen -> (Word64, Gen)
next64 (Gen s) =
  let s' = s + 0x9e3779b97f4a7c15
      z1 = (s' `xor` (s' `shiftR` 30)) * 0xbf58476d1ce4e5b9
      z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94d049bb133111eb
   in (z2 `xor` (z2 `shiftR` 31), Gen s')

-- | A double uniformly in [lo, hi].
uniform :: Double -> Double -> Gen -> (Double, Gen)
uniform lo hi g =
  let (w, g') = next64 g
      u = fromIntegral (w `shiftR` 11) / 9007199254740992.0
   in (lo + u * (hi - lo), g')

-- | n doubles in [lo, hi].
uniforms :: Int -> Double -> Double -> Gen -> ([Double], Gen)
uniforms n lo hi g0 = go n g0 []
  where
    go 0 g acc = (reverse acc, g)
    go k g acc = let (x, g') = uniform lo hi g in go (k - 1) g' (x : acc)

-- ─── the harness ───────────────────────────────────────────────────────

-- | Check a law over @trials@ generated cases; report the first failure.
law :: String -> Int -> Int -> ([Double] -> Maybe String) -> Int -> IO Bool
law name arity trials check seed = go 0 (Gen (fromIntegral seed))
  where
    go i g
      | i >= trials = do
          putStrLn ("  ok    " ++ name ++ "  (" ++ show trials ++ " cases)")
          pure True
      | otherwise =
          let (xs, g') = uniforms arity (-0.5) 1.5 g
           in case check xs of
                Nothing -> go (i + 1) g'
                Just why -> do
                  putStrLn ("  FAIL  " ++ name ++ "  seed=" ++ show seed ++ " case=" ++ show i)
                  putStrLn ("        inputs " ++ show xs ++ ": " ++ why)
                  pure False

unit :: Double -> Bool
unit v = v >= 0 && v <= 1

expect :: Bool -> String -> Maybe String
expect ok why = if ok then Nothing else Just why

eps :: Double
eps = 1e-12

-- ─── the laws ──────────────────────────────────────────────────────────

laws :: [(String, Int, [Double] -> Maybe String)]
laws =
  [ ( "every score lands in [0, 1], whatever comes in"
    , 4
    , \[a, b, t, s] ->
        expect
          ( all unit
              [ normalizeCosine a
              , computeIntelligibility a b
              , computeNovelty a b
              , computeTrajectory a b
              , computeTension a b t
              , computeSuitedness a t
              , pairIntelligibility a (Just b)
              , pairIntelligibility a Nothing
              , pairNovelty (s * 10)
              , pairTension a b
              ]
          )
          "a score left [0, 1]"
    )
  , ( "below the intelligibility floor, tension and suitedness are exactly zero (Bail 2018)"
    , 3
    , \[a, n, t] ->
        let i = clamp01 a * iFloor * 0.999
         in expect (computeTension i n t == 0 && computeSuitedness i t == 0) "nonzero under the floor"
    )
  , ( "no divergence evidence, no tension"
    , 2
    , \[i, t] -> expect (computeTension i 0 t == 0) "tension without novelty"
    )
  , ( "trajectory modulates, never creates: bonus within (1 + λ)"
    , 3
    , \[i, n, t] ->
        let base = computeTension i n 0
            with = computeTension i n t
         in expect (with + eps >= base && with <= min 1 (base * (1 + lambdaTrajectory)) + eps) "bonus out of bounds"
    )
  , ( "tension rises with novelty and with trajectory"
    , 4
    , \[i, n1, n2, t] ->
        let (lo, hi) = (min n1 n2, max n1 n2)
         in expect
              ( computeTension i lo t <= computeTension i hi t + eps
                  && computeTension i lo (min t 0) <= computeTension i lo (max t 1) + eps
              )
              "not monotone"
    )
  , ( "trajectory: zero when not moving toward the topic, saturating at T_MAX"
    , 2
    , \[a, b] ->
        let older = a
            recent = b
            v = computeTrajectory recent older
         in expect
              ( (recent - older > 0 || v == 0)
                  && (recent - older < tMax || v == 1)
              )
              "trajectory off its curve"
    )
  , ( "novelty rises with each kind of evidence"
    , 3
    , \[r1, r2, s] ->
        let (lo, hi) = (min r1 r2, max r1 r2)
         in expect
              (computeNovelty lo s <= computeNovelty hi s + eps && computeNovelty s lo <= computeNovelty s hi + eps)
              "novelty fell as evidence grew"
    )
  , ( "the inverted-U is structural: I = 1 - d, N = d peaks strictly inside"
    , 1
    , \[t] ->
        let tr = clamp01 t
            ds = [fromIntegral k / 200 | k <- [0 .. 200 :: Int]]
            curve = [computeTension (1 - x) x tr | x <- ds]
            peakAt = snd (maximum (zip curve [0 :: Int ..]))
            (rise, fall) = splitAt (peakAt + 1) curve
            up = and (zipWith (<=) rise (drop 1 rise))
            down = and (zipWith (>=) fall (drop 1 fall))
         in expect
              (take 1 curve == [0] && drop 200 curve == [0] && peakAt > 0 && peakAt < 200 && up && down)
              "not a single interior peak"
    )
  , ( "suitedness carries depth and movement, with no divergence reward"
    , 3
    , \[i1, i2, t] ->
        let (lo, hi) = (min i1 i2, max i1 i2)
         in expect (computeSuitedness lo t <= computeSuitedness hi t + eps) "suitedness fell as depth grew"
    )
  , ( "pair novelty: zero for no contest, saturating below 1, rising with strength"
    , 2
    , \[a, b] ->
        let (lo, hi) = (min a b * 10, max a b * 10)
         in expect
              ( pairNovelty 0 == 0
                  && pairNovelty (negate (abs a)) == 0
                  && pairNovelty lo <= pairNovelty hi + eps
                  && pairNovelty 5 < 1
              )
              "pair novelty off its curve"
    )
  , ( "pair intelligibility: with no topic vector, the people half alone"
    , 1
    , \[p] -> expect (pairIntelligibility p Nothing == normalizeCosine p) "not the people half"
    )
  , ( "pair tension is the topic curve with no trajectory"
    , 2
    , \[i, n] -> expect (pairTension i n == computeTension i n 0) "differs from compute_tension(i, n, 0)"
    )
  , ( "a group's mean sits between its lowest and highest pair"
    , 4
    , \xs ->
        let ys = map clamp01 xs
         in case carMean ys of
              Nothing -> Just "no mean for a full group"
              Just m -> expect (m >= minimum ys - eps && m <= maximum ys + eps) "mean outside its pairs"
    )
  , ( "absorptive capacity never lowers shared ground, and keeps it in [0, 1]"
    , 3
    , \[k, i, cap] ->
        let v = absorbWith (k * 3) i cap
         in expect (unit v && v + eps >= clamp01 i) "capacity lowered shared ground"
    )
  , ( "with no capacity, or κ = 0, absorb changes nothing"
    , 2
    , \[k, i] ->
        expect (absorbWith (k * 3) i 0 == clamp01 i && absorbWith 0 i 1 == clamp01 i && absorb i 1 == clamp01 i) "absorb moved a score it should leave"
    )
  , ( "shared ground rises with capacity and with κ"
    , 4
    , \[k, i, c1, c2] ->
        let (lo, hi) = (min c1 c2, max c1 c2)
            kk = abs k * 3
         in expect (absorbWith kk i lo <= absorbWith kk i hi + eps && absorbWith 0 i hi <= absorbWith kk i hi + eps) "not monotone in capacity"
    )
  , ( "the best distance moves outward as capacity grows (Nooteboom 2007, H3)"
    , 2
    , \[c1, c2] ->
        let (lo, hi) = (clamp01 (min c1 c2), clamp01 (max c1 c2))
            ds = [fromIntegral j / 400 | j <- [0 .. 400 :: Int]]
            peak cap = snd (maximum (zip [pairTension (absorbWith 1.5 (1 - x) cap) x | x <- ds] [0 :: Int ..]))
         in expect (peak lo <= peak hi && peak 0 == 200 && peak 1 > 200) "the peak did not move outward"
    )
  , ( "lines against one work: between the strongest alone and the plain sum"
    , 5
    , \(r : xs) ->
        let kept = map (max 0) xs
            v = lineStrengthWith r xs
         in expect
              ( v + eps >= maximum kept
                  && v <= sum kept + eps
                  && abs (lineStrengthWith 0 xs - sum kept) <= eps
                  && abs (lineStrengthWith 1 xs - maximum kept) <= eps
                  && lineStrength xs == lineStrengthWith lineDependence xs
              )
              "line strength outside its bounds"
    )
  , ( "a group's floor lies within its pair scores, and someone with no counterpart sends it to zero"
    , 3
    , \[a, b, x] ->
        let [ab, ac, bc] = map clamp01 [a, b, x]
            trio = [[Nothing, Just ab, Just ac], [Just ab, Nothing, Just bc], [Just ac, Just bc, Nothing]]
            alone = [r ++ [Just 0] | r <- trio] ++ [[Just 0, Just 0, Just 0, Nothing]]
         in case (groupFloor trio, carMean [ab, ac, bc]) of
              (Just f, Just _) ->
                expect
                  ( f <= maximum [ab, ac, bc] + eps
                      && f + eps >= minimum [max ab ac, max ab bc, max ac bc]
                      && groupFloor alone == Just 0
                  )
                  "floor off"
              _ -> Just "no floor for a readable group"
    )
  ]

-- | Facts about the constants: each blend is a whole, the band is a band.
constants :: [(String, Bool)]
constants =
  [ ("W_VEC + W_GRAPH = 1", wVec + wGraph == 1)
  , ("READING_W + STANCE_W = 1", readingW + stanceW == 1)
  , ("W_PEOPLE + W_TOPIC = 1", wPeople + wTopic == 1)
  , ("COS_LO < COS_HI", cosLo < cosHi)
  , ("0 < I_FLOOR < 1", iFloor > 0 && iFloor < 1)
  , ("a car nobody could read has no score", carMean [] == Nothing)
  , ("a group nobody could read has no floor", groupFloor [[Nothing, Nothing], [Nothing, Nothing]] == Nothing)
  , ("κ and ρ are off until fitted", kappaAbsorb == 0 && lineDependence == 0)
  ]

main :: IO ()
main = do
  putStrLn "Cyph.Tension properties"
  results <- mapM (\(k, (name, arity, check)) -> law name arity 5000 check (1000 + k)) (zip [0 ..] laws)
  facts <-
    mapM
      ( \(name, ok) -> do
          putStrLn ((if ok then "  ok    " else "  FAIL  ") ++ name)
          pure ok
      )
      constants
  let failed = length (filter not (results ++ facts))
      total = length results + length facts
  putStrLn (show (total - failed) ++ "/" ++ show total ++ " properties hold")
  if failed == 0 then exitSuccess else exitFailure
