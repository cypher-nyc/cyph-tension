{-# LANGUAGE ForeignFunctionInterface #-}

-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0. If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.

-- | C-callable exports of "Cyph.Tension", for any language with a C
-- foreign-function interface (@tension/_haskell.py@ loads them through
-- ctypes). Doubles in, a double out.
--
-- Two shapes need a convention C does not have:
--
--   * an optional argument (@pair_intelligibility@'s @cos_topic@) is passed
--     as the value plus a @has@ flag (0 / 1);
--   * an optional result (@car_mean@ over no readable pairs) comes back
--     as NaN, which the Python wrapper turns into @None@ (a real mean of
--     values in [0, 1] is never NaN).
--
-- The constants are exported too, so the Python face and the golden test
-- read the very values the model uses.
module Cyph.TensionFFI () where

import Cyph.Tension
import qualified Cyph.Pair as P
import Foreign.C.Types (CDouble (..), CInt (..))
import Foreign.Marshal.Array (peekArray)
import Foreign.Ptr (Ptr)

d :: CDouble -> Double
d = realToFrac

c :: Double -> CDouble
c = realToFrac

ffiNormalizeCosine :: CDouble -> CDouble
ffiNormalizeCosine = c . normalizeCosine . d

ffiComputeIntelligibility :: CDouble -> CDouble -> CDouble
ffiComputeIntelligibility a b = c (computeIntelligibility (d a) (d b))

ffiComputeNovelty :: CDouble -> CDouble -> CDouble
ffiComputeNovelty a b = c (computeNovelty (d a) (d b))

ffiComputeTrajectory :: CDouble -> CDouble -> CDouble
ffiComputeTrajectory a b = c (computeTrajectory (d a) (d b))

ffiComputeTension :: CDouble -> CDouble -> CDouble -> CDouble
ffiComputeTension a b t = c (computeTension (d a) (d b) (d t))

ffiComputeSuitedness :: CDouble -> CDouble -> CDouble
ffiComputeSuitedness a t = c (computeSuitedness (d a) (d t))

ffiPairIntelligibility :: CDouble -> CDouble -> CInt -> CDouble
ffiPairIntelligibility people topic has =
  c (pairIntelligibility (d people) (if has /= 0 then Just (d topic) else Nothing))

ffiPairNovelty :: CDouble -> CDouble
ffiPairNovelty = c . pairNovelty . d

ffiWorkIntelligibility :: CDouble -> CDouble -> CDouble
ffiWorkIntelligibility a b = c (workIntelligibility (d a) (d b))

ffiWorkNovelty :: CDouble -> CDouble
ffiWorkNovelty a = c (workNovelty (d a))

ffiWorkTension :: CDouble -> CDouble -> CDouble -> CDouble
ffiWorkTension a b s = c (workTension (d a) (d b) (d s))

ffiPairTension :: CDouble -> CDouble -> CDouble
ffiPairTension a b = c (pairTension (d a) (d b))

ffiCarMean :: Ptr CDouble -> CInt -> IO CDouble
ffiCarMean ptr n
  | n <= 0 = pure (0 / 0)
  | otherwise = do
      xs <- peekArray (fromIntegral n) ptr
      pure (maybe (0 / 0) c (carMean (map d xs)))

ffiAbsorbWith :: CDouble -> CDouble -> CDouble -> CDouble
ffiAbsorbWith k i cap = c (absorbWith (d k) (d i) (d cap))

ffiLineStrengthWith :: CDouble -> Ptr CDouble -> CInt -> IO CDouble
ffiLineStrengthWith rho ptr n
  | n <= 0 = pure (c (lineStrengthWith (d rho) []))
  | otherwise = do
      xs <- peekArray (fromIntegral n) ptr
      pure (c (lineStrengthWith (d rho) (map d xs)))

-- Hop counts cross as doubles (whole numbers; the Python wrapper sends
-- them so) and are truncated back.
ffiHopWeightWith :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble
ffiHopWeightWith r v limit hops = c (hopWeightWith (d r) (d v) (truncate (d limit)) (truncate (d hops)))

ffiPathContestWith :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble
ffiPathContestWith r v limit hops dis s =
  c (pathContestWith (d r) (d v) (truncate (d limit)) (truncate (d hops)) (truncate (d dis)) (d s))

-- | An n-by-n matrix of pair scores, row by row; NaN marks a pair that could
-- not be read. NaN back when the group has no floor.
ffiGroupFloor :: Ptr CDouble -> CInt -> IO CDouble
ffiGroupFloor ptr n
  | n <= 0 = pure (0 / 0)
  | otherwise = do
      xs <- peekArray (fromIntegral (n * n)) ptr
      let cell x = let v = d x in if isNaN v then Nothing else Just v
          size = fromIntegral n
          rows [] = []
          rows ys = let (r, rest) = splitAt size ys in map cell r : rows rest
      pure (maybe (0 / 0) c (groupFloor (rows xs)))

foreign export ccall "cyph_absorb_with" ffiAbsorbWith :: CDouble -> CDouble -> CDouble -> CDouble
foreign export ccall "cyph_line_strength_with" ffiLineStrengthWith :: CDouble -> Ptr CDouble -> CInt -> IO CDouble
foreign export ccall "cyph_group_floor" ffiGroupFloor :: Ptr CDouble -> CInt -> IO CDouble
foreign export ccall "cyph_hop_weight_with" ffiHopWeightWith :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble
foreign export ccall "cyph_path_contest_with" ffiPathContestWith :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble
foreign export ccall "cyph_normalize_cosine" ffiNormalizeCosine :: CDouble -> CDouble
foreign export ccall "cyph_compute_intelligibility" ffiComputeIntelligibility :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_compute_novelty" ffiComputeNovelty :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_compute_trajectory" ffiComputeTrajectory :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_compute_tension" ffiComputeTension :: CDouble -> CDouble -> CDouble -> CDouble
foreign export ccall "cyph_compute_suitedness" ffiComputeSuitedness :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_pair_intelligibility" ffiPairIntelligibility :: CDouble -> CDouble -> CInt -> CDouble
foreign export ccall "cyph_pair_novelty" ffiPairNovelty :: CDouble -> CDouble
foreign export ccall "cyph_pair_tension" ffiPairTension :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_work_intelligibility" ffiWorkIntelligibility :: CDouble -> CDouble -> CDouble
foreign export ccall "cyph_work_novelty" ffiWorkNovelty :: CDouble -> CDouble
foreign export ccall "cyph_work_tension" ffiWorkTension :: CDouble -> CDouble -> CDouble -> CDouble
foreign export ccall "cyph_car_mean" ffiCarMean :: Ptr CDouble -> CInt -> IO CDouble

-- ─── the pair and table score (Cyph.Pair), at the model's defaults ─────
--
-- The random-pair sample shared ground is read against is the caller's (one
-- per embedding model: it chooses which sample to pass), as an array.
-- Library sizes cross as whole-number doubles. A pair whose library is too
-- small to say anything comes back as -1; an unusable sample or cosine as
-- NaN.

pairConfig :: P.Config
pairConfig = P.defaultConfig "caller"

sampleCals :: Ptr CDouble -> CInt -> IO (Either P.ScoreError P.Calibrations)
sampleCals ptr n = do
  xs <- if n <= 0 then pure [] else peekArray (fromIntegral n) ptr
  pure (P.calibration "caller" (map d xs) >>= \cal -> P.calibrations [cal])

ffiPairScore :: Ptr CDouble -> CInt -> CDouble -> CDouble -> CDouble -> CDouble -> IO CDouble
ffiPairScore ptr n na nb cosAB s = do
  cals <- sampleCals ptr n
  pure $ case cals >>= \cs -> P.scoreReadings pairConfig cs (truncate (d na)) (truncate (d nb)) (d cosAB) 0 (d s) of
    Right (Just t) -> c (P.unUnit t)
    Right Nothing -> -1
    Left _ -> 0 / 0

ffiPairGround :: Ptr CDouble -> CInt -> CDouble -> CDouble -> CDouble -> IO CDouble
ffiPairGround ptr n na nb cosAB = do
  xs <- if n <= 0 then pure [] else peekArray (fromIntegral n) ptr
  pure $ case P.calibration "caller" (map d xs) of
    Right cal | not (isNaN (d cosAB) || isInfinite (d cosAB)) -> c (P.unUnit (P.groundRead pairConfig cal (truncate (d na)) (truncate (d nb)) (d cosAB)))
    _ -> 0 / 0

ffiReaderCapacity :: CDouble -> CDouble
ffiReaderCapacity n = c (P.readerCapacity pairConfig (truncate (d n)))

-- | A table's lower bound, mean − λσ; NaN under two pairs.
ffiTableLower :: Ptr CDouble -> CInt -> IO CDouble
ffiTableLower ptr n
  | n < 2 = pure (0 / 0)
  | otherwise = do
      xs <- peekArray (fromIntegral n) ptr
      pure (maybe (0 / 0) (c . P.unUnit . P.tableLower) (P.scoreTable pairConfig (map (Just . P.clampUnit . d) xs)))

foreign export ccall "cyph_pair_score" ffiPairScore :: Ptr CDouble -> CInt -> CDouble -> CDouble -> CDouble -> CDouble -> IO CDouble
foreign export ccall "cyph_pair_ground" ffiPairGround :: Ptr CDouble -> CInt -> CDouble -> CDouble -> CDouble -> IO CDouble
foreign export ccall "cyph_reader_capacity" ffiReaderCapacity :: CDouble -> CDouble
foreign export ccall "cyph_table_lower" ffiTableLower :: Ptr CDouble -> CInt -> IO CDouble

cTAU, cS0, cN_REF, cLAMBDA_TABLE, cMIN_LIBRARY, cWORK_CAP :: CDouble
cTAU = c (P.tau pairConfig)
cS0 = c (P.s0 pairConfig)
cN_REF = fromIntegral (P.nRef pairConfig)
cLAMBDA_TABLE = c (P.lambda pairConfig)
cMIN_LIBRARY = fromIntegral (P.minLibrarySize pairConfig)
cWORK_CAP = c (P.workCap pairConfig)

foreign export ccall "cyph_const_tau" cTAU :: CDouble
foreign export ccall "cyph_const_s0" cS0 :: CDouble
foreign export ccall "cyph_const_n_ref" cN_REF :: CDouble
foreign export ccall "cyph_const_lambda_table" cLAMBDA_TABLE :: CDouble
foreign export ccall "cyph_const_min_library" cMIN_LIBRARY :: CDouble
foreign export ccall "cyph_const_work_cap" cWORK_CAP :: CDouble

-- The tuning surface, for the Python face and the golden test.
cI_FLOOR, cCOS_LO, cCOS_HI, cW_VEC, cW_GRAPH, cREADING_W, cSTANCE_W :: CDouble
cLAMBDA_TRAJECTORY, cT_MAX, cW_PEOPLE, cW_TOPIC, cKAPPA_ABSORB, cLINE_DEPENDENCE :: CDouble
cI_FLOOR = c iFloor
cCOS_LO = c cosLo
cCOS_HI = c cosHi
cW_VEC = c wVec
cW_GRAPH = c wGraph
cREADING_W = c readingW
cSTANCE_W = c stanceW
cLAMBDA_TRAJECTORY = c lambdaTrajectory
cT_MAX = c tMax
cW_PEOPLE = c wPeople
cW_TOPIC = c wTopic
cKAPPA_ABSORB = c kappaAbsorb
cLINE_DEPENDENCE = c lineDependence

cHOP_REACH, cHOP_NOVELTY, cMAX_HOPS :: CDouble
cHOP_REACH = c hopReach
cHOP_NOVELTY = c hopNovelty
cMAX_HOPS = fromIntegral maxHops

foreign export ccall "cyph_const_i_floor" cI_FLOOR :: CDouble
foreign export ccall "cyph_const_cos_lo" cCOS_LO :: CDouble
foreign export ccall "cyph_const_cos_hi" cCOS_HI :: CDouble
foreign export ccall "cyph_const_w_vec" cW_VEC :: CDouble
foreign export ccall "cyph_const_w_graph" cW_GRAPH :: CDouble
foreign export ccall "cyph_const_reading_w" cREADING_W :: CDouble
foreign export ccall "cyph_const_stance_w" cSTANCE_W :: CDouble
foreign export ccall "cyph_const_lambda_trajectory" cLAMBDA_TRAJECTORY :: CDouble
foreign export ccall "cyph_const_t_max" cT_MAX :: CDouble
foreign export ccall "cyph_const_w_people" cW_PEOPLE :: CDouble
foreign export ccall "cyph_const_w_topic" cW_TOPIC :: CDouble
foreign export ccall "cyph_const_kappa_absorb" cKAPPA_ABSORB :: CDouble
foreign export ccall "cyph_const_line_dependence" cLINE_DEPENDENCE :: CDouble
foreign export ccall "cyph_const_hop_reach" cHOP_REACH :: CDouble
foreign export ccall "cyph_const_hop_novelty" cHOP_NOVELTY :: CDouble
foreign export ccall "cyph_const_max_hops" cMAX_HOPS :: CDouble
