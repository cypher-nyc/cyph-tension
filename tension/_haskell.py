# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
The compiled Haskell model, loaded through ctypes.

`TENSION_BACKEND` chooses:

    auto     (default) Haskell when the library loads, else Python
    haskell  Haskell; if the library cannot load, Python, logged as an error
    python   Python only; the library is never loaded

`CYPH_TENSION_LIB` overrides where the library is looked for; by default
`haskell/build/libcyphtension.{so,dylib}` in this repository. The GHC
runtime is started once per process (`hs_init`).
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_D = ctypes.c_double


def _default_path() -> Path:
    ext = "dylib" if sys.platform == "darwin" else "so"
    return Path(__file__).resolve().parent.parent / "haskell" / "build" / f"libcyphtension.{ext}"


class HaskellTension:
    """The compiled model, called through ctypes. One instance per process."""

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self.path = path
        self._lib = lib

        def fn(name: str, arity: int) -> Callable[..., float]:
            f = getattr(lib, name)
            f.restype = _D
            f.argtypes = [_D] * arity
            return f

        self._normalize_cosine = fn("cyph_normalize_cosine", 1)
        self._intelligibility = fn("cyph_compute_intelligibility", 2)
        self._novelty = fn("cyph_compute_novelty", 2)
        self._trajectory = fn("cyph_compute_trajectory", 2)
        self._tension = fn("cyph_compute_tension", 3)
        self._suitedness = fn("cyph_compute_suitedness", 2)
        self._pair_novelty = fn("cyph_pair_novelty", 1)
        self._pair_tension = fn("cyph_pair_tension", 2)

        self._pair_intelligibility = lib.cyph_pair_intelligibility
        self._pair_intelligibility.restype = _D
        self._pair_intelligibility.argtypes = [_D, _D, ctypes.c_int]

        self._absorb_with = fn("cyph_absorb_with", 3)

        self._line_strength_with = lib.cyph_line_strength_with
        self._line_strength_with.restype = _D
        self._line_strength_with.argtypes = [_D, ctypes.POINTER(_D), ctypes.c_int]

        self._group_floor = lib.cyph_group_floor
        self._group_floor.restype = _D
        self._group_floor.argtypes = [ctypes.POINTER(_D), ctypes.c_int]

        self._car_mean = lib.cyph_car_mean
        self._car_mean.restype = _D
        self._car_mean.argtypes = [ctypes.POINTER(_D), ctypes.c_int]

        self.constants = {
            name: fn(f"cyph_const_{name.lower()}", 0)()
            for name in (
                "I_FLOOR",
                "COS_LO",
                "COS_HI",
                "W_VEC",
                "W_GRAPH",
                "READING_W",
                "STANCE_W",
                "LAMBDA_TRAJECTORY",
                "T_MAX",
                "W_PEOPLE",
                "W_TOPIC",
                "KAPPA_ABSORB",
                "LINE_DEPENDENCE",
            )
        }

    # ─── the model, Python signatures ────────────────────────────────

    def normalize_cosine(self, cosine: float) -> float:
        return self._normalize_cosine(float(cosine))

    def compute_intelligibility(self, cos_to_anchor: float, depth_norm: float) -> float:
        return self._intelligibility(float(cos_to_anchor), float(depth_norm))

    def compute_novelty(self, divergent_reading_ratio: float, stance_ratio: float) -> float:
        return self._novelty(float(divergent_reading_ratio), float(stance_ratio))

    def compute_trajectory(self, cos_recent: float, cos_older: float) -> float:
        return self._trajectory(float(cos_recent), float(cos_older))

    def compute_tension(self, intelligibility: float, novelty: float, trajectory: float = 0.0) -> float:
        return self._tension(float(intelligibility), float(novelty), float(trajectory))

    def compute_suitedness(self, intelligibility: float, trajectory: float = 0.0) -> float:
        return self._suitedness(float(intelligibility), float(trajectory))

    def pair_intelligibility(self, cos_people: float, cos_topic: Optional[float] = None) -> float:
        if cos_topic is None:
            return self._pair_intelligibility(float(cos_people), 0.0, 0)
        return self._pair_intelligibility(float(cos_people), float(cos_topic), 1)

    def pair_novelty(self, contest_strength: float) -> float:
        # Python's `contest_strength or 0.0`: None and 0 both read as no contest
        return self._pair_novelty(float(contest_strength or 0.0))

    def pair_tension(self, intelligibility: float, novelty: float) -> float:
        return self._pair_tension(float(intelligibility), float(novelty))

    def absorb_with(self, kappa: float, intelligibility: float, capacity: float) -> float:
        return self._absorb_with(float(kappa), float(intelligibility), float(capacity))

    def absorb(self, intelligibility: float, capacity: float) -> float:
        return self._absorb_with(self.constants["KAPPA_ABSORB"], float(intelligibility), float(capacity))

    def line_strength_with(self, rho: float, strengths: Iterable[float]) -> float:
        values = [float(s) for s in strengths]
        arr = (_D * len(values))(*values)
        return self._line_strength_with(float(rho), arr, len(values))

    def line_strength(self, strengths: Iterable[float]) -> float:
        return self.line_strength_with(self.constants["LINE_DEPENDENCE"], strengths)

    def group_floor(self, matrix: Iterable[Iterable[Optional[float]]]) -> Optional[float]:
        rows = [[math.nan if v is None else float(v) for v in row] for row in matrix]
        n = len(rows)
        if n == 0 or any(len(row) != n for row in rows):
            raise ValueError("group_floor takes a square matrix")
        flat = [v for row in rows for v in row]
        floor = self._group_floor((_D * len(flat))(*flat), n)
        return None if math.isnan(floor) else floor

    def car_score(self, pair_scores: Iterable[Optional[float]]) -> Optional[float]:
        usable = [float(p) for p in pair_scores if p is not None]
        if not usable:
            return None
        arr = (_D * len(usable))(*usable)
        mean = self._car_mean(arr, len(usable))
        if math.isnan(mean):
            return None
        # rounded here, as car_score always has (Python's round, half-even)
        return round(mean, 4)


_loaded: Optional[HaskellTension] = None
_tried = False
_opened: dict = {}


def library_path() -> Path:
    return Path(os.environ.get("CYPH_TENSION_LIB") or _default_path())


def open_library(path: Optional[Path] = None) -> HaskellTension:
    """Open the compiled model whatever `TENSION_BACKEND` says (the parity
    test uses this to check the library even on a python-backed run). One
    instance per path per process, so the GHC runtime starts once. Raises
    OSError / AttributeError when the library is missing or incomplete."""
    path = Path(path or library_path())
    key = str(path)
    if key not in _opened:
        lib = ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
        lib.hs_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.hs_init.restype = None
        lib.hs_init(None, None)
        _opened[key] = HaskellTension(lib, path)
    return _opened[key]


def backend_choice() -> str:
    choice = os.environ.get("TENSION_BACKEND", "auto").strip().lower()
    return choice if choice in {"auto", "haskell", "python"} else "auto"


def load() -> Optional[HaskellTension]:
    """The compiled model, or None when Python should serve. Loads once."""
    global _loaded, _tried
    if _tried:
        return _loaded
    _tried = True
    choice = backend_choice()
    if choice == "python":
        logger.info("tension backend: python (TENSION_BACKEND=python)")
        return None
    path = library_path()
    try:
        _loaded = open_library(path)
    except (OSError, AttributeError) as exc:
        log = logger.error if choice == "haskell" else logger.info
        log("tension backend: python (Haskell library unavailable at %s: %s)", path, exc)
        return None
    logger.info("tension backend: haskell (%s)", path)
    return _loaded
