#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pleiades–Samian Alignment (STL-PA)
================================

This script:
1) Builds normalised views for Pleiades and Samian Research from CSV files in ./src/
2) Generates spatial candidates and scores them using:
   - Geo score with Pleiades positional accuracy (places_accuracy.csv)
   - String score (RapidFuzz if available; otherwise a lightweight fallback)
   - Time score (interval overlap, expanded by Samian uncertainty, weighted by q_interval)
3) Writes outputs:
   - CSV with all scored candidate pairs
   - JSON summary per Samian site (top candidates + parameters)
   - One or more 300 DPI JPG charts (basic QA/diagnostic statistics)

Folder layout (expected):
  ./mapping.py (this file, rename as you like)
  ./src/places.csv
  ./src/names.csv
  ./src/location_points.csv
  ./src/places_accuracy.csv
  ./src/time_periods.csv            (optional; off by default)
  ./src/samianresearch.csv

Notes:
- place_types.csv is intentionally ignored (it is a vocabulary, not a place→type mapping).
- This is designed to be transparent and paper-friendly (no black-box ML).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

FILES = {
    "places": SRC_DIR / "places.csv",
    "names": SRC_DIR / "names.csv",
    "location_points": SRC_DIR / "location_points.csv",
    "places_accuracy": SRC_DIR / "places_accuracy.csv",
    "time_periods": SRC_DIR / "time_periods.csv",
    "samian": SRC_DIR / "samianresearch.csv",
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------
# CSV reading / parsing
# ---------------------------------------------------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        dtype=str,  # load as strings, then convert
        keep_default_na=False,
        low_memory=False,
    )

    # Normalise column names (BOM/whitespace/case/separators)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


_BC_RE = re.compile(r"^\s*(\d+)\s*BC\s*$", re.IGNORECASE)
_AD_RE = re.compile(r"^\s*(\d+)\s*(AD|CE)?\s*$", re.IGNORECASE)


def parse_time_bound(value: str) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None

    m = _BC_RE.match(v)
    if m:
        return -int(m.group(1))

    m = _AD_RE.match(v)
    if m:
        return int(m.group(1))

    try:
        return int(float(v))
    except ValueError:
        return None


def to_float(value: str) -> Optional[float]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Pleiades view
# ---------------------------------------------------------------------
def build_places_core(places_df: pd.DataFrame) -> pd.DataFrame:
    core = places_df.copy()
    core["pleiades_id"] = core["id"].astype(str).str.strip()
    core["pleiades_uri"] = core["uri"].astype(str).str.strip()
    core["label"] = core["title"].astype(str).str.strip()
    core["latitude"] = core["representative_latitude"].apply(to_float)
    core["longitude"] = core["representative_longitude"].apply(to_float)
    core["bounding_box_wkt"] = core.get("bounding_box_wkt", "").astype(str)
    core["location_precision_place"] = (
        core.get("location_precision", "").astype(str).str.strip().str.lower()
    )
    return core[
        [
            "pleiades_id",
            "pleiades_uri",
            "label",
            "latitude",
            "longitude",
            "bounding_box_wkt",
            "location_precision_place",
        ]
    ]


def build_alt_labels(names_df: pd.DataFrame) -> pd.DataFrame:
    df = names_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()

    cols = [
        c
        for c in [
            "attested_form",
            "romanized_form_1",
            "romanized_form_2",
            "romanized_form_3",
            "title",
        ]
        if c in df.columns
    ]
    if not cols:
        return pd.DataFrame(columns=["pleiades_id", "alt_labels"])

    parts = [df[["place_id", c]].rename(columns={c: "name"}) for c in cols]
    long = pd.concat(parts, ignore_index=True)
    long["name"] = long["name"].astype(str).str.strip()
    long = long[long["name"] != ""]

    # optional certainty filter (names.csv has association_certainty)
    if "association_certainty" in df.columns:
        cert = df[["place_id", "association_certainty"]].copy()
        cert["association_certainty"] = (
            cert["association_certainty"].astype(str).str.strip().str.lower()
        )
        long = long.merge(cert, on="place_id", how="left")
        long = long[long["association_certainty"].isin({"certain", "probable", ""})]
        long = long.drop(columns=["association_certainty"], errors="ignore")

    long = long.drop_duplicates(subset=["place_id", "name"])

    return (
        long.groupby("place_id")["name"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .reset_index()
        .rename(columns={"place_id": "pleiades_id", "name": "alt_labels"})
    )


def build_accuracy(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    df = accuracy_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()

    def fnum(x):
        try:
            v = float(str(x).strip())
            return None if v < 0 else v
        except Exception:
            return None

    df["max_accuracy_m"] = df.get("max_accuracy_meters", "").apply(fnum)
    df["min_accuracy_m"] = df.get("min_accuracy_meters", "").apply(fnum)
    df["accuracy_hull_wkt"] = df.get("accuracy_hull", "").astype(str)
    df["location_precision_accuracy"] = (
        df.get("location_precision", "").astype(str).str.strip().str.lower()
    )

    out = df.rename(columns={"place_id": "pleiades_id"})
    return out[
        [
            "pleiades_id",
            "location_precision_accuracy",
            "accuracy_hull_wkt",
            "min_accuracy_m",
            "max_accuracy_m",
        ]
    ].drop_duplicates(subset=["pleiades_id"])


def build_era_from_locations(
    loc_points_df: pd.DataFrame, allow_rough: bool = False
) -> pd.DataFrame:
    df = loc_points_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()
    df["association_certainty"] = (
        df.get("association_certainty", "").astype(str).str.strip().str.lower()
    )
    df["location_precision"] = (
        df.get("location_precision", "").astype(str).str.strip().str.lower()
    )

    cert_ok = df["association_certainty"].isin({"certain", "probable"})
    if allow_rough:
        prec_ok = df["location_precision"].isin({"precise", "rough"})
    else:
        prec_ok = df["location_precision"].isin({"precise"})
    df = df[cert_ok & prec_ok].copy()

    df["tpq"] = df.get("year_after_which", "").apply(parse_time_bound)
    df["taq"] = df.get("year_before_which", "").apply(parse_time_bound)
    df = df[(df["tpq"].notna()) | (df["taq"].notna())].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["pleiades_id", "earliest_year", "latest_year", "era_source"]
        )

    def agg_min(vals):
        vals = [v for v in vals if v is not None]
        return int(min(vals)) if vals else None

    def agg_max(vals):
        vals = [v for v in vals if v is not None]
        return int(max(vals)) if vals else None

    grouped = (
        df.groupby("place_id")
        .agg(
            earliest_year=("tpq", lambda s: agg_min(s.tolist())),
            latest_year=("taq", lambda s: agg_max(s.tolist())),
        )
        .reset_index()
        .rename(columns={"place_id": "pleiades_id"})
    )
    grouped["era_source"] = "location_points"
    return grouped


def build_time_periods_vocab(time_periods_df: pd.DataFrame) -> pd.DataFrame:
    df = time_periods_df.copy()
    df["key"] = df["key"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    df["lower_i"] = df["lower_bound"].apply(parse_time_bound)
    df["upper_i"] = df["upper_bound"].apply(parse_time_bound)
    df = df[df["lower_i"].notna() & df["upper_i"].notna()].copy()
    df["lower_i"] = df["lower_i"].astype(int)
    df["upper_i"] = df["upper_i"].astype(int)
    return df[["key", "term", "lower_i", "upper_i", "same_as"]]


def interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0, right - left)


def compute_overlapping_time_periods(
    start: Optional[int], end: Optional[int], periods_df: pd.DataFrame
) -> Tuple[str, str]:
    if start is None or end is None or periods_df.empty:
        return "", ""
    if start > end:
        start, end = end, start

    overlaps: List[Tuple[str, str]] = []
    for _, r in periods_df.iterrows():
        lb = r["lower_i"]
        ub = r["upper_i"]
        if lb is None or ub is None:
            continue
        p0, p1 = min(int(lb), int(ub)), max(int(lb), int(ub))
        if interval_overlap(start, end, p0, p1) > 0:
            overlaps.append((r["key"], r["term"]))

    if not overlaps:
        return "", ""
    return ("|".join([k for k, _ in overlaps]), "|".join([t for _, t in overlaps]))


def build_pleiades_view(
    with_time_periods: bool = False, allow_rough_era: bool = False
) -> pd.DataFrame:
    t0 = time.time()
    log("Building Pleiades view...")

    log("Loading places.csv ...")
    places_df = read_csv(FILES["places"])
    log(f"  places: {len(places_df):,}")

    log("Loading names.csv ...")
    names_df = read_csv(FILES["names"])
    log(f"  names: {len(names_df):,}")

    log("Loading location_points.csv ...")
    loc_points_df = read_csv(FILES["location_points"])
    log(f"  location_points: {len(loc_points_df):,}")

    log("Loading places_accuracy.csv ...")
    acc_df = read_csv(FILES["places_accuracy"])
    log(f"  places_accuracy: {len(acc_df):,}")

    periods_vocab = pd.DataFrame()
    if with_time_periods and FILES["time_periods"].exists():
        log("Loading time_periods.csv ...")
        time_periods_df = read_csv(FILES["time_periods"])
        periods_vocab = build_time_periods_vocab(time_periods_df)

    log("Building core places view ...")
    places_core = build_places_core(places_df)

    log("Aggregating alt labels (names.csv) ...")
    t = time.time()
    alt_labels = build_alt_labels(names_df)
    log(f"  alt labels: {len(alt_labels):,} (t={time.time() - t:.1f}s)")

    log("Aggregating era from location_points.csv ...")
    t = time.time()
    era = build_era_from_locations(loc_points_df, allow_rough=allow_rough_era)
    log(f"  era rows: {len(era):,} (t={time.time() - t:.1f}s)")

    log("Loading accuracy info ...")
    accuracy = build_accuracy(acc_df)

    df = places_core.merge(alt_labels, on="pleiades_id", how="left")
    df = df.merge(accuracy, on="pleiades_id", how="left")
    df = df.merge(era, on="pleiades_id", how="left")

    df["alt_labels"] = df["alt_labels"].fillna("")
    df["location_precision"] = df["location_precision_accuracy"].fillna("")
    df.loc[df["location_precision"] == "", "location_precision"] = df[
        "location_precision_place"
    ].fillna("")
    df["location_precision"] = (
        df["location_precision"].astype(str).str.strip().str.lower()
    )

    if with_time_periods and not periods_vocab.empty:
        keys, terms = [], []
        for _, row in df.iterrows():
            k, t = compute_overlapping_time_periods(
                row.get("earliest_year"), row.get("latest_year"), periods_vocab
            )
            keys.append(k)
            terms.append(t)
        df["time_period_keys"] = keys
        df["time_period_terms"] = terms
    else:
        df["time_period_keys"] = ""
        df["time_period_terms"] = ""

    log(f"Pleiades view ready: {len(df):,} rows (total t={time.time() - t0:.1f}s)")
    return df[
        [
            "pleiades_id",
            "pleiades_uri",
            "label",
            "alt_labels",
            "latitude",
            "longitude",
            "location_precision",
            "min_accuracy_m",
            "max_accuracy_m",
            "accuracy_hull_wkt",
            "earliest_year",
            "latest_year",
            "era_source",
            "time_period_keys",
            "time_period_terms",
        ]
    ].copy()


# ---------------------------------------------------------------------
# Samian loader (your schema)
# ---------------------------------------------------------------------
SAMIAN_REQUIRED = [
    "id",
    "label",
    "altlabels",
    "lon",
    "lat",
    "earliest_year",
    "latest_year",
    "q_start",
    "q_end",
    "q_interval",
    "unc_start_years",
    "unc_end_years",
    "unc_interval_years",
]


def load_samian_csv(path: Path) -> pd.DataFrame:
    df = read_csv(path)

    missing = [c for c in SAMIAN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Samian CSV missing columns: {missing}\nFound columns: {list(df.columns)}"
        )

    out = df.copy()
    out["samian_id"] = out["id"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out["alt_labels"] = out["altlabels"].astype(str).fillna("").str.strip()

    out["latitude"] = out["lat"].apply(to_float)
    out["longitude"] = out["lon"].apply(to_float)

    out["earliest_year"] = out["earliest_year"].apply(parse_time_bound)
    out["latest_year"] = out["latest_year"].apply(parse_time_bound)

    for c in ["q_start", "q_end", "q_interval"]:
        out[c] = out[c].apply(to_float)

    for c in ["unc_start_years", "unc_end_years", "unc_interval_years"]:
        out[c] = out[c].apply(parse_time_bound)

    mask = (
        out["earliest_year"].notna()
        & out["latest_year"].notna()
        & (out["earliest_year"] > out["latest_year"])
    )
    if mask.any():
        tmp = out.loc[mask, "earliest_year"].copy()
        out.loc[mask, "earliest_year"] = out.loc[mask, "latest_year"]
        out.loc[mask, "latest_year"] = tmp

    out["has_coords"] = out["latitude"].notna() & out["longitude"].notna()
    out["has_time"] = out["earliest_year"].notna() & out["latest_year"].notna()

    return out[
        [
            "samian_id",
            "label",
            "alt_labels",
            "latitude",
            "longitude",
            "earliest_year",
            "latest_year",
            "q_start",
            "q_end",
            "q_interval",
            "unc_start_years",
            "unc_end_years",
            "unc_interval_years",
            "has_coords",
            "has_time",
        ]
    ].copy()


# ---------------------------------------------------------------------
# STL-PA scoring utilities
# ---------------------------------------------------------------------
def split_pipe(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _norm_name(s)
    # keep letters/numbers, split on everything else
    toks = re.split(r"[^0-9a-zA-Z]+", s)
    toks = [t for t in toks if len(t) >= 2]  # small noise filter
    return toks


def try_rapidfuzz():
    try:
        from rapidfuzz import fuzz  # type: ignore

        return fuzz
    except Exception:
        return None


_RAPIDFUZZ = try_rapidfuzz()


def string_similarity(a: str, b: str) -> float:
    """Return similarity in [0,1]. Uses RapidFuzz if available, otherwise a lightweight fallback."""
    a = _norm_name(a)
    b = _norm_name(b)
    if not a or not b:
        return 0.0

    if _RAPIDFUZZ is not None:
        # token_set_ratio handles word order and additions well
        return float(_RAPIDFUZZ.token_set_ratio(a, b)) / 100.0

    # Fallback: Jaccard over tokens + a mild character-sequence component
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        # very short strings; use a simple ratio-like fallback
        return 1.0 if a == b else 0.0

    jacc = len(ta & tb) / max(1, len(ta | tb))
    # tiny char-level boost for near matches
    # (avoid importing difflib in tight loops)
    if a == b:
        return 1.0
    return float(min(1.0, jacc + 0.15))


def best_string_score(
    samian_label: str, samian_alts: str, pleiades_label: str, pleiades_alts: str
) -> float:
    s_names = [_norm_name(samian_label)] + [
        _norm_name(x) for x in split_pipe(samian_alts)
    ]
    p_names = [_norm_name(pleiades_label)] + [
        _norm_name(x) for x in split_pipe(pleiades_alts)
    ]

    # remove empties and duplicates while keeping small list sizes
    s_names = [x for x in dict.fromkeys([x for x in s_names if x])]
    p_names = [x for x in dict.fromkeys([x for x in p_names if x])]

    best = 0.0
    for a in s_names:
        for b in p_names:
            sc = string_similarity(a, b)
            if sc > best:
                best = sc
                if best >= 0.999:
                    return 1.0
    return best


def haversine_km(
    lat1: float, lon1: float, lat2: Sequence[float], lon2: Sequence[float]
):
    """Vectorised haversine: lat2/lon2 arrays -> distances in km."""
    import numpy as np

    R = 6371.0088  # mean earth radius in km
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(np.array(lat2, dtype=float))
    lon2r = np.radians(np.array(lon2, dtype=float))

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c


def time_score(
    s_start: Optional[int],
    s_end: Optional[int],
    unc_start: Optional[int],
    unc_end: Optional[int],
    q_interval: Optional[float],
    p_start: Optional[int],
    p_end: Optional[int],
) -> float:
    if s_start is None or s_end is None or p_start is None or p_end is None:
        return 0.0
    if s_start > s_end:
        s_start, s_end = s_end, s_start
    if p_start > p_end:
        p_start, p_end = p_end, p_start

    u0 = int(unc_start or 0)
    u1 = int(unc_end or 0)
    s0 = s_start - u0
    s1 = s_end + u1

    overlap = interval_overlap(s0, s1, p_start, p_end)
    span_s = max(1, s1 - s0)
    span_p = max(1, p_end - p_start)
    denom = max(span_s, span_p)
    base = overlap / denom if denom > 0 else 0.0

    q = q_interval if q_interval is not None else 1.0
    q = max(0.0, min(1.0, float(q)))
    return float(base * q)


@dataclass
class STLPAParams:
    # Candidate generation
    radius_km: float = 50.0  # base radius
    topk: int = 25
    adaptive_radius: bool = False
    adaptive_km_per_year: float = 0.5
    adaptive_radius_cap_km: float = 200.0

    # Scoring
    sigma_km: float = 20.0
    rough_penalty: float = 0.6

    # Dynamic weights ("2026+" variant)
    dynamic_weights: bool = False
    dynamic_acc_km_scale: float = 20.0
    dynamic_w_geo_min: float = 0.2

    # Fusion weights (base)
    w_geo: float = 0.5
    w_str: float = 0.4
    w_time: float = 0.1

    # Confidence thresholds
    high_conf: float = 0.75
    medium_conf: float = 0.40

    # Reporting
    html_report: bool = True


def normalise_weights(p: STLPAParams) -> Tuple[float, float, float]:
    s = max(1e-9, p.w_geo + p.w_str + p.w_time)
    return p.w_geo / s, p.w_str / s, p.w_time / s


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dynamic_fusion_weights(
    w_geo: float, w_str: float, w_time: float, acc_km: float, p: STLPAParams
) -> Tuple[float, float, float]:
    """Adjust fusion weights based on Pleiades positional uncertainty.

    Idea ("2026+" variant): if a Pleiades place is spatially uncertain (large accuracy),
    then down-weight the geo component and re-distribute weight to string/time.
    """
    if not p.dynamic_weights:
        return w_geo, w_str, w_time

    scale = max(1e-9, float(p.dynamic_acc_km_scale))
    a = clamp(acc_km / scale, 0.0, 1.0)

    w_geo_adj = (w_geo * (1.0 - a)) + (float(p.dynamic_w_geo_min) * a)
    w_geo_adj = clamp(w_geo_adj, 0.0, 1.0)

    remaining = 1.0 - w_geo_adj
    base_rest = max(1e-9, w_str + w_time)
    w_str_adj = remaining * (w_str / base_rest)
    w_time_adj = remaining * (w_time / base_rest)

    return w_geo_adj, w_str_adj, w_time_adj


def confidence_label(score: float, p: STLPAParams) -> str:
    if score >= p.high_conf:
        return "high"
    if score >= p.medium_conf:
        return "medium"
    return "low"


def run_stlpa(
    pleiades: pd.DataFrame, samian: pd.DataFrame, p: STLPAParams
) -> pd.DataFrame:
    """Return a long DataFrame of (samian, pleiades) candidate pairs with scores."""
    import numpy as np

    w_geo, w_str, w_time = normalise_weights(p)

    # Pre-extract pleiades vectors
    P_lat = pleiades["latitude"].astype(float).to_numpy()
    P_lon = pleiades["longitude"].astype(float).to_numpy()
    P_id = pleiades["pleiades_id"].astype(str).to_numpy()
    P_uri = pleiades["pleiades_uri"].astype(str).to_numpy()
    P_label = pleiades["label"].astype(str).to_numpy()
    P_alts = pleiades["alt_labels"].astype(str).to_numpy()
    P_prec = pleiades["location_precision"].astype(str).to_numpy()
    P_acc_m = (
        pd.to_numeric(pleiades["max_accuracy_m"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    P_start = pd.to_numeric(pleiades["earliest_year"], errors="coerce").to_numpy()
    P_end = pd.to_numeric(pleiades["latest_year"], errors="coerce").to_numpy()

    out_rows: List[Dict[str, object]] = []

    log("Running STL-PA alignment...")
    t0 = time.time()

    for i, srow in samian.iterrows():
        if not bool(srow["has_coords"]):
            continue

        s_id = str(srow["samian_id"])
        s_label = str(srow["label"])
        s_alts = str(srow["alt_labels"])
        s_lat = float(srow["latitude"])
        s_lon = float(srow["longitude"])

        # Spatial candidate distances to all pleiades (vectorised)
        d_km_all = haversine_km(s_lat, s_lon, P_lat, P_lon)

        # Candidate selection: within radius, then take topk by distance
        radius = float(p.radius_km)
        if p.adaptive_radius:
            unc_int = pd.to_numeric(srow.get("unc_interval_years", 0), errors="coerce")
            unc_int = 0.0 if pd.isna(unc_int) else float(unc_int)
            radius = radius + float(p.adaptive_km_per_year) * unc_int
            radius = min(radius, float(p.adaptive_radius_cap_km))
        idx = np.where(d_km_all <= radius)[0]
        if idx.size == 0:
            # fallback to top-k nearest overall
            idx = np.argpartition(d_km_all, min(p.topk, len(d_km_all) - 1))[: p.topk]
        else:
            # limit to topk within radius
            if idx.size > p.topk:
                sub = d_km_all[idx]
                take = np.argpartition(sub, p.topk - 1)[: p.topk]
                idx = idx[take]

        # Score candidates
        candidates = []
        for j in idx.tolist():
            d_km = float(d_km_all[j])
            acc_km = float(P_acc_m[j]) / 1000.0
            d_eff = max(0.0, d_km - acc_km)

            geo = math.exp(-d_eff / float(p.sigma_km))
            if str(P_prec[j]).strip().lower() == "rough":
                geo *= float(p.rough_penalty)

            str_sc = best_string_score(s_label, s_alts, str(P_label[j]), str(P_alts[j]))

            t_sc = time_score(
                srow["earliest_year"],
                srow["latest_year"],
                srow["unc_start_years"],
                srow["unc_end_years"],
                srow["q_interval"],
                None if (pd.isna(P_start[j]) or pd.isna(P_end[j])) else int(P_start[j]),
                None if (pd.isna(P_start[j]) or pd.isna(P_end[j])) else int(P_end[j]),
            )

            w_geo_l, w_str_l, w_time_l = dynamic_fusion_weights(
                w_geo, w_str, w_time, acc_km, p
            )

            geo_contrib = w_geo_l * geo
            str_contrib = w_str_l * str_sc
            time_contrib = w_time_l * t_sc
            final = geo_contrib + str_contrib + time_contrib

            candidates.append(
                (
                    j,
                    d_km,
                    d_eff,
                    geo,
                    str_sc,
                    t_sc,
                    w_geo_l,
                    w_str_l,
                    w_time_l,
                    geo_contrib,
                    str_contrib,
                    time_contrib,
                    final,
                )
            )

        candidates.sort(key=lambda x: x[-1], reverse=True)

        for rank, (
            j,
            d_km,
            d_eff,
            geo,
            str_sc,
            t_sc,
            w_geo_l,
            w_str_l,
            w_time_l,
            geo_contrib,
            str_contrib,
            time_contrib,
            final,
        ) in enumerate(candidates, start=1):
            out_rows.append(
                {
                    "samian_id": s_id,
                    "samian_label": s_label,
                    "samian_alt_labels": s_alts,
                    "samian_lat": s_lat,
                    "samian_lon": s_lon,
                    "samian_earliest_year": srow["earliest_year"],
                    "samian_latest_year": srow["latest_year"],
                    "samian_q_interval": srow["q_interval"],
                    "samian_unc_start_years": srow["unc_start_years"],
                    "samian_unc_end_years": srow["unc_end_years"],
                    "pleiades_id": str(P_id[j]),
                    "pleiades_uri": str(P_uri[j]),
                    "pleiades_label": str(P_label[j]),
                    "pleiades_alt_labels": str(P_alts[j]),
                    "pleiades_lat": float(P_lat[j]),
                    "pleiades_lon": float(P_lon[j]),
                    "pleiades_earliest_year": (
                        None if pd.isna(P_start[j]) else int(P_start[j])
                    ),
                    "pleiades_latest_year": (
                        None if pd.isna(P_end[j]) else int(P_end[j])
                    ),
                    "pleiades_location_precision": str(P_prec[j]),
                    "pleiades_max_accuracy_m": float(P_acc_m[j]),
                    "distance_km": float(d_km),
                    "distance_eff_km": float(d_eff),
                    "geo_score": float(geo),
                    "string_score": float(str_sc),
                    "time_score": float(t_sc),
                    "w_geo": float(w_geo_l),
                    "w_str": float(w_str_l),
                    "w_time": float(w_time_l),
                    "geo_contrib": float(geo_contrib),
                    "string_contrib": float(str_contrib),
                    "time_contrib": float(time_contrib),
                    "final_score": float(final),
                    "rank": int(rank),
                    "confidence": confidence_label(float(final), p),
                }
            )

        if (i + 1) % 25 == 0:
            log(f"  processed {i+1}/{len(samian)} samian sites...")

    out = pd.DataFrame(out_rows)
    log(f"STL-PA done: {len(out):,} candidate pairs (t={time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
def write_outputs(
    df: pd.DataFrame, out_dir: Path, params: STLPAParams, prefix: str = "stlpa"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_candidates_scored.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log(f"Wrote CSV: {csv_path}")

    # JSON summary: per samian site top candidates + parameters
    summary: Dict[str, object] = {
        "params": asdict(params),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samian_sites": {},
    }

    for sid, g in df.groupby("samian_id", sort=False):
        g2 = g.sort_values("final_score", ascending=False).head(10)
        summary["samian_sites"][str(sid)] = {
            "samian_label": str(g2["samian_label"].iloc[0]),
            "top_candidates": [
                {
                    "pleiades_id": r["pleiades_id"],
                    "pleiades_label": r["pleiades_label"],
                    "pleiades_uri": r["pleiades_uri"],
                    "final_score": r["final_score"],
                    "geo_score": r["geo_score"],
                    "string_score": r["string_score"],
                    "time_score": r["time_score"],
                    "w_geo": r.get("w_geo"),
                    "w_str": r.get("w_str"),
                    "w_time": r.get("w_time"),
                    "geo_contrib": r.get("geo_contrib"),
                    "string_contrib": r.get("string_contrib"),
                    "time_contrib": r.get("time_contrib"),
                    "distance_km": r["distance_km"],
                    "confidence": r["confidence"],
                    "rank": int(r["rank"]),
                }
                for _, r in g2.iterrows()
            ],
        }

    json_path = out_dir / f"{prefix}_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Wrote JSON: {json_path}")

    # Convenience: top-1 per samian
    top1 = (
        df.sort_values(["samian_id", "final_score"], ascending=[True, False])
        .groupby("samian_id")
        .head(1)
    )
    top1_path = out_dir / f"{prefix}_top1.csv"
    top1.to_csv(top1_path, index=False, encoding="utf-8")
    log(f"Wrote top-1 CSV: {top1_path}")

    # Basic plots (JPG 300 DPI)
    try:
        import matplotlib.pyplot as plt

        # Histogram of final score
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(df["final_score"].astype(float).to_numpy(), bins=30)
        ax.set_xlabel("final_score")
        ax.set_ylabel("count")
        ax.set_title("STL-PA: distribution of final scores")
        fig_path = out_dir / f"{prefix}_final_score_hist.jpg"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log(f"Wrote plot: {fig_path}")

        # Confidence counts
        fig = plt.figure()
        ax = fig.add_subplot(111)
        counts = df["confidence"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.set_title("STL-PA: confidence class counts")
        fig_path = out_dir / f"{prefix}_confidence_counts.jpg"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log(f"Wrote plot: {fig_path}")

        # -------------------------------------------------------------
        # Extra diagnostic plots for Top-1 matches (JPG 300 DPI)
        # -------------------------------------------------------------
        top1_sorted = top1.sort_values("final_score", ascending=False).copy()

        # Histogram: Top-1 final scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(top1_sorted["final_score"].astype(float).to_numpy(), bins=20)
        ax.set_xlabel("final_score")
        ax.set_ylabel("count")
        ax.set_title("STL-PA Top-1: distribution of final scores")
        fig_path = out_dir / f"{prefix}_top1_final_score_hist.jpg"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log(f"Wrote plot: {fig_path}")

        # Scatter: component contribution vs final_score (Top-1)
        for col in ["geo_contrib", "string_contrib", "time_contrib"]:
            if col not in top1_sorted.columns:
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(
                top1_sorted[col].astype(float).to_numpy(),
                top1_sorted["final_score"].astype(float).to_numpy(),
            )
            ax.set_xlabel(col)
            ax.set_ylabel("final_score")
            ax.set_title(f"STL-PA Top-1: {col} vs final_score")
            fig_path = out_dir / f"{prefix}_top1_{col}_vs_final.jpg"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log(f"Wrote plot: {fig_path}")

        # Score stack (Top-1): stacked contributions by rank (no labels; too many)
        if all(
            c in top1_sorted.columns
            for c in ["geo_contrib", "string_contrib", "time_contrib"]
        ):
            x = list(range(len(top1_sorted)))
            geo = top1_sorted["geo_contrib"].astype(float).to_numpy()
            scc = top1_sorted["string_contrib"].astype(float).to_numpy()
            tim = top1_sorted["time_contrib"].astype(float).to_numpy()

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.bar(x, geo, label="geo_contrib")
            ax.bar(x, scc, bottom=geo, label="string_contrib")
            ax.bar(x, tim, bottom=geo + scc, label="time_contrib")
            ax.set_xlabel("top1_rank (sorted by final_score)")
            ax.set_ylabel("score contribution")
            ax.set_title("STL-PA Top-1: score stack (geo/string/time)")
            ax.set_xticks([])
            ax.legend()
            fig_path = out_dir / f"{prefix}_top1_score_stack.jpg"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log(f"Wrote plot: {fig_path}")

    except Exception as e:
        log(f"Plotting skipped (matplotlib not available or failed): {e}")

    # HTML report (self-contained; references local output files)
    if params.html_report:
        try:
            n_pairs = len(df)
            n_samian = df["samian_id"].nunique()
            avg_cands = n_pairs / max(1, n_samian)

            conf_counts = df["confidence"].value_counts().to_dict()

            top1 = (
                df.sort_values(["samian_id", "final_score"], ascending=[True, False])
                .groupby("samian_id")
                .head(1)
                .sort_values("final_score", ascending=False)
            )

            def esc(s: str) -> str:
                return (
                    str(s)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )

            rows_html = []
            for _, r in top1.iterrows():
                rows_html.append(
                    "<tr>"
                    f"<td>{esc(r['samian_id'])}</td>"
                    f"<td>{esc(r['samian_label'])}</td>"
                    f"<td><a href='{esc(r['pleiades_uri'])}'>{esc(r['pleiades_id'])}</a></td>"
                    f"<td>{esc(r['pleiades_label'])}</td>"
                    f"<td>{float(r['distance_km']):.3f}</td>"
                    f"<td>{float(r['geo_score']):.3f}</td>"
                    f"<td>{float(r['string_score']):.3f}</td>"
                    f"<td>{float(r['time_score']):.3f}</td>"
                    f"<td><b>{float(r['final_score']):.3f}</b></td>"
                    f"<td>{esc(r['confidence'])}</td>"
                    "</tr>"
                )

            params_pretty = esc(json.dumps(asdict(params), indent=2))

            html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>STL-PA Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
    h1,h2 {{ margin: 0.2em 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
    pre {{ background:#f7f7f7; padding: 12px; border-radius: 8px; overflow:auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background:#f2f2f2; text-align: left; }}
    .muted {{ color:#555; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
    .small {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>STL-PA Alignment Report</h1>
  <p class="muted small">
    Generated at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
  </p>

  <div class="grid">
    <div class="card">
      <h2>Summary</h2>
      <ul>
        <li><b>Candidate pairs:</b> {n_pairs:,}</li>
        <li><b>Samian sites:</b> {n_samian:,}</li>
        <li><b>Avg candidates/site:</b> {avg_cands:.2f}</li>
        <li><b>Confidence counts:</b> {esc(conf_counts)}</li>
      </ul>
      <h3>Parameters</h3>
      <pre>{params_pretty}</pre>
    </div>

    <div class="card">
      <h2>Plots</h2>
      <p class="small muted">Files are referenced relatively (same output folder).</p>
      <h3>Final score distribution</h3>
      <img src="{prefix}_final_score_hist.jpg" alt="Histogram of STL-PA final scores"/>
      <h3>Confidence class counts</h3>
      <img src="{prefix}_confidence_counts.jpg" alt="Bar chart of confidence class counts"/>
      <h3>Top-1 final score distribution</h3>
      <img src="{prefix}_top1_final_score_hist.jpg" alt="Histogram of STL-PA top-1 final scores"/>
      <h3>Top-1: geo contribution vs final score</h3>
      <img src="{prefix}_top1_geo_contrib_vs_final.jpg" alt="Scatter of geo contribution vs final score (top-1)"/>
      <h3>Top-1: string contribution vs final score</h3>
      <img src="{prefix}_top1_string_contrib_vs_final.jpg" alt="Scatter of string contribution vs final score (top-1)"/>
      <h3>Top-1: time contribution vs final score</h3>
      <img src="{prefix}_top1_time_contrib_vs_final.jpg" alt="Scatter of time contribution vs final score (top-1)"/>
      <h3>Top-1 score stack</h3>
      <img src="{prefix}_top1_score_stack.jpg" alt="Stacked contributions (geo/string/time) for top-1 matches"/>

    </div>
  </div>

  <div class="card" style="margin-top:18px;">
    <h2>Top-1 matches (all, sorted by score)</h2>
    <table>
      <thead>
        <tr>
          <th>samian_id</th>
          <th>samian_label</th>
          <th>pleiades_id</th>
          <th>pleiades_label</th>
          <th>distance_km</th>
          <th>geo</th>
          <th>string</th>
          <th>time</th>
          <th>final</th>
          <th>conf</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>

</body>
</html>
"""

            report_path = out_dir / f"{prefix}_report.html"
            report_path.write_text(html, encoding="utf-8")
            log(f"Wrote HTML report: {report_path}")
        except Exception as e:
            log(f"HTML report skipped (failed): {e}")


# ---------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="STL-PA alignment: Pleiades ↔ Samian Research"
    )
    ap.add_argument(
        "--with-time-periods",
        action="store_true",
        help="Compute Pleiades time_period_keys/terms (slow).",
    )
    ap.add_argument(
        "--allow-rough-era",
        action="store_true",
        help="Allow rough Pleiades locations when aggregating era.",
    )
    ap.add_argument(
        "--radius-km",
        type=float,
        default=50.0,
        help="Spatial candidate radius in km (default: 50).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=25,
        help="Max candidates per Samian site (default: 25).",
    )

    ap.add_argument(
        "--adaptive-radius",
        action="store_true",
        help="Enable adaptive search radius: radius_km + adaptive_km_per_year*unc_interval_years.",
    )
    ap.add_argument(
        "--adaptive-km-per-year",
        type=float,
        default=0.5,
        help="Adaptive radius scale in km per year of unc_interval_years (default: 0.5).",
    )
    ap.add_argument(
        "--adaptive-radius-cap-km",
        type=float,
        default=200.0,
        help="Upper cap for adaptive radius in km (default: 200).",
    )
    ap.add_argument(
        "--dynamic-weights",
        action="store_true",
        help='Enable dynamic fusion weights based on Pleiades max_accuracy_m ("2026+" variant).',
    )
    ap.add_argument(
        "--dynamic-acc-km-scale",
        type=float,
        default=20.0,
        help="Accuracy scale (km) where geo-weight approaches dynamic_w_geo_min (default: 20).",
    )
    ap.add_argument(
        "--dynamic-w-geo-min",
        type=float,
        default=0.2,
        help="Minimum geo weight when accuracy is large (default: 0.2).",
    )
    ap.add_argument(
        "--no-html-report",
        action="store_true",
        help="Disable HTML report output.",
    )
    ap.add_argument(
        "--sigma-km",
        type=float,
        default=20.0,
        help="Geo score sigma in km (default: 20).",
    )
    ap.add_argument(
        "--rough-penalty",
        type=float,
        default=0.6,
        help="Penalty factor for rough precision (default: 0.6).",
    )
    ap.add_argument(
        "--w-geo", type=float, default=0.5, help="Weight for geo score (default: 0.5)."
    )
    ap.add_argument(
        "--w-str",
        type=float,
        default=0.3,
        help="Weight for string score (default: 0.3).",
    )
    ap.add_argument(
        "--w-time",
        type=float,
        default=0.2,
        help="Weight for time score (default: 0.2).",
    )
    ap.add_argument(
        "--high-conf",
        type=float,
        default=0.75,
        help="High-confidence threshold (default: 0.75).",
    )
    ap.add_argument(
        "--medium-conf",
        type=float,
        default=0.50,
        help="Medium-confidence threshold (default: 0.50).",
    )
    ap.add_argument(
        "--out-dir", type=str, default="out", help="Output directory (default: ./out)."
    )
    ap.add_argument(
        "--prefix", type=str, default="stlpa", help="Output filename prefix."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Validate files exist early
    for k, pth in FILES.items():
        if k == "time_periods" and not args.with_time_periods:
            continue
        if not pth.exists():
            raise FileNotFoundError(f"Missing required input file: {pth}")

    pleiades_view_df = build_pleiades_view(
        with_time_periods=args.with_time_periods, allow_rough_era=args.allow_rough_era
    )
    samian_df = load_samian_csv(FILES["samian"])

    log(f"Samian ready: {len(samian_df):,} rows")

    params = STLPAParams(
        radius_km=float(args.radius_km),
        topk=int(args.topk),
        sigma_km=float(args.sigma_km),
        rough_penalty=float(args.rough_penalty),
        w_geo=float(args.w_geo),
        w_str=float(args.w_str),
        w_time=float(args.w_time),
        high_conf=float(args.high_conf),
        medium_conf=float(args.medium_conf),
    )

    scored = run_stlpa(pleiades_view_df, samian_df, params)

    out_dir = (BASE_DIR / args.out_dir).resolve()
    write_outputs(scored, out_dir=out_dir, params=params, prefix=str(args.prefix))

    log("Done.")


if __name__ == "__main__":
    main()
