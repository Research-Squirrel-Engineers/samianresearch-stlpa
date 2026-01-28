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
    radius_km: float = 50.0
    topk: int = 25
    sigma_km: float = 20.0
    rough_penalty: float = 0.6
    w_geo: float = 0.5
    w_str: float = 0.3
    w_time: float = 0.2
    high_conf: float = 0.75
    medium_conf: float = 0.50


def normalise_weights(p: STLPAParams) -> Tuple[float, float, float]:
    s = max(1e-9, p.w_geo + p.w_str + p.w_time)
    return p.w_geo / s, p.w_str / s, p.w_time / s


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

            final = (w_geo * geo) + (w_str * str_sc) + (w_time * t_sc)

            candidates.append((j, d_km, d_eff, geo, str_sc, t_sc, final))

        candidates.sort(key=lambda x: x[-1], reverse=True)

        for rank, (j, d_km, d_eff, geo, str_sc, t_sc, final) in enumerate(
            candidates, start=1
        ):
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

    except Exception as e:
        log(f"Plotting skipped (matplotlib not available or failed): {e}")


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
