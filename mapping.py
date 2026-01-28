#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pleiades + Samian loader for alignment (fast, no place_types).

What it does
- Loads Pleiades CSVs from ./pleiades/
- Consolidates them into `pleiades_view_df` (id/uri/label/alt_labels/coords/bbox/precision/accuracy/era)
- Loads Samian CSV from ./samianresearch.csv into `samian_df`

Notes on performance
- Computing "time_period_keys/terms" by overlapping every place interval with every time_period
  is O(N*M) and can be slow for the full Pleiades dump. Therefore it is OFF by default.
  Enable with:  python mapping.py --with-time-periods
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PLEIADES_DIR = BASE_DIR / "pleiades"
SAMIAN_PATH = BASE_DIR / "samianresearch.csv"

FILES = {
    "places": "places.csv",
    "names": "names.csv",
    "loc_points": "location_points.csv",
    "loc_polygons": "location_polygons.csv",  # optional, not required
    "places_accuracy": "places_accuracy.csv",
    "time_periods": "time_periods.csv",
}

ERA_LOCATION_FILTER = {
    "association_certainty": {"certain", "probable"},
    "location_precision": {"precise"},
}
ALLOW_ROUGH_LOCATIONS_FOR_ERA = False


# -----------------------------
# Helpers
# -----------------------------
_BC_RE = re.compile(r"^\s*(\d+)\s*BC\s*$", re.IGNORECASE)
_AD_RE = re.compile(r"^\s*(\d+)\s*(AD|CE)?\s*$", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, flush=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    # Normalize column names (BOM/whitespace/case/separators)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


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


def interval_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    a0, a1 = a
    b0, b1 = b
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0, right - left)


# -----------------------------
# Pleiades builders
# -----------------------------
def build_places_core(places_df: pd.DataFrame) -> pd.DataFrame:
    core = places_df.copy()
    core["pleiades_id"] = core["id"].astype(str).str.strip()
    core["pleiades_uri"] = core["uri"].astype(str).str.strip()
    core["label"] = core["title"].astype(str).str.strip()
    core["latitude"] = core.get("representative_latitude", "").apply(to_float)
    core["longitude"] = core.get("representative_longitude", "").apply(to_float)
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
    """
    Aggregates name variants per place_id into a '|' separated string.

    This is designed to be fast on the full dump:
    - filter by association_certainty early
    - melt only required columns
    - groupby once
    """
    df = names_df.copy()
    if "place_id" not in df.columns:
        # Some dumps might use "placeid"
        if "placeid" in df.columns:
            df = df.rename(columns={"placeid": "place_id"})
        else:
            return pd.DataFrame(columns=["pleiades_id", "alt_labels"])

    df["place_id"] = df["place_id"].astype(str).str.strip()

    if "association_certainty" in df.columns:
        cert = df["association_certainty"].astype(str).str.strip().str.lower()
        df = df[cert.isin({"", "certain", "probable"})].copy()

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

    long = df[["place_id"] + cols].melt(
        id_vars=["place_id"], value_vars=cols, value_name="name"
    )
    long["name"] = long["name"].astype(str).str.strip()
    long = long[long["name"] != ""]
    long = long.drop_duplicates(subset=["place_id", "name"])

    agg = (
        long.groupby("place_id")["name"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .reset_index()
        .rename(columns={"place_id": "pleiades_id", "name": "alt_labels"})
    )
    return agg


def build_accuracy(acc_df: pd.DataFrame) -> pd.DataFrame:
    df = acc_df.copy()
    if "place_id" not in df.columns:
        if "placeid" in df.columns:
            df = df.rename(columns={"placeid": "place_id"})
        else:
            return pd.DataFrame(
                columns=[
                    "pleiades_id",
                    "min_accuracy_m",
                    "max_accuracy_m",
                    "accuracy_hull_wkt",
                    "location_precision_accuracy",
                ]
            )

    def fnum(x):
        try:
            v = float(str(x).strip())
            return None if v < 0 else v
        except Exception:
            return None

    df["pleiades_id"] = df["place_id"].astype(str).str.strip()
    df["min_accuracy_m"] = df.get("min_accuracy_meters", "").apply(fnum)
    df["max_accuracy_m"] = df.get("max_accuracy_meters", "").apply(fnum)
    df["accuracy_hull_wkt"] = df.get("accuracy_hull", "").astype(str)
    df["location_precision_accuracy"] = (
        df.get("location_precision", "").astype(str).str.strip().str.lower()
    )

    return df[
        [
            "pleiades_id",
            "location_precision_accuracy",
            "accuracy_hull_wkt",
            "min_accuracy_m",
            "max_accuracy_m",
        ]
    ].drop_duplicates(subset=["pleiades_id"])


def build_era_from_locations(loc_points_df: pd.DataFrame) -> pd.DataFrame:
    df = loc_points_df.copy()
    if "place_id" not in df.columns:
        if "placeid" in df.columns:
            df = df.rename(columns={"placeid": "place_id"})
        else:
            return pd.DataFrame(
                columns=["pleiades_id", "earliest_year", "latest_year", "era_source"]
            )

    df["place_id"] = df["place_id"].astype(str).str.strip()
    df["association_certainty"] = (
        df.get("association_certainty", "").astype(str).str.strip().str.lower()
    )
    df["location_precision"] = (
        df.get("location_precision", "").astype(str).str.strip().str.lower()
    )

    cert_ok = df["association_certainty"].isin(
        ERA_LOCATION_FILTER["association_certainty"]
    )
    if ALLOW_ROUGH_LOCATIONS_FOR_ERA:
        prec_ok = df["location_precision"].isin({"precise", "rough"})
    else:
        prec_ok = df["location_precision"].isin(
            ERA_LOCATION_FILTER["location_precision"]
        )

    df = df[cert_ok & prec_ok].copy()

    df["tpq"] = df.get("year_after_which", "").apply(parse_time_bound)
    df["taq"] = df.get("year_before_which", "").apply(parse_time_bound)
    df = df[(df["tpq"].notna()) | (df["taq"].notna())].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["pleiades_id", "earliest_year", "latest_year", "era_source"]
        )

    grouped = (
        df.groupby("place_id")
        .agg(
            earliest_year=(
                "tpq",
                lambda s: (
                    int(min([v for v in s.tolist() if v is not None]))
                    if any(v is not None for v in s.tolist())
                    else None
                ),
            ),
            latest_year=(
                "taq",
                lambda s: (
                    int(max([v for v in s.tolist() if v is not None]))
                    if any(v is not None for v in s.tolist())
                    else None
                ),
            ),
        )
        .reset_index()
        .rename(columns={"place_id": "pleiades_id"})
    )
    grouped["era_source"] = "location_points"
    return grouped


def build_time_periods_vocab(periods_df: pd.DataFrame) -> pd.DataFrame:
    df = periods_df.copy()
    df["key"] = df["key"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    df["lower_i"] = df.get("lower_bound", "").apply(parse_time_bound)
    df["upper_i"] = df.get("upper_bound", "").apply(parse_time_bound)
    df = df[df["lower_i"].notna() & df["upper_i"].notna()].copy()
    df["lower_i"] = df["lower_i"].astype(int)
    df["upper_i"] = df["upper_i"].astype(int)
    return df[["key", "term", "lower_i", "upper_i"]]


def add_time_periods(df: pd.DataFrame, periods_vocab: pd.DataFrame) -> pd.DataFrame:
    """
    Slow-ish by nature (interval overlap across many places).
    We keep it optional and only compute for rows that actually have both bounds.
    """
    out = df.copy()
    out["time_period_keys"] = ""
    out["time_period_terms"] = ""

    # pre-read periods into python lists for speed
    periods = [
        (
            r["key"],
            r["term"],
            min(r["lower_i"], r["upper_i"]),
            max(r["lower_i"], r["upper_i"]),
        )
        for _, r in periods_vocab.iterrows()
    ]

    mask = out["earliest_year"].notna() & out["latest_year"].notna()
    idxs = out.index[mask].tolist()
    total = len(idxs)
    if total == 0:
        return out

    log(
        f"Computing time-period overlaps for {total:,} places (this can take a while)..."
    )
    t0 = time.time()
    for i, idx in enumerate(idxs, 1):
        start = int(out.at[idx, "earliest_year"])
        end = int(out.at[idx, "latest_year"])
        if start > end:
            start, end = end, start

        ks, ts = [], []
        for k, term, p0, p1 in periods:
            if interval_overlap((start, end), (p0, p1)) > 0:
                ks.append(k)
                ts.append(term)
        out.at[idx, "time_period_keys"] = "|".join(ks)
        out.at[idx, "time_period_terms"] = "|".join(ts)

        if i % 5000 == 0:
            log(f"  ...{i:,}/{total:,} done")

    log(f"Time-period mapping done in {time.time() - t0:.1f}s")
    return out


def build_pleiades_view(with_time_periods: bool) -> pd.DataFrame:
    t_all = time.time()

    log("Loading places.csv ...")
    places_df = read_csv(PLEIADES_DIR / FILES["places"])
    log(f"  places: {len(places_df):,}")

    log("Loading names.csv ...")
    names_df = read_csv(PLEIADES_DIR / FILES["names"])
    log(f"  names: {len(names_df):,}")

    log("Loading location_points.csv ...")
    loc_points_df = read_csv(PLEIADES_DIR / FILES["loc_points"])
    log(f"  location_points: {len(loc_points_df):,}")

    log("Loading places_accuracy.csv ...")
    acc_df = read_csv(PLEIADES_DIR / FILES["places_accuracy"])
    log(f"  places_accuracy: {len(acc_df):,}")

    # build blocks
    log("Building core places view ...")
    places_core = build_places_core(places_df)

    log("Aggregating alt labels (names.csv) ...")
    t0 = time.time()
    alt_labels = build_alt_labels(names_df)
    log(f"  alt labels: {len(alt_labels):,} (t={time.time()-t0:.1f}s)")

    log("Aggregating era from location_points.csv ...")
    t0 = time.time()
    era = build_era_from_locations(loc_points_df)
    log(f"  era rows: {len(era):,} (t={time.time()-t0:.1f}s)")

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

    # optional time periods
    if with_time_periods:
        periods_df = read_csv(PLEIADES_DIR / FILES["time_periods"])
        periods_vocab = build_time_periods_vocab(periods_df)
        df = add_time_periods(df, periods_vocab)
    else:
        df["time_period_keys"] = ""
        df["time_period_terms"] = ""

    out = df[
        [
            "pleiades_id",
            "pleiades_uri",
            "label",
            "alt_labels",
            "latitude",
            "longitude",
            "bounding_box_wkt",
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

    log(f"Pleiades view ready: {len(out):,} rows (total t={time.time()-t_all:.1f}s)")
    return out


# -----------------------------
# Samian loader
# -----------------------------
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
            f"Samian CSV missing columns: {missing}. Found: {list(df.columns)}"
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


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--with-time-periods",
        action="store_true",
        help="Compute time_period_keys/terms (slow).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    log("Building Pleiades view...")
    pleiades_view_df = build_pleiades_view(with_time_periods=args.with_time_periods)

    log("\nLoading Samian research CSV...")
    if not SAMIAN_PATH.exists():
        raise FileNotFoundError(f"Samian file not found: {SAMIAN_PATH}")
    samian_df = load_samian_csv(SAMIAN_PATH)
    log(f"Samian ready: {len(samian_df):,} rows")

    # Expose for debugging in VS Code "Python: Debug Console"
    globals()["pleiades_view_df"] = pleiades_view_df
    globals()["samian_df"] = samian_df

    log("\nDone.")


if __name__ == "__main__":
    main()
