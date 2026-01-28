#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pleiades–Samian Alignment Preparation Script
===========================================

This script builds two normalised DataFrames:

1. pleiades_view_df
   - one row per Pleiades place
   - geometry (representative lat/lon)
   - alternative labels aggregated from names.csv
   - temporal extent aggregated from location_points.csv
   - spatial accuracy metadata

2. samian_df
   - one row per Samian Research site
   - geometry (lat/lon)
   - temporal extent with quality and uncertainty measures

All CSV input files are expected in the folder:

    ./src/

This script intentionally ignores place types, as the available
place_types.csv is a vocabulary only and not a place→type mapping.

The resulting DataFrames are intended as input for a semi-automatic
alignment algorithm (spatial + string + temporal similarity).
"""

import time
import pandas as pd
from pathlib import Path

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
    "samian": SRC_DIR / "samianresearch.csv",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------
# Build Pleiades view
# ---------------------------------------------------------------------


def build_pleiades_view() -> pd.DataFrame:
    t0 = time.time()
    log("Building Pleiades view...")

    # Load core tables
    log("Loading places.csv ...")
    places = pd.read_csv(FILES["places"])
    log(f"  places: {len(places):,}")

    log("Loading names.csv ...")
    names = pd.read_csv(FILES["names"])
    log(f"  names: {len(names):,}")

    log("Loading location_points.csv ...")
    loc_pts = pd.read_csv(FILES["location_points"])
    log(f"  location_points: {len(loc_pts):,}")

    log("Loading places_accuracy.csv ...")
    acc = pd.read_csv(FILES["places_accuracy"])
    log(f"  places_accuracy: {len(acc):,}")

    # -----------------------------------------------------------------
    # Core places table
    # -----------------------------------------------------------------

    core = places[
        [
            "id",
            "uri",
            "title",
            "representative_latitude",
            "representative_longitude",
            "location_precision",
            "bounding_box_wkt",
        ]
    ].rename(
        columns={
            "id": "place_id",
            "title": "label",
            "representative_latitude": "lat",
            "representative_longitude": "lon",
        }
    )

    # -----------------------------------------------------------------
    # Alternative labels
    # -----------------------------------------------------------------

    log("Aggregating alt labels (names.csv) ...")
    t = time.time()

    names["name"] = (
        names["attested_form"].fillna(names["romanized_form_1"]).fillna(names["title"])
    )

    alt_labels = (
        names.groupby("place_id")["name"]
        .apply(lambda x: "|".join(sorted(set(str(v) for v in x if pd.notna(v)))))
        .reset_index()
        .rename(columns={"name": "alt_labels"})
    )

    core = core.merge(alt_labels, on="place_id", how="left")
    log(f"  alt labels: {len(alt_labels):,} (t={time.time() - t:.1f}s)")

    # -----------------------------------------------------------------
    # Temporal extent from location_points
    # -----------------------------------------------------------------

    log("Aggregating era from location_points.csv ...")
    t = time.time()

    era = (
        loc_pts.groupby("place_id")
        .agg(
            earliest_year=("year_after_which", "min"),
            latest_year=("year_before_which", "max"),
        )
        .reset_index()
    )

    core = core.merge(era, on="place_id", how="left")
    log(f"  era rows: {len(era):,} (t={time.time() - t:.1f}s)")

    # -----------------------------------------------------------------
    # Accuracy metadata
    # -----------------------------------------------------------------

    log("Loading accuracy info ...")
    acc_sel = acc[
        [
            "place_id",
            "min_accuracy_meters",
            "max_accuracy_meters",
            "accuracy_hull",
        ]
    ]

    core = core.merge(acc_sel, on="place_id", how="left")

    log(f"Pleiades view ready: {len(core):,} rows (total t={time.time() - t0:.1f}s)")
    return core


# ---------------------------------------------------------------------
# Load Samian Research data
# ---------------------------------------------------------------------


def load_samian() -> pd.DataFrame:
    log("Loading Samian research CSV...")
    df = pd.read_csv(FILES["samian"])

    df = df.rename(
        columns={
            "id": "samian_id",
            "altlabels": "alt_labels",
        }
    )

    log(f"Samian ready: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    build_pleiades_view()
    load_samian()
    log("Done.")


if __name__ == "__main__":
    main()
