# samianresearch-stlpa

* Working Paper: https://github.com/squirrelpapers/SqP-8-1-1-SamianResearchSites--STL-PA

## Introduction

**samianresearch-stlpa** implements *STL‑PA* (Spatio‑Temporal‑Lexical Place Alignment), a semi‑automatic method for aligning archaeological site references from the *SamianResearch* dataset with places in the *Pleiades* gazetteer.

The tool is designed for research contexts where place references are uncertain, fuzzy, or historically ambiguous. Instead of relying on a single criterion, STL‑PA combines **spatial proximity**, **lexical similarity**, and **temporal overlap** into a transparent, weighted scoring model. The workflow supports evaluation against manually curated links and produces both machine‑readable outputs and human‑readable reports.

The primary use case is the reconciliation of Roman‑period Samian Ware production and distribution sites with canonical Pleiades place identifiers, but the approach is generic and reusable.

---

## Method (STL‑PA)

STL‑PA operates in four main stages, implemented in `stlpa_run.py`:

1. **Normalisation and View Building**  
   Input CSV files are normalised into internal *views*:
   - a Samian site view (labels, coordinates, temporal spans, uncertainty)
   - a Pleiades place view (titles, alternative names, coordinates, temporal coverage, spatial accuracy)

2. **Candidate Generation (Spatial)**  
   For each Samian site, potential Pleiades candidates are selected using a spatial radius around the Samian coordinates.  
   The radius can be *adaptive*, expanding with increasing temporal uncertainty. A Top‑K fallback ensures candidates are available even in dense regions.

3. **Scoring per Candidate**  
   Each Samian–Pleiades pair is scored along three dimensions:
   - **Geo score**: exponential decay based on haversine distance, corrected for Pleiades spatial accuracy and penalised for “rough” locations.
   - **String score**: maximum lexical similarity between labels and alternative names.
   - **Time score**: normalised overlap of temporal intervals, including uncertainty, weighted by a temporal confidence factor.

   The final score is a weighted fusion of these components. Weights are normalised and can be dynamically adjusted (e.g. reduced geographic weight for imprecise locations).

4. **Ranking and Evaluation**  
   Candidates are ranked per Samian site, confidence classes are assigned, and (if available) manual Samian→Pleiades links are injected for evaluation and quality control.

---

## Data

### SamianResearch
Input file:
- `samianresearch.csv`

Contains Samian Ware sites with identifiers, labels, coordinates, chronological information, and uncertainty estimates. An optional file,
- `samianresearch_pleiades.csv`,
provides manually curated links to Pleiades for evaluation.

### Pleiades
Input files:
- `places.csv`
- `names.csv`
- `location_points.csv`
- `places_accuracy.csv`
- `time_periods.csv` (optional, for temporal normalisation)

These files are combined into a single Pleiades view, including alternative names, spatial coordinates, temporal coverage, and maximum spatial accuracy.

---

## Output

All results are written to the `out/` directory. Core outputs include:

- `stlpa_candidates_scored.csv`  
  All evaluated Samian–Pleiades candidate pairs with individual score components and final scores.

- `stlpa_top1.csv`  
  The top‑ranked Pleiades match per Samian site.

- `stlpa_summary.json`  
  Structured per‑site summaries, including parameters and ranked candidates.

- `stlpa_summary.html`  
  Human‑readable rendering of the summary JSON for inspection and review.

- `stlpa_report.html`  
  Compact analytical report focusing on Top‑1 results, including contribution plots for geo, string, and time scores.

If manual mappings are provided, additional evaluation files (`stlpa_manual_eval.*`) are generated.

---

## Why STL‑PA?

STL‑PA makes the alignment process:
- **transparent** (explicit scores and weights),
- **reproducible** (CSV‑based inputs and deterministic scoring),
- **evaluatable** (manual links integrated as first‑class citizens),
- **adaptable** (weights, thresholds, and uncertainty handling are configurable).

This makes it suitable both for exploratory research and for methodologically explicit publications in computational archaeology and digital gazetteer studies.
