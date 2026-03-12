"""
Debug the 2022 main_mode pipeline end to end.

Goal:
    Check whether the 2022 mode-choice pipeline is internally consistent from
    raw TP.xlsx through filtering, harmonization, split, evaluation inputs, and
    saved prediction files.

This script operationalizes the checklist in
    TRAIL_bug_checklist_main_mode.md

Usage:
    python scripts/debug_mode_pipeline.py
    python scripts/debug_mode_pipeline.py --fewshot_ratio 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trail.data.filters import apply_all_filters, filter_2022_main_modes
from trail.data.harmonizer import harmonize_2022
from trail.data.loader import load_tcs2022_raw
from trail.data.schema import MODE_LABELS
from trail.data.splitter import split_2022
from trail.utils.config import PROJECT_ROOT, load_data_config


DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug" / "mode_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug TCS 2022 main_mode pipeline")
    parser.add_argument("--fewshot_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_series_counts(series: pd.Series, out_path: Path, normalize: bool = False) -> pd.DataFrame:
    counts = (
        series.value_counts(dropna=False, normalize=normalize)
        .sort_index()
        .rename("share" if normalize else "count")
        .reset_index()
    )
    counts.columns = ["value", "share" if normalize else "count"]
    counts.to_csv(out_path, index=False, encoding="utf-8-sig")
    return counts


def summarize_series(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().tolist()
    return {
        "dtype": str(series.dtype),
        "n_total": int(len(series)),
        "n_null": int(series.isna().sum()),
        "unique_sorted": sorted(pd.Series(values).drop_duplicates().tolist()) if values else [],
        "head_20": series.head(20).tolist(),
    }


def detect_mode_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "mode" in c.lower()]


def assert_no_illegal_raw_modes(df: pd.DataFrame, mode_col: str) -> list[str]:
    issues: list[str] = []
    illegal = {10, 99, 999}
    numeric = pd.to_numeric(df[mode_col], errors="coerce").dropna().astype(int)
    present = sorted(set(numeric).intersection(illegal))
    if present:
        issues.append(f"Filtered raw data still contains illegal Main_mode values: {present}")
    if not numeric.empty and numeric.max() > 9:
        issues.append("Filtered raw data has Main_mode > 9")
    return issues


def build_joined_from_filtered(
    df_hh: pd.DataFrame,
    df_hm: pd.DataFrame,
    df_tp_filtered: pd.DataFrame,
    cfg_2022: dict[str, Any],
) -> pd.DataFrame:
    hh_cfg = cfg_2022["hh_columns"]
    hm_cfg = cfg_2022["hm_columns"]
    tp_cfg = cfg_2022["tp_columns"]

    hh_id = hh_cfg["household_id"]

    car_cols = [c for c in hh_cfg["car_cols"] if c in df_hh.columns]
    hh_sel_cols = [hh_id, hh_cfg["hh_type"], hh_cfg["hh_weight"], hh_cfg.get("hh_size", "A3")] + car_cols
    hh_sel_cols = [c for c in hh_sel_cols if c in df_hh.columns]
    df_hh_sel = df_hh[hh_sel_cols].copy()

    if car_cols:
        df_hh_sel["_car_count"] = df_hh_sel[car_cols].fillna(0).sum(axis=1)
        df_hh_sel["car_availability_raw"] = df_hh_sel["_car_count"].apply(
            lambda x: 0 if x == 0 else (1 if x == 1 else 2)
        )
        df_hh_sel.drop(columns=["_car_count"] + car_cols, inplace=True)
    else:
        df_hh_sel["car_availability_raw"] = 0

    hm_sel_cols = [
        hm_cfg["household_id"],
        hm_cfg["member_id"],
        hm_cfg["sex"],
        hm_cfg["age"],
        hm_cfg["employment"],
        hm_cfg["member_weight"],
    ]
    hm_sel_cols = [c for c in hm_sel_cols if c in df_hm.columns]
    df_hm_sel = df_hm[hm_sel_cols].copy()

    tp_core = [
        tp_cfg["household_id"],
        tp_cfg["member_id"],
        tp_cfg["trip_no"],
        tp_cfg["trip_purpose"],
        tp_cfg["main_mode"],
        tp_cfg["journey_time"],
        tp_cfg["depart_time"],
        tp_cfg["time_period"],
        tp_cfg["trip_weight"],
        tp_cfg.get("origin_zone", "O_26PDD"),
        tp_cfg.get("dest_zone", "D_26PDD"),
    ]
    tp_core = [c for c in tp_core if c in df_tp_filtered.columns]
    df_tp_sel = df_tp_filtered[tp_core].copy()

    df_joined = df_tp_sel.merge(
        df_hm_sel,
        on=[tp_cfg["household_id"], tp_cfg["member_id"]],
        how="left",
    )
    df_joined = df_joined.merge(df_hh_sel, on=hh_id, how="left")
    return df_joined


def save_distribution(name: str, series: pd.Series, out_dir: Path) -> dict[str, Any]:
    series = series.rename(name)
    counts_path = out_dir / f"{name}_counts.csv"
    shares_path = out_dir / f"{name}_shares.csv"
    counts_df = save_series_counts(series, counts_path, normalize=False)
    shares_df = save_series_counts(series, shares_path, normalize=True)
    return {
        "counts_file": str(counts_path),
        "shares_file": str(shares_path),
        "counts_preview": counts_df.to_dict(orient="records"),
        "shares_preview": shares_df.to_dict(orient="records"),
    }


def inspect_prediction_files(out_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    summaries: list[dict[str, Any]] = []
    issues: list[str] = []

    if not pred_dir.exists():
        return summaries, issues

    for pred_path in sorted(pred_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pred_path)
        except Exception as exc:
            summaries.append({
                "file": pred_path.name,
                "rows": None,
                "true_unique": [],
                "pred_unique": [],
                "read_error": str(exc),
            })
            issues.append(f"{pred_path.name}: could not be read as parquet ({exc})")
            continue

        true_unique = sorted(df["main_mode"].dropna().astype(int).unique().tolist()) if "main_mode" in df.columns else []
        pred_unique = (
            sorted(df["predicted_mode"].dropna().astype(int).unique().tolist())
            if "predicted_mode" in df.columns
            else []
        )
        summary = {
            "file": pred_path.name,
            "rows": int(len(df)),
            "true_unique": true_unique,
            "pred_unique": pred_unique,
        }
        summaries.append(summary)

        if "predicted_mode" in df.columns:
            invalid = sorted(set(pred_unique).difference({1, 2, 3, 4, 5}))
            if invalid:
                issues.append(f"{pred_path.name}: predicted_mode has out-of-schema labels {invalid}")

        if "main_mode" in df.columns:
            invalid_true = sorted(set(true_unique).difference({1, 2, 3, 4, 5}))
            if invalid_true:
                issues.append(f"{pred_path.name}: main_mode has out-of-schema labels {invalid_true}")

        if "main_mode" in df.columns:
            save_series_counts(df["main_mode"], out_dir / f"{pred_path.stem}_true_counts.csv")
        if "predicted_mode" in df.columns:
            save_series_counts(df["predicted_mode"], out_dir / f"{pred_path.stem}_pred_counts.csv")

    return summaries, issues


def main() -> None:
    args = parse_args()
    ensure_dir(DEBUG_DIR)

    cfg_2022 = load_data_config("tcs2022")
    harm_cfg = load_data_config("harmonization")
    tp_mode_col = cfg_2022["tp_columns"]["main_mode"]

    findings: list[str] = []
    notes: list[str] = []

    df_hh, df_hm, df_tp = load_tcs2022_raw(cfg_2022)
    raw_mode_series = df_tp[tp_mode_col]
    raw_numeric = pd.to_numeric(raw_mode_series, errors="coerce")

    raw_info = {
        "tp_columns": df_tp.columns.tolist(),
        "candidate_mode_columns": detect_mode_columns(df_tp),
        "main_mode_summary": summarize_series(raw_mode_series),
        "main_mode_numeric_summary": summarize_series(raw_numeric),
    }
    write_text(DEBUG_DIR / "raw_mode_metadata.json", json.dumps(raw_info, ensure_ascii=False, indent=2))
    save_distribution("raw_main_mode", raw_numeric, DEBUG_DIR)

    df_tp_numeric = df_tp.copy()
    df_tp_numeric[tp_mode_col] = pd.to_numeric(df_tp_numeric[tp_mode_col], errors="coerce")
    save_distribution("raw_main_mode_numeric", df_tp_numeric[tp_mode_col], DEBUG_DIR)

    df_tp_filtered = filter_2022_main_modes(df_tp_numeric, mode_col=tp_mode_col, max_mode=9)
    save_distribution("after_filter_main_mode", df_tp_filtered[tp_mode_col], DEBUG_DIR)

    findings.extend(assert_no_illegal_raw_modes(df_tp_filtered, tp_mode_col))

    raw_illegal = sorted(set(raw_numeric.dropna().astype(int).unique()).intersection({10, 99, 999}))
    notes.append(f"Raw Main_mode includes excluded values before filtering: {raw_illegal}")
    notes.append(
        "Current code keeps Main_mode <= 9. Under the existing mapping, "
        "Main_mode 7 -> Other and Main_mode 9 -> Walking, so those classes are expected after harmonization."
    )

    df_joined = build_joined_from_filtered(df_hh, df_hm, df_tp_filtered, cfg_2022)
    df_harm = harmonize_2022(df_joined, cfg_2022, harm_cfg)
    save_distribution("after_harmonize_unified_main_mode", df_harm["main_mode"], DEBUG_DIR)

    mapping_df = (
        pd.DataFrame(
            {
                "raw_Main_mode": pd.to_numeric(df_joined[tp_mode_col], errors="coerce"),
                "unified_main_mode": pd.to_numeric(df_harm["main_mode"], errors="coerce"),
            }
        )
        .drop_duplicates()
        .sort_values(["raw_Main_mode", "unified_main_mode"])
        .reset_index(drop=True)
    )
    mapping_df["unified_label"] = mapping_df["unified_main_mode"].map(MODE_LABELS)
    mapping_df.to_csv(DEBUG_DIR / "raw_to_unified_mode_mapping.csv", index=False, encoding="utf-8-sig")

    raw_values_after_filter = set(pd.to_numeric(df_tp_filtered[tp_mode_col], errors="coerce").dropna().astype(int).unique())
    if not raw_values_after_filter.issubset(set(range(1, 10))):
        findings.append("Filtered raw TP contains values outside 1..9; filter or dtype handling is inconsistent.")

    if 9 in raw_values_after_filter and 4 in set(df_harm["main_mode"].dropna().astype(int).unique()):
        notes.append("Walking appears because raw Main_mode=9 survives the <=9 filter and maps to unified mode 4.")
    if 7 in raw_values_after_filter and 5 in set(df_harm["main_mode"].dropna().astype(int).unique()):
        notes.append("Other appears because raw Main_mode=7 survives the <=9 filter and maps to unified mode 5.")

    df_clean = apply_all_filters(df_harm, survey_year=2022)
    save_distribution("after_apply_all_filters_main_mode", df_clean["main_mode"], DEBUG_DIR)

    processed_path = PROJECT_ROOT / "data/interim/harmonized/tcs2022_harmonized.parquet"
    processed_summary: dict[str, Any] = {"exists": processed_path.exists()}
    if processed_path.exists():
        df_processed = pd.read_parquet(processed_path)
        save_distribution("cached_processed_main_mode", df_processed["main_mode"], DEBUG_DIR)
        processed_summary.update(
            {
                "rows": int(len(df_processed)),
                "unique_main_mode": sorted(df_processed["main_mode"].dropna().astype(int).unique().tolist()),
                "fresh_rows": int(len(df_clean)),
                "fresh_unique_main_mode": sorted(df_clean["main_mode"].dropna().astype(int).unique().tolist()),
                "row_count_matches_fresh": int(len(df_processed)) == int(len(df_clean)),
                "label_set_matches_fresh": sorted(df_processed["main_mode"].dropna().astype(int).unique().tolist())
                == sorted(df_clean["main_mode"].dropna().astype(int).unique().tolist()),
            }
        )
        if not processed_summary["row_count_matches_fresh"] or not processed_summary["label_set_matches_fresh"]:
            findings.append("Cached harmonized parquet differs from fresh recomputation; stale cache is possible.")
    write_text(DEBUG_DIR / "processed_comparison.json", json.dumps(processed_summary, ensure_ascii=False, indent=2))

    split = split_2022(
        df_clean,
        test_ratio=args.test_ratio,
        fewshot_ratio=args.fewshot_ratio,
        seed=args.seed,
    )
    for split_name, split_df in split.items():
        save_distribution(f"{split_name}_main_mode", split_df["main_mode"], DEBUG_DIR)

    split_sets = {name: sorted(df["main_mode"].dropna().astype(int).unique().tolist()) for name, df in split.items()}
    for split_name, labels in split_sets.items():
        if set(labels).difference({1, 2, 3, 4, 5}):
            findings.append(f"{split_name} split contains out-of-schema labels: {labels}")

    pred_summaries, pred_issues = inspect_prediction_files(DEBUG_DIR)
    findings.extend(pred_issues)
    write_text(DEBUG_DIR / "prediction_file_summary.json", json.dumps(pred_summaries, ensure_ascii=False, indent=2))

    readme_path = PROJECT_ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if "Main_mode <= 9" in readme_text and ("Walking" in readme_text or "Other" in readme_text):
        notes.append(
            "README simultaneously states 'Main_mode <= 9' and reports a 5-class target with Walking/Other. "
            "That is internally consistent with the current mapping, but inconsistent with a strict mechanised-only interpretation."
        )

    raw_counts = save_series_counts(raw_numeric, DEBUG_DIR / "raw_main_mode_counts_snapshot.csv")
    filtered_counts = save_series_counts(df_tp_filtered[tp_mode_col], DEBUG_DIR / "filtered_main_mode_counts_snapshot.csv")
    unified_counts = save_series_counts(df_clean["main_mode"], DEBUG_DIR / "clean_unified_main_mode_counts_snapshot.csv")

    summary = {
        "raw_mode_column": tp_mode_col,
        "raw_candidate_mode_columns": detect_mode_columns(df_tp),
        "raw_unique_main_mode": sorted(raw_numeric.dropna().astype(int).unique().tolist()),
        "filtered_unique_main_mode": sorted(df_tp_filtered[tp_mode_col].dropna().astype(int).unique().tolist()),
        "unified_unique_main_mode": sorted(df_clean["main_mode"].dropna().astype(int).unique().tolist()),
        "mode_mapping": mapping_df.to_dict(orient="records"),
        "raw_counts": raw_counts.to_dict(orient="records"),
        "filtered_counts": filtered_counts.to_dict(orient="records"),
        "unified_counts": unified_counts.to_dict(orient="records"),
        "split_label_sets": split_sets,
        "prediction_files": pred_summaries,
        "findings": findings,
        "notes": notes,
        "likely_root_cause": (
            "The current pipeline is not mechanised-only. It filters out values > 9, "
            "but still retains raw Main_mode 7 and 9. These map to unified Other and Walking."
        ),
    }
    write_text(DEBUG_DIR / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    report_lines = [
        "# TRAIL Main Mode Pipeline Debug Report",
        "",
        "## Key conclusion",
        "",
        "The current code path is **not** a strict mechanised-only task.",
        "It filters raw `Main_mode` to `<= 9`, then maps:",
        "- raw `7` -> unified `5 (Other)`",
        "- raw `9` -> unified `4 (Walking)`",
        "",
        "So if `Walking` and `Other` appear after harmonization, that is expected under the current implementation.",
        "The likely issue is a **task-definition / README wording mismatch**, not necessarily a broken filter.",
        "",
        "## Findings",
        "",
    ]

    if findings:
        report_lines.extend([f"- {item}" for item in findings])
    else:
        report_lines.append("- No hard pipeline break was detected in the checked stages.")

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            *[f"- {item}" for item in notes],
            "",
            "## Artifacts",
            "",
            f"- Raw counts: `{(DEBUG_DIR / 'raw_main_mode_counts_snapshot.csv').relative_to(PROJECT_ROOT)}`",
            f"- Filtered counts: `{(DEBUG_DIR / 'filtered_main_mode_counts_snapshot.csv').relative_to(PROJECT_ROOT)}`",
            f"- Unified counts: `{(DEBUG_DIR / 'clean_unified_main_mode_counts_snapshot.csv').relative_to(PROJECT_ROOT)}`",
            f"- Raw-to-unified mapping: `{(DEBUG_DIR / 'raw_to_unified_mode_mapping.csv').relative_to(PROJECT_ROOT)}`",
            f"- Full summary: `{(DEBUG_DIR / 'summary.json').relative_to(PROJECT_ROOT)}`",
        ]
    )
    write_text(DEBUG_DIR / "debug_report.md", "\n".join(report_lines))

    print("Debug artifacts written to:", DEBUG_DIR)
    print("Likely root cause:")
    print(summary["likely_root_cause"])
    if findings:
        print("Findings:")
        for item in findings:
            print("-", item)


if __name__ == "__main__":
    main()

