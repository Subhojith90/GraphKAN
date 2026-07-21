from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


QUARANTINE_RULES = {
    "quarantined_underbudget_pilot": {
        "reason": "Under-budget pilot; excluded from scientific inference.",
        "allowed_use": "Engineering failure provenance only.",
        "headline_excluded": True,
    },
    "quarantined_n10_smoke": {
        "reason": "Failed or incomplete N=10 smoke provenance.",
        "allowed_use": "Engineering provenance only.",
        "headline_excluded": True,
    },
    "n10_fullbudget_smoke_pre_clean_rbm_metadata": {
        "reason": "Pre-clean RBM metadata version.",
        "allowed_use": "Historical provenance only; superseded by clean smoke.",
        "headline_excluded": True,
    },
    "n10_rbm_only_smoke": {
        "reason": "RBM dispatch engineering smoke, not a replicated scientific baseline.",
        "allowed_use": "Dispatch validation only.",
        "headline_excluded": True,
    },
}

HEADLINE_PROFILES = {
    "mechanism_mini",
    "low_detuning_autopsy_fullbudget",
    "n10_primary_cell1_fiveseed",
}

SUPPORTING_PROFILES = {
    "n10_fullbudget_smoke",
    "fullbudget_smoke",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results_stage4d")
    parser.add_argument("--output", default="release/quarantine_manifest.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for profile_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = profile_dir.name
        rule = QUARANTINE_RULES.get(name)

        if rule:
            classification = "QUARANTINED"
            reason = rule["reason"]
            allowed_use = rule["allowed_use"]
            headline_excluded = rule["headline_excluded"]
        elif name in HEADLINE_PROFILES:
            classification = "HEADLINE_EVIDENCE"
            reason = "Registered Stage 4D scientific evidence."
            allowed_use = "Scientific summary and paired analysis."
            headline_excluded = False
        elif name in SUPPORTING_PROFILES:
            classification = "SUPPORTING_ENGINEERING"
            reason = "Smoke or engineering validation profile."
            allowed_use = "Engineering and dispatch evidence."
            headline_excluded = True
        else:
            classification = "UNCLASSIFIED"
            reason = "Not yet classified; exclude from headline aggregation."
            allowed_use = "Manual review required."
            headline_excluded = True

        rows.append(
            {
                "profile_directory": name,
                "classification": classification,
                "headline_excluded": headline_excluded,
                "reason": reason,
                "allowed_use": allowed_use,
            }
        )

        metrics = profile_dir / "raw" / "stage4d_metrics.csv"
        if metrics.exists():
            df = pd.read_csv(metrics)
            status_rows.append(
                {
                    "profile_directory": name,
                    "classification": classification,
                    "rows": len(df),
                    "completed": int((df.get("status") == "COMPLETED").sum())
                    if "status" in df.columns
                    else 0,
                    "exact_valid": int(
                        (df.get("scientific_validity_exact") == "VALID").sum()
                    )
                    if "scientific_validity_exact" in df.columns
                    else 0,
                    "sample_valid": int(
                        (df.get("scientific_validity_sample") == "VALID").sum()
                    )
                    if "scientific_validity_sample" in df.columns
                    else 0,
                    "headline_excluded": headline_excluded,
                }
            )

    pd.DataFrame(rows).to_csv(output, index=False)

    profile_status = output.with_name("stage4d_profile_status.csv")
    pd.DataFrame(status_rows).to_csv(profile_status, index=False)

    print(f"Wrote {output}")
    print(f"Wrote {profile_status}")


if __name__ == "__main__":
    main()
