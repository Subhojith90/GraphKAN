from __future__ import annotations


def failed_run_row(profile: str, instance_id: str, model: str, seed: int, stage: str, reason: str, config_hash: str, source_file_hash: str = "") -> dict:
    return {
        "profile": profile, "instance_id": instance_id, "model": model, "seed": seed, "stage": stage,
        "run_status": "failed", "reason": reason, "config_hash": config_hash, "source_file_hash": source_file_hash,
    }
