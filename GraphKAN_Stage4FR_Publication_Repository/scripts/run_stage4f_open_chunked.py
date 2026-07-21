from pathlib import Path
import copy
import os
import signal
import subprocess
import sys
import time

import pandas as pd
import yaml

BASE_CONFIG = Path("configs/netket_stage4f_boundary_patch_open.yaml")
TMP_DIR = Path("configs/_stage4f_chunks/open")
LOG_DIR = Path("logs/stage4f_chunks/open")
METRICS = Path("results_stage4f/boundary_patch/open/raw/stage4e_metrics.csv")

DELTAS = [1.25, 1.5, 1.75, 2.0]
C6S = [3, 4, 5]

MODELS = [
    "rbf_radial_b32",
    "endpoint_sum_b32",
    "endpoint_product_b32",
    "endpoint_hybrid_b32",
    "endpoint_full_b32",
    "mlp_endpoint_medium",
    "mlp_endpoint_large",
    "rbm_h32",
]

TIMEOUT_SECONDS = 2400

TMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

base = yaml.safe_load(BASE_CONFIG.read_text())
all_specs = base["model_specs"]

def completed_count(delta, c6, model):
    if not METRICS.exists():
        return 0
    df = pd.read_csv(METRICS)
    mask = (
        (df["delta"].astype(float) == float(delta))
        & (df["c6"].astype(float) == float(c6))
        & (df["model_label"] == model)
    )
    return int(mask.sum())

def write_chunk_config(delta, c6, model):
    cfg = copy.deepcopy(base)
    cfg["parameters"]["delta"] = [float(delta)]
    cfg["parameters"]["c6"] = [float(c6)]
    cfg["training"]["seeds"] = [0, 1, 2, 3, 4]
    cfg["model_specs"] = {model: copy.deepcopy(all_specs[model])}

    safe_delta = str(delta).replace(".", "p")
    safe_c6 = str(c6).replace(".", "p")
    out = TMP_DIR / f"stage4f_open_D{safe_delta}_C{safe_c6}_{model}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out

def run_chunk(delta, c6, model):
    already = completed_count(delta, c6, model)
    if already >= 5:
        print(f"SKIP complete open D={delta} C6={c6} model={model} rows={already}")
        return 0

    cfg_path = write_chunk_config(delta, c6, model)
    safe_delta = str(delta).replace(".", "p")
    safe_c6 = str(c6).replace(".", "p")
    log_path = LOG_DIR / f"open_D{safe_delta}_C{safe_c6}_{model}.log"
    exit_path = LOG_DIR / f"open_D{safe_delta}_C{safe_c6}_{model}.exitcode"

    cmd = [
        sys.executable,
        "scripts/run_stage4e.py",
        "--config",
        str(cfg_path),
        "--resume",
    ]

    print("=" * 100)
    print(f"RUN open D={delta} C6={c6} model={model} existing_rows={already}")
    print(" ".join(cmd))
    print("=" * 100)

    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = "src"

    with log_path.open("w") as log:
        log.write(f"RUN open D={delta} C6={c6} model={model}\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        try:
            rc = proc.wait(timeout=TIMEOUT_SECONDS)
            exit_path.write_text(str(rc) + "\n")
            print(f"DONE open D={delta} C6={c6} model={model} rc={rc}")
            return rc
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(5)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            exit_path.write_text(f"TIMEOUT_AFTER_{TIMEOUT_SECONDS}_SECONDS\n")
            print(f"TIMEOUT open D={delta} C6={c6} model={model}")
            return 124

overall = 0

for delta in DELTAS:
    for c6 in C6S:
        for model in MODELS:
            rc = run_chunk(delta, c6, model)
            if rc not in (0,):
                overall = 1
                print(f"NONZERO_OR_TIMEOUT: open D={delta} C6={c6} model={model} rc={rc}")

print("OPEN CHUNKED COMPLETE overall", overall)
sys.exit(overall)
