# Stage 4F-R.1b independent replay closure

## Decision

The preregistered Stage 4F-R.1b independent Linux replay passed. The external
integrity and reproducibility gate requested in the 22 July 2026 supervisory
review is closed for this release.

This closes the release-integrity gate. It does not change the registered
scientific interpretation: the evidence supports a boundary-conditioned,
multi-objective trade-off and failure map, not general KAN or Graph-KAN
superiority.

## Immutable replay identity

- GitHub Actions run: `29981334627`
- Job: `89123554796`
- Replay commit: `8992748cef1cd526891c3a15b084112567de58c9`
- Host: clean GitHub-hosted Ubuntu 24.04 x86-64 runner
- Python: 3.12.7
- Backend: JAX CPU
- Clean pre-replay Git status: yes
- Environment attestation: PASS, with no package-version mismatches

The original GitHub artifact wrapper had SHA-256
`8325cff38bb301f1ebbf9b59b6fd4192d2b3744716c8fa57152d30807dd2e354`.
The durable inner replay archive in `releases/` has SHA-256
`cf14747198ae8302d6bad298428faf3bb156d610841b9126590813be214ee8e6`.

## Gate results

- Replay rows: 20/20 complete
- Core rows: 16/16 complete
- Control rows: 4/4 complete
- Strict bundle/registry validation: PASS
- Observable/checkpoint comparisons: 20/20 PASS
- Periodic-collapse invariants: 4/4 PASS
- Blocking errors: 0
- Checkpoints present and validated: 20/20

The comparator checked exact energy, complete site-density vectors, complete
connected-correlation matrices, health fields, parameter/model metadata,
checkpoint structure and hashes, protocol/config identity, commit provenance,
and the preregistered periodic-collapse invariants. The locally regenerated
comparison record was byte-for-byte identical to the record produced on the
Linux runner.

## Durable evidence

- `releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip`
- `releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip.sha256`
- `releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.verification.json`

The archive contains 20 canonical replay bundles, 20 readable checkpoints,
the registry, expected-run roster, validation and comparison records, Git
commit/tree/status evidence, GitHub run identity, pinned-environment
attestation, and complete `pip freeze`.

## Independent verification

From the repository root:

```bash
cd releases
shasum -a 256 -c GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip.sha256
unzip -tq GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip
cd ..
```

For a strict revalidation after extraction, preserve the campaign-relative
layout recorded by the immutable replay registry:

```bash
tmp_dir="$(mktemp -d)"
mkdir -p "$tmp_dir/results_stage4fr1b"
unzip -q releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip \
  -d "$tmp_dir/archive"
mv "$tmp_dir/archive/independent_replay" \
  "$tmp_dir/results_stage4fr1b/independent_replay"

repo_root="$PWD"
(
  cd "$tmp_dir"
  PYTHONPATH="$repo_root/src:$repo_root/scripts" \
    python "$repo_root/scripts/stage4f1c_campaign.py" validate \
      --campaign results_stage4fr1b/independent_replay \
      --strict-integrity \
      --reconcile-registry \
      --require-terminal \
      --require-complete \
      --no-write
)

PYTHONPATH="$PWD/src" python scripts/compare_stage4fr1b_replay.py \
  --replay "$tmp_dir/results_stage4fr1b/independent_replay" \
  --reference results_stage4f_r/frozen \
  --config configs/stage4fr1b_second_host_replay.yaml \
  --tolerances release_integrity/stage4fr1b_replay_tolerances.json \
  --expected-commit 8992748cef1cd526891c3a15b084112567de58c9 \
  --output "$tmp_dir/replay_comparison.rechecked.json"
```

The layout step is required because the immutable SQLite registry records the
campaign-relative bundle paths used on the replay host. No replay payload is
rewritten during verification.
