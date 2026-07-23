# Stage 4F-R.1b final composite release

## Scope

This patch closes the non-scientific release-hygiene items identified after
formal acceptance of the Stage 4F-R.1b campaign and independent replay. It does
not modify any frozen campaign bundle, completion record, checkpoint, replay
bundle, prospective tolerance, or scientific result.

## Commit roles

| Role | Identity | Meaning |
|---|---|---|
| Campaign release build | `8a71f8e8a32f3e8167d385626e0b13da1aa732e9` | Source used to construct the byte-preserving hardened release |
| Replay dispatch | `8992748cef1cd526891c3a15b084112567de58c9` | Exact clean Ubuntu checkout used for the independent replay |
| Replay closure documentation | `8656c2f4` | Durable replay evidence and acceptance record added to `main` |
| Final composite release | tag `stage4fr1b-final-20260723` | Resolves the exact release-hygiene commit and binds public assets |

The machine-readable version is
`release_integrity/commit_roles.json`.

## Closed hygiene findings

- The root manifest now identifies Stage 4F-R.1b as authoritative.
- The former Stage 4D root manifest is retained under
  `release_integrity/historical/` and explicitly marked superseded.
- The 22 July hardened build record is explicitly historical; the final
  composite build record reports the accepted replay.
- Every third-party GitHub Action in the archival workflow is pinned to a full
  commit SHA.
- A deterministic SPDX 2.3 SBOM covers the project and the 59 packages in the
  hash-pinned replay environment.
- The replay-verifier regression performs a clean extraction, strict read-only
  validation, all comparison gates, all periodic invariants, and byte equality
  of the regenerated decision record.
- The authoritative source TAR and source ZIP are created with `git archive`.
  Tracked cache files are absent and the explicit source ZIP preserves the
  executable mode of `verify_repository.sh`. Verification through
  `bash verify_repository.sh` remains portable even when a generic file
  transfer strips executable metadata.

## Local verification

```bash
bash verify_repository.sh

python scripts/validate_stage4fr1b_replay.py --strict \
  --output /tmp/stage4fr1b_replay_regression.json

python scripts/generate_stage4fr1b_sbom.py
git diff --exit-code release_integrity/GraphKAN_Stage4FR1b.sbom.spdx.json
```

After the final patch is committed and tagged:

```bash
python scripts/build_stage4fr1b_composite_release.py \
  --output-dir /tmp/graphkan-stage4fr1b-final
```

The builder refuses a dirty tracked worktree or an untagged commit. It tests
all ZIP structures, confirms source-archive cache hygiene and executable mode,
and writes a composite provenance record plus SHA-256 sidecars.

## Claim boundary

The accepted result is a boundary-conditioned, multi-objective mechanism and
failure map. Endpoint-full and endpoint-hybrid repair compact radial-model
density and connected-correlation errors on open chains, while the MLP
controls retain the energy advantage. Periodic endpoint-disabled variants
collapse exactly to radial. This release does not claim general KAN,
Graph-KAN, or architecture superiority.
