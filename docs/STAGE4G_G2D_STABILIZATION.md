# Stage 4G G2d checkpoint-stabilization tail

G2d preserves the G2b and G2c failures. The G2c seed-level trajectories show
that three unique groups remain in a late optimization tail: endpoint-full is
still improving monotonically, the open MLP trades improving energy against
worsening density and correlation, and the periodic MLP oscillates close to
the frozen correlation limit.

The four matched G2c checkpoints (including the radial partner needed for the
endpoint-full comparison) are continued rather than restarted. Model
parameters are restored by exact NPZ leaf identity, the Adam optimizer state is
reset, and the learning rate is reduced eightfold from 0.0002 to 0.000025.
The stabilization tail adds 3,000 steps and records exact sentinels at total
steps 8,000, 9,000, and 10,000.

All convergence thresholds remain byte-for-value identical to G2b. G2d passes
only if all 12 target group/outcome decisions pass, the 72-row G2b/G2c/G2d
composite closes, endpoint-full remains positive in at least two open cells,
all 20 target runs pass exact health, and the 30/30 same-parameter periodic
alias proof remains valid.
