# Stage 4G G2c targeted recovery

G2b remains a frozen scientific failure: 67/72 late-sentinel group
comparisons passed, endpoint-full passed its two stability checks,
endpoint-hybrid passed none, and independently sharded periodic aliases did not
follow byte-identical optimisation trajectories.

G2c does not change those facts or loosen a G2b threshold. It addresses two
separate questions prospectively:

1. The five groups implicated by the G2b convergence decision are rerun from
   their frozen seeds at a four-times smaller learning rate for 7,000 steps.
   The radial partner of the open endpoint-full group is rerun at the same
   budget so the final mechanism comparison remains matched.
2. Periodic endpoint/radial invariance is tested as an implementation property:
   the three aliases are initialized from the same parameter key and evaluated
   on all 1,024 ten-site configurations. This avoids conflating architectural
   identity with small cross-run optimisation differences on separate hosts.

The numerical convergence limits remain exactly those frozen before G2b.
Endpoint-full is the primary mechanism hypothesis because it passed the G2b
stability decision. Endpoint-hybrid remains visible as a negative diagnostic;
its G2b failure is not converted to a positive result.
