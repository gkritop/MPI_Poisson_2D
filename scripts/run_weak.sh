#!/bin/bash
# Weak scaling (fixed local size per rank). Auto-detect cores; oversubscribe if needed.
set -euo pipefail

NXL=${1:-256}
NYL=${2:-256}
OUT=${3:-results/weak.csv}

# Detect logical CPUs (macOS or Linux)
if command -v sysctl >/dev/null 2>&1; then
  CORES=$( (sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu) )
else
  CORES=$(nproc)
fi

# Default sweep (override: PROCS_LIST="1 2 4 8"); Same reason...
PROCS_LIST="${PROCS_LIST:-1 2 4 8}"

# Ensure output directory exists, even if OUT has a parent like ../results/weak.csv
mkdir -p "$(dirname "$OUT")"

# CSV header
echo "mode,nprocs,px,py,nxg,nyg,nxl,nyl,iters,time_s,residual" > "$OUT"

for P in $PROCS_LIST; do
  echo ">> weak: P=$P nxl=$NXL nyl=$NYL (cores=$CORES)"
  if [ -n "${SLURM_JOB_ID-}" ]; then
    POISSON2D_CSV=1 srun -n "$P" ./poisson2d --nx "$NXL" --ny "$NYL" --local >> "$OUT"
  else
    MPI_CMD=(mpirun -np "$P")
    if [ "$P" -gt "$CORES" ]; then
      MPI_CMD=(mpirun --oversubscribe -np "$P")
    fi
    POISSON2D_CSV=1 "${MPI_CMD[@]}" ./poisson2d --nx "$NXL" --ny "$NYL" --local >> "$OUT"
  fi
done

echo "Wrote $OUT"
