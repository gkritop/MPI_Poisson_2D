#!/bin/bash
# Strong scaling (fixed global size). Auto-detect cores; oversubscribe if needed.
set -euo pipefail

NXG=${1:-1024}
NYG=${2:-1024}
OUT=${3:-results/strong.csv}

# Detect logical CPUs (macOS or Linux)
if command -v sysctl >/dev/null 2>&1; then
  CORES=$( (sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu) )
else
  CORES=$(nproc)
fi

# Default sweep (override: PROCS_LIST="1 2 4 8"); since my computer has 10 cores to not cause errors while compiling
PROCS_LIST="${PROCS_LIST:-1 2 4 8}"

# Ensure output directory exists, even if OUT has a parent like ../results/strong.csv
mkdir -p "$(dirname "$OUT")"

# CSV header
echo "mode,nprocs,px,py,nxg,nyg,nxl,nyl,iters,time_s,residual" > "$OUT"

for P in $PROCS_LIST; do
  echo ">> strong: P=$P nxg=$NXG nyg=$NYG (cores=$CORES)"
  if [ -n "${SLURM_JOB_ID-}" ]; then
    POISSON2D_CSV=1 srun -n "$P" ./poisson2d --nx "$NXG" --ny "$NYG" >> "$OUT"
  else
    MPI_CMD=(mpirun -np "$P")
    if [ "$P" -gt "$CORES" ]; then
      MPI_CMD=(mpirun --oversubscribe -np "$P")
    fi
    POISSON2D_CSV=1 "${MPI_CMD[@]}" ./poisson2d --nx "$NXG" --ny "$NYG" >> "$OUT"
  fi
done

echo "Wrote $OUT"
