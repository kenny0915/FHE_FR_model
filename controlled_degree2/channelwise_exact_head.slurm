#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=channelwise-exact-head
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=channelwise-%j.out
#SBATCH --error=channelwise-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${CHANNEL_SOURCE:?}" "${CHANNEL_OUTPUT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WORLD_SIZE=1
PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
deadline=$(( $(date +%s) + 3480 ))
mkdir "$CHANNEL_OUTPUT"
git rev-parse HEAD > "$CHANNEL_OUTPUT/recipe_commit.txt"
common=(--checkpoint "$CHANNEL_SOURCE" --batch-size 64 --lr .003
        --seed 20260924 --deadline "$deadline" --parameter-scope head
        --range-weight 0 --auxiliary-weight 0 --exact-kd-weight 1)
"$PYTHON" -m controlled_degree2.calibrate_ijbc_channelwise "${common[@]}" \
    --output "$CHANNEL_OUTPUT/smoke" --steps 25 --exact-mse-weight 1
# Both arms restart the same finite source and use the same sampled images.
for mse in 0 1; do
    candidate="$CHANNEL_OUTPUT/mse${mse}"
    "$PYTHON" -m controlled_degree2.calibrate_ijbc_channelwise "${common[@]}" \
        --output "$candidate" --steps 2000 --exact-mse-weight "$mse"
    export CHECKPOINT="$candidate/last.pt" RESULT_DIR="$candidate/ijbc_full"
    export CHANNELWISE_ACCEPTANCE=1 POLYNOMIAL_EXPORT=1 FINITE_AUDIT=1
    export POLYNOMIAL_COEFFICIENT_MODE=channelwise NETWORK=r50_controlled_d2
    export EVAL_JOB_NAME="exact_head_mse${mse}"
    status=0
    bash controlled_degree2/job_ijbc.slurm || status=$?
    # Accuracy failure is a completed experiment; extraction failures abort.
    "$PYTHON" - "$RESULT_DIR/acceptance.json" "$status" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert int(sys.argv[2]) in (0, 1), sys.argv[2]
assert all(v for k, v in report['checks'].items() if k != 'tar'), report
PY
    if "$PYTHON" - "$RESULT_DIR/acceptance.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['target_met'] else 1)
PY
    then
        break
    fi
done
