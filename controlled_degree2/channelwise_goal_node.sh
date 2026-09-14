#!/usr/bin/env bash
set -euo pipefail
node_ip=$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')
bootstrap_interface=$(ip -o -4 addr show | awk -v addr="$node_ip" '$4 ~ ("^" addr "/") {print $2; exit}')
test -n "$bootstrap_interface"
export NCCL_SOCKET_IFNAME="=$bootstrap_interface"
export GLOO_SOCKET_IFNAME="$bootstrap_interface"
extra=()
if [[ -n "${CHANNEL_RESUME:-}" ]]; then
    extra+=(--resume "$CHANNEL_RESUME")
fi
extra+=(--capture-numerical-failure)
extra+=(--pathological-fraction "${CHANNEL_PATHOLOGICAL_FRACTION:-0.02}")
if [[ -n "${CHANNEL_RESUME_POLICY_REVISION:-}" ]]; then
    extra+=(--resume-policy-revision "$CHANNEL_RESUME_POLICY_REVISION")
fi
if [[ "$CHANNEL_SMOKE" == 1 ]]; then
    extra+=(--smoke --workers 1 --calibration-images 128)
else
    extra+=(--deadline "${CHANNEL_DEADLINE:?}")
fi
exec "$CHANNEL_PYTHON" -m torch.distributed.run \
    --nnodes="${SLURM_NNODES:?}" --nproc_per_node=8 --node_rank="${SLURM_PROCID:?}" \
    --master_addr="$CHANNEL_MASTER" --master_port="$CHANNEL_PORT" \
    -m controlled_degree2.recipe_a "$1" --output "$CHANNEL_OUTPUT" \
    --sitewise-unclipped --train-channel-coefficients \
    --head-warmup 1 --conversion-epochs 12.5 --epochs 24 \
    --lr 0.001 --head-lr 0.005 --coefficient-lr 0.00002 \
    --batchnorm-mode frozen --range-weight 1 "${extra[@]}"
