#!/usr/bin/env bash
set -euo pipefail
echo "recipe node=$(hostname) rank=$SLURM_PROCID stage=$1"
# NCCL's automatic IPoIB bootstrap selection is not routable between these
# nodes. Use the address Slurm/torchrun resolve, retaining IB for payloads.
node_ip=$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')
bootstrap_interface=$(ip -o -4 addr show | awk -v addr="$node_ip" '$4 ~ ("^" addr "/") {print $2; exit}')
if [[ -z "$bootstrap_interface" ]]; then
    echo "Cannot resolve bootstrap interface for $node_ip" >&2
    exit 1
fi
export NCCL_SOCKET_IFNAME="=$bootstrap_interface"
export GLOO_SOCKET_IFNAME="$bootstrap_interface"
echo "bootstrap interface=$bootstrap_interface address=$node_ip"
extra=()
if [[ "$1" == recovery || "$1" == recovery-v2 ]]; then
    recovery_module=controlled_degree2.recipe_a_recovery
    if [[ "$1" == recovery-v2 ]]; then
        recovery_module=controlled_degree2.recipe_a_recovery_v2
    fi
    if [[ -n "${RECOVERY_RESUME:-}" ]]; then
        extra+=(--resume "$RECOVERY_RESUME")
    fi
    if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && -f "$RECIPE_OUTPUT/source_epoch8.pt" ]]; then
        extra+=(--reuse-output)
    fi
    if [[ "$RECIPE_SMOKE" == 1 ]]; then
        extra+=(--smoke --workers 1 --gate-images 32 --no-continue-training)
    fi
    exec "$RECIPE_PYTHON" -m torch.distributed.run \
        --nnodes="${SLURM_NNODES:?}" --nproc_per_node=8 \
        --node_rank="${SLURM_PROCID:?}" \
        --master_addr="$RECIPE_MASTER_ADDR" --master_port="$RECIPE_MASTER_PORT" \
        -m "$recovery_module" \
        --source "${RECOVERY_SOURCE:-work_dirs/recipe_a_376833}" \
        --output "$RECIPE_OUTPUT" "${extra[@]}"
fi
if [[ "${RECIPE_SHARED:-0}" == 1 ]]; then
    extra+=(--shared --initialization "${SHARED_INIT:-fit}"
            --coefficient-lr "${SHARED_COEFF_LR:-0.0001}"
            --batchnorm-mode "${SHARED_BN:-train}"
            --head-warmup "${SHARED_WARMUP:-0.25}"
            --conversion-epochs "${SHARED_CONVERSION:-3}"
            --epochs "${SHARED_EPOCHS:-12}"
            --lr "${SHARED_LR:-0.004}" --deadline "${SHARED_DEADLINE:?}")
fi
if [[ "$RECIPE_SMOKE" == 1 ]]; then
    extra+=(--smoke --workers 1 --calibration-images 128)
fi
exec "$RECIPE_PYTHON" -m torch.distributed.run \
    --nnodes="${SLURM_NNODES:?}" --nproc_per_node=8 \
    --node_rank="${SLURM_PROCID:?}" \
    --master_addr="$RECIPE_MASTER_ADDR" --master_port="$RECIPE_MASTER_PORT" \
    -m controlled_degree2.recipe_a "$1" --output "$RECIPE_OUTPUT" "${extra[@]}"
