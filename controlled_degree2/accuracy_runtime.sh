#!/usr/bin/env bash
# Shared by production and smoke: do not depend on module/Conda PATH activation.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
ACCURACY_PYTHON="${ACCURACY_PYTHON:-${HOME}/.conda/envs/face_recog/bin/python}"
if [[ ! -x "$ACCURACY_PYTHON" ]]; then
  echo "Accuracy recovery Python is not executable: $ACCURACY_PYTHON" >&2
  exit 1
fi
"$ACCURACY_PYTHON" -c 'import sys, numpy as np, torch, mxnet; print("runtime:", sys.executable, "numpy", np.__version__, "torch", torch.__version__, "mxnet", mxnet.__version__, flush=True); assert int(np.__version__.split(".")[0]) < 2; torch.from_numpy(np.zeros(1, dtype=np.float32)); assert torch.cuda.device_count() >= 4, "four visible GPUs required"'
