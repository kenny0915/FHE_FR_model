## Project overview

This repository contains experiments for FHE-compatible face-recognition models.

Primary goal:
- Train and evaluate face-recognition backbones such as iResNet / iResNet50.
- Replace FHE-unfriendly activations such as PReLU / ReLU with polynomial approximations.
- Compare baseline and modified models under normal inference and FHE-oriented constraints.

The project likely involves:
- PyTorch training and evaluation.
- Face-recognition losses such as ArcFace / AdaFace / PartialFC.
- Datasets with aligned face images, commonly shaped like `[N, 3, 112, 112]`.
- Experiment logs, checkpoints, and result tables.

## FHE-specific constraints

For FHE-compatible experiments:
- Avoid non-polynomial nonlinearities in the encrypted path.
- Treat ReLU, PReLU, sigmoid, softmax, max-pooling, division, and data-dependent branching as suspicious unless they occur outside the encrypted path.
- Prefer low-degree polynomial activations.
- Track multiplicative depth implications when increasing polynomial degree.
- Do not introduce operations that make FHE inference harder unless clearly justified.

For polynomial activations:
- Always state the approximation target and interval.
- Consider input distribution before selecting the interval.
- Check activation ranges during inference/training.
- Compare against baseline PReLU/ReLU accuracy.
- Watch for exploding activations when polynomial degree or coefficients are poorly scaled.

## Execution environment

- Full training runs on a separate GPU server with 4 NVIDIA V100-SXM2-32GB GPUs.
- This repository checkout is only for code modifications and small, lightweight tests.
- Do not launch full training jobs in this environment; prepare and validate the code here, then run training on the GPU server.

# Repository working instructions

- When the user asks to fix, modify, or generate repository content, finish the requested work and run appropriate tests.
- After the tests confirm the requested change works, commit all files belonging to that request and push the commit to the current branch's configured upstream, unless the user explicitly asks not to commit or push.
- Do not include unrelated working-tree changes in the commit.
