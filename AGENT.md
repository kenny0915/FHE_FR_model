# AGENTS.md

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
