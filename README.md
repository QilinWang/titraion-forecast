This repository contains the implementation accompanying:

Qilin Wang. Noise Titration: Exact Distributional Benchmarking for Probabilistic Time Series Forecasting.
arXiv:2603.22219.

Noise Titration studies probabilistic time-series forecasting through controlled injection of known observation noise. Instead of treating real-world uncertainty as an unknown nuisance, the benchmark introduces a measurable experimental dial: Gaussian noise with known variance. This makes it possible to evaluate whether a model’s predictive distribution behaves correctly as uncertainty is increased.

The core idea is simple: when the injected noise level is known, distributional claims become testable. Forecast errors can be normalized by the known noise scale, calibration can be checked through coverage and PIT diagnostics, and degradation under increasing noise can be measured directly. This turns robustness and uncertainty evaluation into a controlled intervention rather than a purely observational comparison.

The codebase includes tools for generating clean trajectories, injecting calibrated noise, running probabilistic and deterministic forecasting models under matched noise conditions, and computing distributional diagnostics such as NLL, coverage, PIT behavior, calibration error, and degradation curves across noise levels.

The benchmark is designed to be model-agnostic. Models with native predictive distributions can be evaluated through their reported uncertainty, while deterministic models can be assessed through known-noise residual diagnostics or post-hoc residual variance baselines. The goal is to make probabilistic forecasting claims falsifiable: as the noise level changes, a well-calibrated model should degrade in predictable and measurable ways.

This repository focuses on the Noise Titration protocol and its reproducible experimental setup. A fuller benchmark/data-generation database, including reusable dynamical-system definitions, intervention protocols, and configuration utilities, is being organized as a separate Python module. A mixed Rust/Python implementation is also in progress for faster data generation, reproducible experiment orchestration, and deployment-oriented workflows.
