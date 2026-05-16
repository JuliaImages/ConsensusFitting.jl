# ConsensusFitting.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaimages.org/ConsensusFitting.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaimages.org/ConsensusFitting.jl/dev/)
[![Build Status](https://github.com/JuliaImages/ConsensusFitting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaImages/ConsensusFitting.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaImages/ConsensusFitting.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaImages/ConsensusFitting.jl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ConsensusFitting.jl provides robust, consensus-based algorithms for fitting models to data contaminated by outliers — observations that do not follow the underlying model. By explicitly partitioning the data into *inliers* and *outliers*, these methods recover accurate fits where ordinary least squares would be derailed by even a handful of bad points. Such algorithms are widely used in computer vision, photogrammetry, and geometric estimation.

## Algorithms

- `ransac` — classic [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) (Fischler & Bolles, 1981); fast and robust to large outlier fractions.
- `optimalransac` — Optimal RANSAC (Hast et al., 2013); adds resample, rescore, and prune refinement steps for repeatable, near-deterministic results.
- `iusac` — Iterative Update Sample Consensus (Kim et al., 2024); adds an inner iterative-update loop that drives each hypothesis toward the global optimum.

## Installation

```julia
using Pkg
Pkg.add("ConsensusFitting")
```

## Usage

The three algorithms share a common call shape:

```julia
M, inliers = ransac(x, fittingfn, distfn, s, t; kwargs...)
```

You supply two functions describing your model: `fittingfn` fits a candidate model to a minimal sample of `s` points, and `distfn` scores a model against the full dataset and returns its inlier set. Each call returns the best model `M` and the indices of the points consistent with it.

See the [documentation](https://juliaimages.org/ConsensusFitting.jl/stable/) for worked examples of each algorithm.
