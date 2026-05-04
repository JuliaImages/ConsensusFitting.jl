# This code is based on the implementation from https://github.com/peterkovesi/ImageProjectiveGeometry.jl/blob/master/src/ransac.jl in the ImageProjectiveGeometry.jl package by Peter Kovesi, which is licensed under the MIT Expat License, reproduced below.

# Copyright (c) 2016: Peter Kovesi.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
    ransac(x, fittingfn, distfn, s, t;
           rng = Random.default_rng(),
           degenfn = _ -> false, 
           verbose = false,
           max_data_trials = 100, 
           max_trials = 1000, 
           p = 0.99)

Robustly fit a model to data using the RANSAC (Random Sample Consensus) algorithm.

# Arguments

  - `x`: Data array of size `[...] × N` (arbitrary dimensionality per 
    data point is supported, but the last dimension must correspond 
    to the number of data points). Commonly will be `d x N`, where 
    `d` is the dimensionality of each data point and `N` is the total 
    number of data points.
  - `fittingfn`: Function that fits a model to a minimal sample of `s` data
    points. Must have the signature `M = fittingfn(x)`.  The function should
    return an empty collection when it cannot produce a valid model (e.g.,
    degenerate input).  It may also return a collection of multiple candidate
    models (for example, up to three fundamental matrices can be recovered from
    seven point correspondences); in that case `distfn` is responsible for
    selecting the best one.
  - `distfn`: Function that scores a model against all data points and returns
    the inlier set.  Must have the signature `(inliers, M) = distfn(M, x, t)`,
    where `inliers` is a vector of last-dimension indices into `x` for which the
    residual is below threshold `t` (i.e., the inlier data points are
    `x[:, :, ..., inliers]`). When `M` holds multiple candidate models
    this function should select and return the one with the most inliers.
  - `s`: Minimum number of data points required by `fittingfn` to fit a model
    (e.g., 2 for a line, 3 for a plane, 4 for a homography).
  - `t`: Distance threshold below which a data point is classified as an
    inlier.

# Keyword Arguments

  - `rng::Random.AbstractRNG`: Random number generator to use for sampling.  Defaults to
    `Random.default_rng()`.
  - `degenfn`: Function that tests whether a candidate minimal sample would
    produce a degenerate model.  Must have the signature `r = degenfn(x)` and
    return `true` when the sample is degenerate.  Defaults to `_ -> false`,
    which treats every sample as non-degenerate and leaves degeneracy detection
    entirely to `fittingfn` (which should return an empty collection for
    degenerate inputs).
  - `verbose::Bool`: When `true`, prints the current trial number and the adaptive
    estimate of the total number of trials required at each iteration.  Defaults
    to `false`.
  - `verbose_io::IO`: `IO` stream to which verbose output is written.  Defaults to
    `stdout`.  Primarily useful for testing or redirecting output.
  - `max_data_trials::Integer`: Maximum number of attempts to draw a non-degenerate
    minimal sample before emitting a warning and advancing to the next outer
    iteration.  Defaults to `100`.
  - `max_trials::Integer`: Hard upper bound on the number of RANSAC iterations.
    Defaults to `1000`.
  - `p::Real`: Desired probability of drawing at least one outlier-free sample.
    Controls the adaptive stopping criterion.  Defaults to `0.99`.

# Returns

  - `M`: The model with the greatest number of inliers found during the search.
  - `inliers`: Vector of column indices of `x` that are inliers to `M`.

# Extended Help

RANSAC alternates between two steps.  In the *hypothesis* step a minimal
random sample of `s` points is drawn and a candidate model is fitted to it.
In the *verification* step every data point is tested against the candidate
model; those within distance `t` form the consensus set (inlier set) of that
model.  The hypothesis with the largest consensus set is retained. The
algorithm terminates adaptively once the expected number of trials required to
find an outlier-free sample with probability `p` has been reached, or after
`max_trials` iterations, whichever comes first.

The adaptive estimate used for the number of trials is

```math
N = \\frac{\\log(1 - p)}{\\log\\!\\left(1 - \\varepsilon^s\\right)}
```

where ``\\varepsilon`` is the current best estimate of the inlier fraction and
``s`` is the minimum sample size.

# References

  - [Fischler1981](@citet)
  - [Hartley2004](@citet)
"""
function ransac(x, fittingfn, distfn, s, t;
                rng::AbstractRNG = default_rng(),
                degenfn = _ -> false,
                verbose::Bool = false,
                verbose_io::IO = stdout,
                max_data_trials::Integer = 100,
                max_trials::Integer = 1000,
                p::Real = 0.99)

    npts = size(x, ndims(x))
    npts ≥ s || error("ransac: data has $npts points but the minimum sample size is s = $s")

    best_M = nothing
    best_inliers = Int[]
    best_score = 0
    trial_count = 0
    N = Float64(max_trials)  # adaptive upper bound on required trials

    while trial_count < N
        # ── Draw a non-degenerate minimal sample ──────────────────────────────
        M = nothing
        for k in 1:max_data_trials
            ind = randperm(rng, npts)[1:s]
            xk = selectdim(x, ndims(x), ind)
            if !degenfn(xk)
                candidate = fittingfn(xk)
                if !isempty(candidate)
                    M = candidate
                    break
                end
            end
            if k == max_data_trials
                @warn "ransac: could not draw a non-degenerate sample after " *
                      "$max_data_trials attempts"
            end
        end

        # Skip this trial if no valid model could be formed from the sample.
        if isnothing(M) || isempty(M)
            trial_count += 1
            continue
        end

        # ── Score the model against all data ──────────────────────────────────
        inliers, M = distfn(M, x, t)
        ninliers = length(inliers)

        if ninliers > best_score
            best_score = ninliers
            best_inliers = copy(inliers)
            best_M = M

            # Adaptively tighten the stopping criterion.
            frac = ninliers / npts
            p_no_outliers = clamp(1 - frac^s, eps(), 1 - eps())
            N = min(log(1 - p) / log(p_no_outliers), Float64(max_trials))
        end

        trial_count += 1
        verbose && println(verbose_io, "trial $trial_count out of estimated $(ceil(Int, N)) required")
    end

    isnothing(best_M) && error("ransac: could not find a valid model")

    return best_M, best_inliers
end