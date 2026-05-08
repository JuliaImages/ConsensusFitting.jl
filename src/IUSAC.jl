# Implementation of the IUSAC algorithm described in:
#   Kim, Lee, Zanetti, Miller, Kim (2024). "Iterative Update Sample Consensus
#   (IUSAC): A repeatable algorithm for optimal consensus set."
#   Journal of Computational and Applied Mathematics, 436, 115423.
#   https://doi.org/10.1016/j.cam.2023.115423

"""
    iusac(x, fittingfn, distfn, s, t;
          rng = Random.default_rng(),
          degenfn = _ -> false,
          verbose = false,
          verbose_io = stdout,
          max_data_trials = 100,
          max_trials = 1000,
          p = 0.99,
          eta_b = 1.0,
          epsilon = 0.001,
          max_inner_iterations = 100)

Robustly fit a model to data using the Iterative Update Sample Consensus (IUSAC)
algorithm of [Kim2024](@citet).

# Arguments

  - `x`: Data array of size `[...] × N` (arbitrary dimensionality per
    data point is supported, but the last dimension must correspond to
    the number of data points). Commonly will be `d × N`, where `d` is the
    dimensionality of each data point and `N` is the total number of data
    points.
  - `fittingfn`: Function that fits a model to a sample of data points.
    Must have the signature `M = fittingfn(x)`. **This function is called with
    subsets of varying size** — from the minimal `s` points (outer sampling step)
    up to the full current inlier set (iterative update step). It must therefore
    implement a least-squares or otherwise over-determined fit when given more
    than `s` points. The function should return an empty collection when it
    cannot produce a valid model (e.g., degenerate input). It may also return a
    collection of multiple candidate models; in that case `distfn` is responsible
    for selecting the best one.
  - `distfn`: Function that scores a model against all data points and returns
    a `NamedTuple`. Must have the signature `nt = distfn(M, x, t)`, where
    `nt.inliers` is a vector of last-dimension indices into `x` for which the
    residual is below threshold `t` (i.e., the inlier data points are
    `x[:, :, ..., nt.inliers]`) and `nt.model` is the scored model. When `M`
    holds multiple candidate models this function should select and return only
    the model with the most inliers.
  - `s`: Minimum number of data points required by `fittingfn` to fit a model
    (e.g., 2 for a line, 3 for a plane, 4 for a homography).
  - `t`: Distance threshold below which a data point is classified as an inlier.

# Keyword Arguments

  - `rng::Random.AbstractRNG`: Random number generator to use for sampling.
    Defaults to `Random.default_rng()`.
  - `degenfn`: Function that tests whether a candidate minimal sample would
    produce a degenerate model. Must have the signature `r = degenfn(x)` and
    return `true` when the sample is degenerate. Defaults to `_ -> false`,
    which treats every sample as non-degenerate and leaves degeneracy detection
    entirely to `fittingfn` (which should return an empty collection for
    degenerate inputs).
  - `verbose::Bool`: When `true`, prints per-iteration diagnostics. Defaults to
    `false`.
  - `verbose_io::IO`: `IO` stream to which verbose output is written. Defaults
    to `stdout`. Primarily useful for testing or redirecting output.
  - `max_data_trials::Integer`: Maximum number of attempts to draw a
    non-degenerate minimal sample before emitting a warning and advancing to
    the next outer iteration. Defaults to `100`.
  - `max_trials::Integer`: Hard upper bound on the number of outer IUSAC
    iterations. Defaults to `1000`.
  - `p::Real`: Desired probability of drawing at least one outlier-free sample,
    used for the same adaptive stopping criterion as RANSAC. Defaults to `0.99`.
  - `eta_b::Real`: Fraction of total data points that the best consensus set
    must reach for the algorithm to terminate early. When `|C*| ≥ eta_b * N`
    the algorithm stops immediately and returns the current best result. Defaults
    to `1.0` (disabled; the algorithm always runs until the adaptive stopping
    criterion or `max_trials` is reached).
  - `epsilon::Real`: Relative convergence tolerance for the inner iterative
    update loop. The inner loop terminates when the consensus set grows by less
    than a fraction `epsilon` relative to its previous size, i.e., when
    ``|C_k| < (1 + \\varepsilon)|C_{k-1}|``. Defaults to `0.001`.
  - `max_inner_iterations::Integer`: Hard upper bound on the number of inner
    iterative update steps performed per outer iteration. Defaults to `100`.

# Returns

  - `M`: The model with the greatest number of inliers found during the search.
  - `inliers`: Vector of last-dimension indices of `x` that are inliers to `M`.

# Extended Help

IUSAC augments the standard RANSAC hypothesis-and-verify loop with an inner
*iterative update* step.  Each outer iteration proceeds as follows.

**Initialization** (outer loop body): a minimal random sample of `s` points
is drawn and a candidate model ``p_0`` is fitted to it.  The initial consensus
set ``C_0 = \\operatorname{inlier}(X, p_0, \\tau_e)`` is computed by scoring
the model against all data points.

**Iterative update** (inner loop): starting from ``C_0`` and ``p_0``, the
model is repeatedly re-estimated from the *entire* current inlier set and
rescored against all data until one of two convergence conditions is met:

1. ``|C_k| < |C_{k-1}|`` (consensus set shrank): the previous set
   ``C_{k-1}`` is declared optimal and the inner loop terminates.
2. ``|C_k| < (1 + \\varepsilon)|C_{k-1}|`` (stagnated growth): the current
   set ``C_k`` is declared optimal and the inner loop terminates.

While neither condition holds the inner loop continues, always moving toward
the larger set.

**Stopping criterion**: if the best consensus set ``C^*`` found in this outer
iteration satisfies ``|C^*| \\geq \\hat{\\eta} N`` (controlled by `eta_b`), the
algorithm terminates early.  Otherwise the outer loop continues, subject to the
adaptive estimate of the number of trials required (identical to the formula
used by `ransac`) and the hard limit `max_trials`.

The key insight of IUSAC is that the iterative update — re-estimating the model
from the full current inlier set rather than from only `s` points — drives the
solution toward the global optimum as long as the initial hypothesis is close
enough to the true model.  This makes the algorithm notably more repeatable than
standard RANSAC while adding only modest overhead per outer iteration.

# References

  - [Kim2024](@citet)
"""
function iusac(x, fittingfn, distfn, s, t;
               rng::AbstractRNG = default_rng(),
               degenfn = _ -> false,
               verbose::Bool = false,
               verbose_io::IO = stdout,
               max_data_trials::Integer = 100,
               max_trials::Integer = 1000,
               p::Real = 0.99,
               eta_b::Real = 1.0,
               epsilon::Real = 0.001,
               max_inner_iterations::Integer = 100)

    npts = size(x, ndims(x))
    npts ≥ s || error("iusac: data has $npts points but the minimum sample size is s = $s")

    best_M = nothing
    best_inliers = Int[]
    best_score = 0
    trial_count = 0
    N = Float64(max_trials)  # adaptive upper bound on required trials

    while trial_count < N
        # ── Initialization: draw a non-degenerate minimal sample ──────────────
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
                @warn "iusac: could not draw a non-degenerate sample after " *
                      "$max_data_trials attempts"
            end
        end

        # Skip this trial if no valid model could be formed from the sample.
        if isnothing(M) || isempty(M)
            trial_count += 1
            continue
        end

        # ── Initial scoring: C0 = inlier(X, p0, τe) ──────────────────────────
        result_init = distfn(M, x, t)
        C_prev = result_init.inliers   # C_{k-1} in the inner loop (starts as C_0)
        M = result_init.model
        n_prev = length(C_prev)

        # C* tracks the best consensus set found in this outer iteration.
        # Initialised to C_0 so that if the inner loop exits immediately
        # (e.g., because C_0 is too small to refit) we still have a valid set.
        C_star = copy(C_prev)
        M_star = M

        # ── Iterative update (Algorithm steps 2a–2c of Kim2024) ───────────────
        for _ in 1:max_inner_iterations
            # Need at least s inliers to refit a model.
            n_prev < s && break

            # (a) Refit from all current inliers: p_k = g(C_{k-1})
            M_cand = fittingfn(selectdim(x, ndims(x), C_prev))
            isempty(M_cand) && break

            # (b) Rescore against all data: C_k = inlier(X, p_k, τe)
            result_new = distfn(M_cand, x, t)
            C_new = result_new.inliers
            M_new = result_new.model
            n_new = length(C_new)

            # (c) Convergence check (Section 3.1 of Kim2024)
            if n_new < n_prev
                # Consensus set shrank: the previous set is already stored in
                # C_star (it was set during the last growing iteration or at
                # initialisation), so we just stop.
                break
            elseif n_new < (1 + epsilon) * n_prev
                # Growth stagnated: accept the (slightly larger or same-sized)
                # current set as the optimal result and stop.
                C_star = C_new
                M_star = M_new
                break
            else
                # Set grew significantly: accept and continue iterating.
                C_prev = C_new
                M = M_new
                n_prev = n_new
                C_star = C_new
                M_star = M_new
            end
        end

        n_star = length(C_star)

        if n_star > best_score
            best_score = n_star
            best_inliers = copy(C_star)
            best_M = M_star

            # Adaptively tighten the stopping criterion (same formula as RANSAC).
            frac = n_star / npts
            p_no_outliers = clamp(1 - frac^s, eps(), 1 - eps())
            N = min(log(1 - p) / log(p_no_outliers), Float64(max_trials))
        end

        trial_count += 1
        verbose && println(verbose_io,
            "trial $trial_count: inliers = $n_star, best = $best_score, " *
            "estimated trials needed = $(ceil(Int, N))")

        # ── Early termination: stopping threshold eta_b reached ─────────────────
        best_score ≥ eta_b * npts && break
    end

    isnothing(best_M) && error("iusac: could not find a valid model")

    return best_M, best_inliers
end
