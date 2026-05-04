# Implementation of the Optimal RANSAC algorithm described in:
#   Hast, Nysjö, Marchetti (2013). "Optimal RANSAC – Towards a Repeatable Algorithm
#   for Finding the Optimal Set." Journal of WSCG, 21, 21–30.

"""
    _oransac_rescore(x, fittingfn, distfn, t, T_init, M_init;
                     max_iterations::Integer = 20)

Algorithm 3 from [Hast2013](@citet): Rescore.

Repeatedly re-estimates the model from the current tentative inliers and
rescores the model against the full dataset until the inlier set stops
changing or 20 iterations are completed.

# Arguments
 - `x`: full data array (last dimension indexes points)
 - `fittingfn`: model-fitting function; called with ≥ s points
 - `distfn`: scoring/distance function; `nt = distfn(M, x, t)` returns a NamedTuple with keys `model` and `inliers`
 - `t`: inlier distance threshold used for scoring
 - `T_init`: initial tentative inlier indices (global indices into x)
 - `M_init`: initial model

# Keyword Arguments

Default values taken directly from [Hast2013](@citet).

 - `max_iterations::Integer=20`: maximum number of rescore iterations, prevents infinite rescoring.
 - `min_inliers::Integer=5`: minimum inlier count required to continue rescoring. Higher numbers
    may be desirable to prevent excessive rescoring on small inlier sets, but highly contaminated
    sets may require a lower threshold to allow the algorithm to escape a poor initial model.

Returns `(M, T, η)` – the best model, its inlier set (global indices into x), and the inlier count found during rescoring.
"""
function _oransac_rescore(x, fittingfn, distfn, t, T_init, M_init;
                          max_iterations::Integer=20,
                          min_inliers::Integer=5)
    @assert max_iterations > 1
    M_best = M_init # Best model
    T_best = copy(T_init) # Best inlier set (global indices into x)
    η_best = length(T_init) # Best inlier count
    T_work = T_best  # T in Algorithm 3's pseudocode (updated each iteration)

    for _ in 1:max_iterations
        # Refit model on all current tentative inliers
        M_raw = fittingfn(selectdim(x, ndims(x), T_work))
        isempty(M_raw) && break

        # Rescore against all data
        result = distfn(M_raw, x, t)
        T_new  = result.inliers
        M_new  = result.model
        η_new  = length(T_new)

        # If enough inliers are found, check for convergence or improvement
        if η_new > min_inliers
            if η_new != η_best
                # Inlier count changed (better or worse) — keep going
                η_best = η_new
                T_best = T_new
                M_best = M_new
                T_work = T_new
            elseif T_new == T_work
                # Exact same inlier set — converged
                M_best = M_new
                T_best = T_new
                break
            else
                # Same count but different composition (wobbling) — keep going
                η_best = η_new
                T_best = T_new
                M_best = M_new
                T_work = T_new
            end
        else
            break  # too few inliers; stop iterating
        end
    end

    return M_best, T_best, η_best
end

"""
_oransac_resample(x, fittingfn, distfn, s, t_search,
                  T_init, η_init, M_init, min_inliers, rng;
                  niter::Integer = 8, sample_fraction::Real = 0.25)

Algorithm 2 from [Hast2013](@citet): Resample.

Repeatedly draws subsets of the current tentative-inlier set, fits candidate
models, and (when promising) rescores those candidates against the full
dataset.  If a larger inlier set is found the resampling loop restarts from
that larger set.

# Arguments
 - `x`: full data array (last dimension indexes points)
 - `fittingfn`: model-fitting function; called with ≥ s points
 - `distfn`: scoring function `nt = distfn(M, x, t)`; returns a NamedTuple with keys `model` and `inliers`
 - `s`: minimum sample size
 - `t_search`: inlier threshold for resampling/rescoring (≥ t)
 - `T_init`: initial tentative inlier indices (global)
 - `η_init`: length(T_init)
 - `M_init`: initial model
 - `min_inliers`: minimum inlier count required to trigger a rescore
 - `rng`: random number generator

# Keyword Arguments

Default values taken directly from [Hast2013](@citet).

 - `niter::Integer=8`: maximum number of resampling iterations per call. Defaults to `8`.
 - `sample_fraction::Real=0.25`: fraction of the current inlier set to sample for each candidate. Defaults to `0.25`.

Returns `(M, T, η)` -- the best model, its inlier set (global indices into x), and the inlier count found during resampling.
"""
function _oransac_resample(x, fittingfn, distfn, s, t_search,
                           T_init, η_init, M_init, min_inliers, rng;
                           niter::Integer = 8, sample_fraction::Real = 0.25)
    @assert niter > 1
    @assert 0 < sample_fraction < 1
    η = η_init # current inlier count
    T = T_init # global indices into x
    M = M_init # current model

    i = 0
    while i < niter
        i += 1

        # Sample max(s, round(Int, sample_fraction * η)) points from the current tentative-inlier set.
        n_sample = max(s, round(Int, sample_fraction * η))
        sub_local = randperm(rng, η)[1:n_sample]   # local indices into T
        x_sample  = selectdim(x, ndims(x), T[sub_local])

        M_raw = fittingfn(x_sample)
        isempty(M_raw) && continue

        # Quick score on the current inlier subset (cheap rejection step).
        result_sub = distfn(M_raw, selectdim(x, ndims(x), T), t_search)
        T_sub_local = result_sub.inliers
        M_cand = result_sub.model
        length(T_sub_local) ≤ min_inliers && continue

        # Convert local subset indices to global, then run full rescore.
        T_global_cand = T[T_sub_local]
        M_cand, T_global_cand, η_new =
            _oransac_rescore(x, fittingfn, distfn, t_search, T_global_cand, M_cand)

        if η_new > η
            η = η_new
            T = T_global_cand
            M = M_cand
            i = 0           # found a larger set — restart the inner loop
        end
    end

    return M, T, η
end

"""
    _oransac_pruneset(x, fittingfn, distfn, t, T_global, M_init)

Algorithm 4 from [Hast2013](@citet): Pruneset.

Iteratively removes the point with the largest residual from the working set
and re-estimates the model until every remaining point lies within tolerance
`t`.  This step requires `distfn` to compute per-point residuals.

# Arguments
 - `x`: full data array
 - `fittingfn`: model-fitting function
 - `distfn`: scoring function `nt = distfn(M, x, t)`; must return a NamedTuple
   with key `residuals` (a non-negative vector, one per point in `x`) in
   addition to `model` and `inliers`, in order to identify the most extreme
   inlier.  If the returned NamedTuple lacks the `residuals` key the pruning
   step is skipped and the current set is returned unchanged.
 - `t`: tight (pruning) tolerance
 - `T_global`: current inlier indices (global into x)
 - `M_init`: current model

Returns `(M, kept_local)` where `kept_local` are the retained indices into
`T_global` (i.e. the global indices of the pruned set are `T_global[kept_local]`).
"""
function _oransac_pruneset(x, fittingfn, distfn, t, T_global, M_init)
    n = length(T_global)
    kept = collect(1:n)   # start with all points
    M = M_init

    length(kept) ≤ 5 && return M, kept

    done = false
    while length(kept) > 5 && !done
        x_kept = selectdim(x, ndims(x), T_global[kept])
        result_kept = distfn(M, x_kept, t)
        residuals = result_kept.residuals

        max_res, max_idx = findmax(residuals)

        if max_res > t
            # Remove the most extreme point and refit.
            deleteat!(kept, max_idx)
            length(kept) < 2 && (done = true; break)

            M_new = fittingfn(selectdim(x, ndims(x), T_global[kept]))
            if isempty(M_new)
                done = true
            else
                M = M_new
            end
        else
            done = true   # all remaining points are within tolerance
        end
    end

    return M, kept
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    optimalransac(x, fittingfn, distfn, s, t;
                  rng = Random.default_rng(),
                  t_search = t,
                  degenfn = _ -> false,
                  verbose = false,
                  max_data_trials = 100,
                  max_trials = 1000,
                  min_inliers = 5,
                  min_consensus = 2)

Robustly fit a model to data using the Optimal RANSAC algorithm of
[Hast2013](@citet), which extends standard RANSAC with three refinement steps
(resample, rescore, and optionally pruneset) to produce a repeatable result.

# Arguments

  - `x`: Data array of size `[...] × N` (arbitrary dimensionality per
    data point is supported, but the last dimension must correspond to
    the number of data points). Commonly will be `d × N`.
  - `fittingfn`: Function that fits a model to a sample of data points.
    Must have the signature `M = fittingfn(x)`.  **Unlike `ransac`, this
    function is called with subsets of varying size** — from the minimal `s`
    points (outer sampling) up to the full current inlier set (rescore step).
    It must therefore implement a least-squares or otherwise over-determined
    fit when given more than `s` points.  The function should return an empty
    collection when it cannot produce a valid model (e.g., degenerate input).
    It may also return a collection of multiple candidate models; in that case
    `distfn` is responsible for selecting the best one.
  - `distfn`: Function that scores a model against all data points and returns
    a `NamedTuple`.  Must have the signature `nt = distfn(M, x, t)`, where
    `nt.inliers` is a vector of last-dimension indices into `x` for which the
    residual is below threshold `t` and `nt.model` is the scored model.  When
    `M` holds multiple candidate models this function should select and return
    the one with the most inliers via `nt.model`.  To enable the optional
    pruning step (Algorithm 4), the returned `NamedTuple` must also contain a
    key `residuals` with a non-negative vector of per-point residuals (one
    entry per last-dimension slice of the `x` passed to `distfn`).  Pruning is
    only activated when both `t_search > t` and `distfn` provides
    `nt.residuals`.
  - `s`: Minimum number of data points required by `fittingfn` to fit a model.
  - `t`: Primary inlier distance threshold (used for the outer sampling step
    and, when pruning is enabled, as the final tight tolerance).

# Keyword Arguments

  - `rng::Random.AbstractRNG`: Random number generator.  Defaults to `Random.default_rng()`.
  - `t_search::Real`: Inlier threshold used during the resampling and rescoring
    steps.  Must satisfy `t_search ≥ t`.  When `t_search > t` and `distfn`
    returns `residuals` in its `NamedTuple`, a pruning pass is applied after
    resampling to trim the result back to the tight tolerance `t`, yielding the
    highest-precision final inlier set.  Defaults to `t` (no separate search
    tolerance; pruning is skipped).
  - `degenfn`: Function that tests whether a minimal sample would produce a
    degenerate model.  Must have the signature `r = degenfn(x)` and return
    `true` when the sample is degenerate.  Defaults to `_ -> false`.
  - `verbose::Bool`: When `true`, prints per-iteration diagnostics.  Defaults to
    `false`.
  - `verbose_io::IO`: `IO` stream for verbose output.  Defaults to `stdout`.
  - `max_data_trials::Integer`: Maximum attempts to draw a non-degenerate minimal
    sample in the outer loop before emitting a warning.  Defaults to `100`.
  - `max_trials::Integer`: Hard upper bound on outer-loop iterations.  Acts as a
    safety limit; the algorithm's primary stopping criterion is the
    `min_consensus` convergence condition.  Defaults to `1000`.
  - `min_inliers::Integer`: Minimum tentative inlier count required to trigger the
    resampling/rescoring optimization.  Defaults to `5`.
  - `min_consensus::Integer`: Number of times the same-size inlier set must be
    found before the algorithm declares convergence.  Defaults to
    `2`, such that the algorithm must find the same inlier count twice in a row.
    For small inlier sets (fewer than ≈ 30 expected inliers) the paper
    recommends increasing this to `2` or more to avoid premature termination
    on a sub-optimal set.

# Returns

  - `M`: The model with the largest (or, with pruning, the highest-quality)
    inlier set found by the algorithm.
  - `inliers`: Vector of last-dimension indices of `x` that are inliers to `M`.

# Extended Help

Optimal RANSAC augments the standard hypothesis-then-verify loop with three
additional refinement steps taken whenever a hypothesis yields more than
`min_inliers` tentative inliers.

**Resample** (Algorithm 2 of [Hast2013](@citet)) draws subsets of up to a
quarter of the current tentative-inlier set, fits new candidate models, and
rescores them against the full dataset.  If a larger inlier set is found the
loop restarts from that larger set; the procedure repeats up to 8 times per
outer iteration.

**Rescore** (Algorithm 3) repeatedly re-estimates the model from the full
current inlier set and rescores against all data until the inlier set stops
changing or 20 iterations are exhausted.

**Pruneset** (Algorithm 4) is an optional final step enabled when
`t_search > t` and `distfn` returns `residuals` in its `NamedTuple`.  It
iteratively removes the point with the largest residual from the working set
and re-estimates the model until every remaining point lies within the tight
threshold `t`.

The algorithm terminates when the same inlier-set *size* is found
`min_consensus` times in succession, indicating convergence to the optimal
set.  Unlike standard RANSAC, the stopping criterion does not depend on a
statistical threshold over the inlier fraction; this makes the algorithm
applicable to very highly contaminated sets (inlier ratios well below 5%).

### Repeatability

The strong local-refinement steps (resample + rescore) reliably guide the
search toward the global maximum from almost any starting hypothesis, making
Optimal RANSAC near-deterministic across runs even with different random seeds,
provided the data has a single dominant structure.  Pass the same seeded `rng`
to obtain bit-for-bit identical results.

### Limitations

The algorithm is only appropriate when the model can be fitted to *more than*
the minimal `s` points — the `fittingfn` must accept over-determined inputs.
For problems with multiple competing structures of similar quality the
algorithm may converge to a locally optimal set rather than the global
optimum.

# References

  - [Hast2013](@citet)
"""
function optimalransac(x, fittingfn, distfn, s, t;
                       rng::AbstractRNG = default_rng(),
                       t_search::Real = t,
                       degenfn = _ -> false,
                       verbose::Bool = false,
                       verbose_io::IO = stdout,
                       max_data_trials::Integer = 100,
                       max_trials::Integer = 1000,
                       min_inliers::Integer = 5,
                       min_consensus::Integer = 2)

    min_consensus = min_consensus - 1 # convert to "number of times to find the same size after the first" for easier tracking
    min_consensus ≥ 1 || throw(ArgumentError("min_consensus must be at least 2 to allow for convergence tracking"))
    t_search ≥ t ||
        throw(ArgumentError("t_search ($t_search) must be ≥ t ($t)"))

    npts = size(x, ndims(x))
    npts ≥ s ||
        error("optimalransac: data has $npts points but minimum sample size is s = $s")

    # Pruning requires a strictly wider search tolerance; whether distfn
    # actually provides residuals is checked at runtime inside _oransac_pruneset.
    do_pruning = t_search > t

    best_M = nothing
    best_T = Int[]   # current tracked inlier set (T in Algorithm 1)
    best_η = 0       # best confirmed inlier count (for convergence tracking)
    σ = 0       # consecutive same-size confirmation counter
    trial_count = 0

    while σ < min_consensus && trial_count < max_trials
        trial_count += 1

        # ── Draw a minimal sample from all data points.  Using global
        #    (uniform) sampling rather than biased sampling from the current
        #    best inlier set avoids getting trapped in local optima, which is
        #    critical for repeatability across different random seeds.  The
        #    strong local refinement (resample + rescore) is what guides the
        #    algorithm to the optimal set from any starting hypothesis. ──────
        M = nothing
        for k in 1:max_data_trials
            sub = randperm(rng, npts)[1:s]
            xk  = selectdim(x, ndims(x), sub)
            if !degenfn(xk)
                candidate = fittingfn(xk)
                if !isempty(candidate)
                    M = candidate
                    break
                end
            end
            if k == max_data_trials
                @warn "optimalransac: could not draw a non-degenerate sample " *
                      "after $max_data_trials attempts"
            end
        end

        (isnothing(M) || isempty(M)) && continue

        # ── Initial scoring with the primary tolerance t (ε₀ in the paper) ─
        result_init = distfn(M, x, t)
        T_cand = result_init.inliers
        M = result_init.model
        η_cand = length(T_cand)
        # Now that we have a result from distfn, check whether it provides residuals for pruning; if not, we'll skip pruning later.
        do_pruning = do_pruning && haskey(result_init, :residuals)

        if η_cand > min_inliers
            # ── Resample (Algorithm 2), using the wider search tolerance ────
            M, T_cand, η_resample = _oransac_resample(
                x, fittingfn, distfn, s, t_search,
                T_cand, η_cand, M, min_inliers, rng)

            # ── Optional pruneset (Algorithm 4), tightening to tolerance t ─
            if do_pruning
                M, kept = _oransac_pruneset(x, fittingfn, distfn, t, T_cand, M)
                T_cand = T_cand[kept]
                η_cand = length(T_cand)
            else
                # No pruning: the candidate count is the resample's best.
                η_cand = η_resample
            end
        end

        # ── Stopping-criterion update (Algorithm 1 convergence logic) ───────
        #
        # Save the best count from before this iteration so that we can detect
        # whether the current candidate is an improvement.  best_η is only
        # updated *inside* the branches below, so the comparison is always
        # against the previous best (avoiding a tautological η == η' when no
        # pruning is used).
        prev_best_η = best_η

        if η_cand > prev_best_η
            # Strictly larger inlier set found — reset convergence counter
            # and record the new best.
            σ = 0
            best_η = η_cand
            best_T = copy(T_cand)
            best_M = M
        elseif η_cand == prev_best_η && prev_best_η > min_inliers
            # Same count as current best — check whether the set composition
            # also matches (paper's `|T| = |T'|` comparison).
            if length(best_T) == length(T_cand)
                # If same size, increment convergence counter
                σ += 1
                best_T = copy(T_cand)   # record most recent realisation
                best_M = M
            else
                # Same count but different composition (shouldn't occur with
                # the previous `> prev_best_η` branch already handled, but
                # guards the very first valid result when best_T is empty).
                best_T = copy(T_cand)
                best_M = M
            end
        elseif η_cand == prev_best_η - 1 && prev_best_η > min_inliers
            # One fewer inlier than the current best (edge case noted in the
            # paper where pruning reduces the set by exactly one point).
            σ = 0
            best_η = η_cand
            best_T = copy(T_cand)
            best_M = M
        end
        # All other cases (significant degradation or optimization skipped):
        # leave best_η, best_T, best_M and σ unchanged.

        verbose && println(verbose_io,
            "trial $trial_count: candidates = $η_cand, max candidates = $best_η, " *
            "σ = $σ/$min_consensus")
    end

    verbose && println(verbose_io, "OptimalRANSAC pruning enabled: $do_pruning")

    isnothing(best_M) && error("optimalransac: could not find a valid model")
    return best_M, best_T
end
