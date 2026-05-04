using ConsensusFitting
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
using Test

# ── Shared test fixture ────────────────────────────────────────────────────────
# True model: y = a_true * x + b_true
const _A_TRUE = 2.0
const _B_TRUE = 3.0
const _N_INLIERS  = 100
const _N_OUTLIERS = 40
const _seed = 1234 # Seed to use when initializing RNGs

function _make_line_data(rng)
    x_in  = collect(range(-10.0, 10.0; length=_N_INLIERS))
    y_in  = _A_TRUE .* x_in .+ _B_TRUE .+ 0.2 .* randn(rng, _N_INLIERS)
    x_out = -10.0 .+ 20.0 .* rand(rng, _N_OUTLIERS)
    y_out = -25.0 .+ 50.0 .* rand(rng, _N_OUTLIERS)
    return [vcat(x_in, x_out)'; vcat(y_in, y_out)']
end

# fittingfn: fit y = a*x + b through ≥ 2 points using least squares
function _fit_line(pts)
    n = size(pts, 2)
    n < 2 && return []
    if n == 2
        x1, y1 = pts[1, 1], pts[2, 1]
        x2, y2 = pts[1, 2], pts[2, 2]
        isapprox(x1, x2; atol=1e-10) && return []
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return [a, b]
    else
        # Least-squares fit for n > 2 points
        A = hcat(pts[1, :], ones(n))
        b = pts[2, :]
        coef = A \ b
        return [coef[1], coef[2]]
    end
end

# distfn: classify inliers by vertical (y) residual
function _line_dist(M, x, t)
    a, b = M[1], M[2]
    residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(residuals .< t)
    return inliers, M
end

# residualfn: per-point vertical residual (needed for pruning tests)
function _line_residual(M, x)
    a, b = M[1], M[2]
    return abs.(x[2, :] .- (a .* x[1, :] .+ b))
end

@testset "Optimal RANSAC line fitting" begin
    rng = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    M, inliers = optimalransac(data, _fit_line, _line_dist, 2, 0.5; rng=rng)

    # Recovered slope and intercept should be close to the true values
    @test abs(M[1] - _A_TRUE) < 0.1
    @test abs(M[2] - _B_TRUE) < 0.1

    # Should recover the majority of the inlier points
    @test length(inliers) ≥ round(Int, 0.9 * _N_INLIERS)

    # All returned indices must be valid column indices of data
    @test all(1 .≤ inliers .≤ size(data, 2))
end

@testset "Optimal RANSAC repeatability" begin
    # A key property of Optimal RANSAC: repeated runs with DIFFERENT random
    # seeds should usually converge to the same inlier set (the optimal set).
    rng = StableRNG(_seed)
    data = _make_line_data(rng)
    seeds = 1:10 # different seeds to test repeatability across

    results = map(seeds) do seed
        M, inliers = optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                                   rng=StableRNG(seed))
    end

    # All runs should find ~ the same model and the same
    # inlier set when the data has a single dominant structure or feature
    @test length(unique([r[1:2] for r in results])) < round(Int, 0.3 * length(seeds)) # most agree on slope/intercept
    slopes = [r[1][1] for r in results]
    intercepts = [r[1][2] for r in results]
    @test mean(slopes) ≈ _A_TRUE atol=0.01
    @test mean(intercepts) ≈ _B_TRUE atol=0.01
    @test std(slopes) < 0.03
    @test std(intercepts) < 0.03
end

@testset "Optimal RANSAC determinism with fixed seed" begin
    data = _make_line_data(StableRNG(_seed))

    M1, inliers1 = optimalransac(data, _fit_line, _line_dist, 2, 0.5; rng=StableRNG(_seed))
    M2, inliers2 = optimalransac(data, _fit_line, _line_dist, 2, 0.5; rng=StableRNG(_seed))

    # Bit-for-bit identical results with the same seed
    @test M1 == M2
    @test inliers1 == inliers2
end

@testset "Optimal RANSAC with higher-dimensional input" begin
    rng  = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    # Reshape to (1, 2, N) to verify arbitrary leading dimensions are handled
    data_3d = reshape(data, 1, 2, :)
    M, inliers = optimalransac(data_3d,
                               x -> _fit_line(reshape(x, 2, size(x, 3))),
                               (M, x, t) -> _line_dist(M, reshape(x, 2, size(x, 3)), t),
                               2, 0.5; rng=rng)
    @test abs(M[1] - _A_TRUE) < 0.1
    @test abs(M[2] - _B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * _N_INLIERS)
end

@testset "Optimal RANSAC two-tolerance mode with pruning" begin
    # Use t_search > t to enable the pruneset step.
    # With a wider search tolerance the algorithm can find more tentative
    # inliers; pruning then trims them back to the tight tolerance.
    rng  = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    # t_search = 1.0 (wide search), t = 0.3 (tight pruning)
    M_pruned, inliers_pruned = optimalransac(data, _fit_line, _line_dist, 2, 0.3;
                                             t_search=1.0,
                                             residualfn=_line_residual,
                                             rng=rng)

    # Model quality should still be good
    @test abs(M_pruned[1] - _A_TRUE) < 0.1
    @test abs(M_pruned[2] - _B_TRUE) < 0.1

    # After pruning all inliers should lie within the tight tolerance
    residuals = _line_residual(M_pruned, data[:, inliers_pruned])
    @test all(residuals .< 0.3)
end

@testset "Optimal RANSAC min_consensus parameter" begin
    # min_consensus = 3 requires the same inlier-set size to be found three times in a row,
    # making termination more conservative.
    rng  = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    # Should still converge to a good model; just verifying it runs correctly
    M, inliers = optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                               min_consensus=3, rng=rng)
    @test abs(M[1] - _A_TRUE) < 0.1
    @test abs(M[2] - _B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * _N_INLIERS)
end

@testset "Optimal RANSAC error conditions" begin
    # Error when data has fewer points than minimum sample size
    bad_data = rand(2, 1)
    @test_throws ErrorException optimalransac(bad_data, identity,
                                              (M, x, t) -> ([], M), 2, 0.5)

    # t_search < t should throw an ArgumentError
    data     = _make_line_data(StableRNG(_seed))
    @test_throws ArgumentError optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                                             t_search=0.1)
    
    # min_consensus < 2 should throw an ArgumentError
    @test_throws ArgumentError optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                                             min_consensus=1)

    # Degenerate input: all N points are identical
    degenerate_data = repeat([1.0; 2.0], 1, 10)
    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        optimalransac(degenerate_data, _fit_line, _line_dist, 2, 0.5;
                      max_trials=1, max_data_trials=1, rng=StableRNG(_seed))
    catch
    end
    @test_throws ErrorException optimalransac(degenerate_data, _fit_line, _line_dist,
                                              2, 0.5; max_trials=5, max_data_trials=3,
                                              rng=StableRNG(_seed))
end

@testset "Optimal RANSAC verbose output" begin
    rng = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    buf = IOBuffer()
    optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                  verbose=true, verbose_io=buf, rng=rng)
    output = String(take!(buf))
    @test occursin(r"trial \d+: candidate = \d+, best = \d+, σ = \d+/\d+", output)
end

@testset "Optimal RANSAC degenfn respected" begin
    # degenfn returns true for every sample → should warn and eventually error
    data = _make_line_data(StableRNG(_seed))

    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                      degenfn=_ -> true,
                      max_trials=1, max_data_trials=1,
                      rng=StableRNG(_seed))
    catch
    end
end

@testset "Optimal RANSAC max_trials safety limit" begin
    # With max_trials=1 the algorithm must still return a model if one was found
    rng  = StableRNG(_seed)
    data = _make_line_data(rng)
    Random.seed!(rng, _seed)   # reset for algorithm

    # Should not throw; model may be sub-optimal but must be returned
    M, inliers = optimalransac(data, _fit_line, _line_dist, 2, 0.5;
                               max_trials=5, rng=rng)
    @test !isnothing(M)
    @test !isempty(inliers)
    @test all(1 .≤ inliers .≤ size(data, 2))
end
