using ConsensusFitting
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
using Test

# ── Shared test fixture ────────────────────────────────────────────────────────
# True model: y = a_true * x + b_true
const A_TRUE = 2.0
const B_TRUE = 3.0
const N_INLIERS  = 100
const N_OUTLIERS = 40
const SEED = 1234 # Seed to use when initializing RNGs

function make_line_data(rng)
    x_in  = collect(range(-10.0, 10.0; length=N_INLIERS))
    y_in  = A_TRUE .* x_in .+ B_TRUE .+ 0.2 .* randn(rng, N_INLIERS)
    x_out = -10.0 .+ 20.0 .* rand(rng, N_OUTLIERS)
    y_out = -25.0 .+ 50.0 .* rand(rng, N_OUTLIERS)
    return [vcat(x_in, x_out)'; vcat(y_in, y_out)']
end

# fittingfn: fit y = a*x + b through ≥ 2 points using least squares
function fit_line(pts)
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
function line_dist(M, x, t)
    a, b = M[1], M[2]
    residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(residuals .< t)
    return (model=M, inliers=inliers)
end

# distfn with residuals: same as line_dist but also returns per-point residuals
# (needed so that optimalransac can perform the optional pruneset step)
function line_dist_with_residuals(M, x, t)
    a, b = M[1], M[2]
    residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(residuals .< t)
    return (model=M, inliers=inliers, residuals=residuals)
end

# per-point vertical residual (used for test assertions only)
function line_residual(M, x)
    a, b = M[1], M[2]
    return abs.(x[2, :] .- (a .* x[1, :] .+ b))
end

@testset "Optimal RANSAC line fitting" begin
    rng = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    M, inliers = optimalransac(data, fit_line, line_dist, 2, 0.5; rng=rng)

    # Recovered slope and intercept should be close to the true values
    @test abs(M[1] - A_TRUE) < 0.1
    @test abs(M[2] - B_TRUE) < 0.1

    # Should recover the majority of the inlier points
    @test length(inliers) ≥ round(Int, 0.9 * N_INLIERS)

    # All returned indices must be valid column indices of data
    @test all(1 .≤ inliers .≤ size(data, 2))

    # Test that providing residuals from distfn but leaving t_search=t gives same answer
    Random.seed!(rng, SEED)   # reset for algorithm
    M2, inliers2 = optimalransac(data, fit_line, line_dist_with_residuals, 2, 0.5; rng=rng, t_search=0.5)
    @test M2 == M
    @test inliers2 == inliers
end

@testset "Optimal RANSAC repeatability" begin
    # A key property of Optimal RANSAC: repeated runs with DIFFERENT random
    # seeds should usually converge to the same inlier set (the optimal set).
    rng = StableRNG(SEED)
    data = make_line_data(rng)
    seeds = 1:10 # different seeds to test repeatability across

    results = map(seeds) do seed
        M, inliers = optimalransac(data, fit_line, line_dist, 2, 0.5;
                                   rng=StableRNG(seed))
    end

    # All runs should find ~ the same model and the same
    # inlier set when the data has a single dominant structure or feature
    @test length(unique([r[1:2] for r in results])) < round(Int, 0.3 * length(seeds)) # most agree on slope/intercept
    slopes = [r[1][1] for r in results]
    intercepts = [r[1][2] for r in results]
    @test mean(slopes) ≈ A_TRUE atol=0.01
    @test mean(intercepts) ≈ B_TRUE atol=0.01
    @test std(slopes) < 0.03
    @test std(intercepts) < 0.05
end

@testset "Optimal RANSAC determinism with fixed seed" begin
    data = make_line_data(StableRNG(SEED))

    M1, inliers1 = optimalransac(data, fit_line, line_dist, 2, 0.5; rng=StableRNG(SEED))
    M2, inliers2 = optimalransac(data, fit_line, line_dist, 2, 0.5; rng=StableRNG(SEED))

    # Bit-for-bit identical results with the same seed
    @test M1 == M2
    @test inliers1 == inliers2
end

@testset "Optimal RANSAC with higher-dimensional input" begin
    rng  = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    # Reshape to (1, 2, N) to verify arbitrary leading dimensions are handled
    data_3d = reshape(data, 1, 2, :)
    M, inliers = optimalransac(data_3d,
                               x -> fit_line(reshape(x, 2, size(x, 3))),
                               (M, x, t) -> line_dist(M, reshape(x, 2, size(x, 3)), t),
                               2, 0.5; rng=rng)
    @test abs(M[1] - A_TRUE) < 0.1
    @test abs(M[2] - B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * N_INLIERS)
end

@testset "Optimal RANSAC two-tolerance mode with pruning" begin
    # Use t_search > t to enable the pruneset step.
    # With a wider search tolerance the algorithm can find more tentative
    # inliers; pruning then trims them back to the tight tolerance.
    rng  = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    # t_search = 1.0 (wide search), t = 0.3 (tight pruning)
    M_pruned, inliers_pruned = optimalransac(data, fit_line, line_dist_with_residuals, 2, 0.3;
                                             t_search=1.0,
                                             rng=rng)

    # Model quality should still be good
    @test abs(M_pruned[1] - A_TRUE) < 0.1
    @test abs(M_pruned[2] - B_TRUE) < 0.1

    # After pruning all inliers should lie within the tight tolerance
    residuals = line_residual(M_pruned, data[:, inliers_pruned])
    @test all(residuals .< 0.3)

    # Use verbose output and test that pruning is enabled
    buf = IOBuffer()
    M_pruned_verbose, inliers_pruned_verbose = optimalransac(data, fit_line, line_dist_with_residuals, 2, 0.3;
                                                            t_search=1.0,
                                                            verbose=true, verbose_io=buf,
                                                            rng=rng)
    output = String(take!(buf))
    @test occursin("pruning enabled: true", output)
end

@testset "Optimal RANSAC min_consensus parameter" begin
    # min_consensus = 3 requires the same inlier-set size to be found three times in a row,
    # making termination more conservative.
    rng  = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    # Should still converge to a good model; just verifying it runs correctly
    M, inliers = optimalransac(data, fit_line, line_dist, 2, 0.5;
                               min_consensus=3, rng=rng)
    @test abs(M[1] - A_TRUE) < 0.1
    @test abs(M[2] - B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * N_INLIERS)
end

@testset "Optimal RANSAC error conditions" begin
    # Error when data has fewer points than minimum sample size
    bad_data = rand(2, 1)
    @test_throws ErrorException optimalransac(bad_data, identity,
                                              (M, x, t) -> (model=M, inliers=Int[]), 2, 0.5)

    # t_search < t should throw an ArgumentError
    data     = make_line_data(StableRNG(SEED))
    @test_throws ArgumentError optimalransac(data, fit_line, line_dist, 2, 0.5;
                                             t_search=0.1)
    
    # min_consensus < 2 should throw an ArgumentError
    @test_throws ArgumentError optimalransac(data, fit_line, line_dist, 2, 0.5;
                                             min_consensus=1)

    # Degenerate input: all N points are identical
    degenerate_data = repeat([1.0; 2.0], 1, 10)
    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        optimalransac(degenerate_data, fit_line, line_dist, 2, 0.5;
                      max_trials=1, max_data_trials=1, rng=StableRNG(SEED))
    catch
    end
    @test_throws ErrorException optimalransac(degenerate_data, fit_line, line_dist,
                                              2, 0.5; max_trials=5, max_data_trials=3,
                                              rng=StableRNG(SEED))
end

@testset "Optimal RANSAC verbose output" begin
    rng = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    buf = IOBuffer()
    optimalransac(data, fit_line, line_dist, 2, 0.5;
                  verbose=true, verbose_io=buf, rng=rng)
    output = String(take!(buf))
    @test occursin(r"trial \d+: candidates = \d+, max candidates = \d+, σ = \d+/\d+", output)
end

@testset "Optimal RANSAC degenfn respected" begin
    # degenfn returns true for every sample → should warn and eventually error
    data = make_line_data(StableRNG(SEED))

    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        optimalransac(data, fit_line, line_dist, 2, 0.5;
                      degenfn=_ -> true,
                      max_trials=1, max_data_trials=1,
                      rng=StableRNG(SEED))
    catch
    end
end

@testset "Optimal RANSAC max_trials safety limit" begin
    # With max_trials=1 the algorithm must still return a model if one was found
    rng  = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    # Should not throw; model may be sub-optimal but must be returned
    M, inliers = optimalransac(data, fit_line, line_dist, 2, 0.5;
                               max_trials=5, rng=rng)
    @test !isnothing(M)
    @test !isempty(inliers)
    @test all(1 .≤ inliers .≤ size(data, 2))
end
