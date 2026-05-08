using ConsensusFitting
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
using Test

# ── Shared test fixture ────────────────────────────────────────────────────────
# True model: y = a_true * x + b_true
const _IUSAC_A_TRUE = 2.0
const _IUSAC_B_TRUE = 3.0
const _IUSAC_N_INLIERS  = 100
const _IUSAC_N_OUTLIERS = 40
const _IUSAC_SEED = 4321

function _iusac_make_line_data(rng)
    x_in  = collect(range(-10.0, 10.0; length=_IUSAC_N_INLIERS))
    y_in  = _IUSAC_A_TRUE .* x_in .+ _IUSAC_B_TRUE .+ 0.2 .* randn(rng, _IUSAC_N_INLIERS)
    x_out = -10.0 .+ 20.0 .* rand(rng, _IUSAC_N_OUTLIERS)
    y_out = -25.0 .+ 50.0 .* rand(rng, _IUSAC_N_OUTLIERS)
    return [vcat(x_in, x_out)'; vcat(y_in, y_out)']
end

# fittingfn: fit y = a*x + b through ≥ 2 points using least squares
function _iusac_fit_line(pts)
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
function _iusac_line_dist(M, x, t)
    a, b = M[1], M[2]
    residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(residuals .< t)
    return (model=M, inliers=inliers)
end

@testset "IUSAC line fitting" begin
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    Random.seed!(rng, _IUSAC_SEED)   # reset for algorithm

    M, inliers = iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5; rng=rng)

    # Recovered slope and intercept should be close to the true values
    @test abs(M[1] - _IUSAC_A_TRUE) < 0.1
    @test abs(M[2] - _IUSAC_B_TRUE) < 0.1

    # Should recover the majority of the inlier points
    @test length(inliers) ≥ round(Int, 0.9 * _IUSAC_N_INLIERS)

    # All returned indices must be valid column indices of data
    @test all(1 .≤ inliers .≤ size(data, 2))
end

@testset "IUSAC repeatability" begin
    # A key property of IUSAC: repeated runs with DIFFERENT random seeds should
    # usually converge to the same inlier set (the optimal set).
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    seeds = 1:10  # different seeds to test repeatability across

    results = map(seeds) do seed
        iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
              rng=StableRNG(seed))
    end

    # All runs should find approximately the same model when the data has a
    # single dominant structure.
    slopes     = [r[1][1] for r in results]
    intercepts = [r[1][2] for r in results]
    @test mean(slopes)     ≈ _IUSAC_A_TRUE atol=0.05
    @test mean(intercepts) ≈ _IUSAC_B_TRUE atol=0.05
    @test std(slopes)     < 0.05
    @test std(intercepts) < 0.1
end

@testset "IUSAC determinism with fixed seed" begin
    data = _iusac_make_line_data(StableRNG(_IUSAC_SEED))

    M1, inliers1 = iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5; rng=StableRNG(_IUSAC_SEED))
    M2, inliers2 = iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5; rng=StableRNG(_IUSAC_SEED))

    # Bit-for-bit identical results with the same seed
    @test M1 == M2
    @test inliers1 == inliers2
end

@testset "IUSAC higher-dimensional input" begin
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    Random.seed!(rng, _IUSAC_SEED)   # reset for algorithm

    # Reshape to (1, 2, N) to verify arbitrary leading dimensions are handled
    data_3d = reshape(data, 1, 2, :)
    M, inliers = iusac(data_3d,
                       x -> _iusac_fit_line(reshape(x, 2, size(x, 3))),
                       (M, x, t) -> _iusac_line_dist(M, reshape(x, 2, size(x, 3)), t),
                       2, 0.5; rng=rng)
    @test abs(M[1] - _IUSAC_A_TRUE) < 0.1
    @test abs(M[2] - _IUSAC_B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * _IUSAC_N_INLIERS)
end

@testset "IUSAC eta_b early stopping" begin
    # With eta_b set to a low fraction the algorithm should stop as soon as
    # it finds a consensus set of that size.
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    Random.seed!(rng, _IUSAC_SEED)

    # eta_b = 0.5 means stop once 50% of data points are inliers; the
    # inlier fraction is ~71% (100/140), so this should trigger early.
    M, inliers = iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
                       rng=rng, eta_b=0.5)
    @test abs(M[1] - _IUSAC_A_TRUE) < 0.1
    @test abs(M[2] - _IUSAC_B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.5 * size(data, 2))
end

@testset "IUSAC verbose output" begin
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    Random.seed!(rng, _IUSAC_SEED)

    buf = IOBuffer()
    iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
          verbose=true, verbose_io=buf, rng=rng)
    output = String(take!(buf))
    @test occursin(r"trial \d+ out of estimated", output) || occursin(r"trial \d+: inliers", output)
end

@testset "IUSAC error conditions" begin
    # Error when data has fewer points than minimum sample size
    bad_data = rand(2, 1)
    @test_throws ErrorException iusac(bad_data, identity,
                                      (M, x, t) -> (model=M, inliers=Int[]), 2, 0.5)

    # Error on degenerate input: all points are the same
    degenerate_data = repeat([1.0; 2.0], 1, 10)

    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        iusac(degenerate_data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
              max_trials=1, max_data_trials=1, rng=StableRNG(1))
    catch
    end

    @test_throws ErrorException iusac(degenerate_data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
                                      max_trials=5, max_data_trials=3,
                                      rng=StableRNG(1))
end

@testset "IUSAC epsilon convergence parameter" begin
    # With epsilon = 0.0, the inner loop only terminates when the consensus
    # set stops growing or shrinks (strict condition), otherwise behaviour
    # is the same as the default.
    rng  = StableRNG(_IUSAC_SEED)
    data = _iusac_make_line_data(rng)
    Random.seed!(rng, _IUSAC_SEED)

    M, inliers = iusac(data, _iusac_fit_line, _iusac_line_dist, 2, 0.5;
                       rng=rng, epsilon=0.0)
    @test abs(M[1] - _IUSAC_A_TRUE) < 0.1
    @test abs(M[2] - _IUSAC_B_TRUE) < 0.1
    @test length(inliers) ≥ round(Int, 0.9 * _IUSAC_N_INLIERS)
end
