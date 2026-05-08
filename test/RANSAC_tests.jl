using ConsensusFitting
using Random
using StableRNGs: StableRNG
using Test

# Constants for shared test setup
const A_TRUE = 2.0
const B_TRUE = 3.0
const N_INLIERS = 100
const N_OUTLIERS = 40
const SEED = 42

# Shared test fixture for generating line data
function make_line_data(rng)
    x_in = collect(range(-10.0, 10.0; length=N_INLIERS))
    y_in = A_TRUE .* x_in .+ B_TRUE .+ 0.2 .* randn(rng, N_INLIERS)
    x_out = -10.0 .+ 20.0 .* rand(rng, N_OUTLIERS)
    y_out = -25.0 .+ 50.0 .* rand(rng, N_OUTLIERS)
    return [vcat(x_in, x_out)'; vcat(y_in, y_out)']
end

# fittingfn: fit y = a*x + b through exactly 2 points
function fit_line(pts)
    x1, y1 = pts[1, 1], pts[2, 1]
    x2, y2 = pts[1, 2], pts[2, 2]
    isapprox(x1, x2; atol=1e-10) && return []  # vertical → degenerate
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return [a, b]
end

# distfn: classify using vertical (y) residual
function line_dist(M, x, t)
    a, b = M[1], M[2]
    residuals = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(residuals .< t)
    return (model=M, inliers=inliers)
end

@testset "RANSAC line fitting" begin
    rng = StableRNG(SEED)
    data = make_line_data(rng)

    M, inliers = ransac(data, fit_line, line_dist, 2, 0.5; rng=rng)

    # Recovered slope and intercept should be close to the true values
    @test abs(M[1] - A_TRUE) < 0.1
    @test abs(M[2] - B_TRUE) < 0.1

    # Should recover the majority of the inlier points
    @test length(inliers) ≥ round(Int, 0.9 * N_INLIERS)

    # All returned indices must be valid column indices of data
    @test all(1 .≤ inliers .≤ size(data, 2))
end

@testset "RANSAC with higher-dimensional input" begin
    rng = StableRNG(SEED)
    data = make_line_data(rng)

    # Pack the data into a (1, 2, N) array to test higher-dimensional input
    data_3d = reshape(data, 1, 2, :)
    M_3d, inliers_3d = ransac(data_3d, 
                              x -> fit_line(reshape(x, 2, 2)), 
                              (M, x, t) -> line_dist(M, reshape(x, 2, :), t), 2, 0.5; rng=rng)
    @test abs(M_3d[1] - A_TRUE) < 0.1
    @test abs(M_3d[2] - B_TRUE) < 0.1
    @test length(inliers_3d) ≥ round(Int, 0.9 * N_INLIERS)
end

@testset "RANSAC error conditions" begin
    rng = StableRNG(SEED)
    # Error on too few points
    bad_data = rand(2, 1)  # only 1 point, need s=2
    @test_throws ErrorException ransac(bad_data, identity, (M, x, t) -> (model=M, inliers=Int[]), 2, 0.5)

    # Error on degenerate input: all N points are the same
    degenerate_data = repeat([1.0; 2.0], 1, 10)

    # Warning is emitted every time max_data_trials are exhausted without a
    # valid model. The exception is caught here so we can test the two failure
    # modes independently.
    @test_logs (:warn, r"could not draw a non-degenerate sample after") try
        ransac(degenerate_data, fit_line, line_dist, 2, 0.5;
               max_trials=1, max_data_trials=1, rng=rng)
    catch
    end

    # ErrorException is thrown after all outer iterations fail.
    @test_throws ErrorException ransac(degenerate_data, fit_line, line_dist, 2, 0.5;
                                       max_trials=5, max_data_trials=3,
                                       rng=rng)

end

@testset "RANSAC verbose output" begin
    rng = StableRNG(SEED)
    data = make_line_data(rng)
    Random.seed!(rng, SEED)   # reset for algorithm

    buf = IOBuffer()
    ransac(data, fit_line, line_dist, 2, 0.5;
           verbose=true, verbose_io=buf, rng=rng)
    output = String(take!(buf))
    @test occursin(r"trial \d+ out of estimated \d+ required", output)
end
