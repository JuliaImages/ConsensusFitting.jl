```@meta
CurrentModule = ConsensusFitting
```

# RANSAC

## Background

**RANSAC** (Random Sample Consensus) is a robust model-fitting algorithm
introduced by [Fischler1981](@cite).  Unlike ordinary least-squares methods,
which are sensitive to outliers, RANSAC explicitly divides the data into
*inliers* (points well explained by the model) and *outliers* (points that are
not), and seeks the model that maximises the size of the inlier set.  It has
become a foundational tool in computer vision, photogrammetry, and geometric
estimation; a thorough treatment can be found in [Hartley2004](@cite).

### The algorithm

Given a dataset of ``N`` points, the minimum number of points ``s`` needed to
uniquely determine a model (e.g. ``s = 2`` for a line, ``s = 3`` for a plane),
and an inlier distance threshold ``t``, one iteration of RANSAC proceeds as
follows:

1. **Hypothesis** — draw a random minimal sample of ``s`` points and fit a
   candidate model to them.
2. **Verification** — evaluate every data point against the candidate model;
   collect those within distance ``t`` as its *consensus set* (inlier set).
3. **Update** — if the consensus set is larger than the current best, record
   the candidate model and its inlier set.

The number of iterations is chosen adaptively so that, with probability ``p``
(typically 0.99), at least one drawn sample is free from outliers.  If the
current best inlier fraction is ``\varepsilon``, the expected number of trials
required is

```math
N = \frac{\log(1 - p)}{\log\!\left(1 - \varepsilon^s\right)}.
```

RANSAC updates this estimate after each improvement, allowing it to terminate
early when a satisfying model is found quickly.

### When to use RANSAC

RANSAC is well suited to problems where

- the fraction of outliers is large (up to 50 % or more),
- the inlier noise is small relative to the outlier spread, and
- the fitting step for the minimal sample is fast.

It is less suitable when the minimum sample size ``s`` is large (the required
number of trials grows exponentially with ``s``) or when computing the
full-dataset residuals is expensive.

---

## API

```@docs
ransac
```

---

## Example: fitting a line in the presence of outliers

The following example generates 100 inlier points near the line
``y = 2x + 3``, adds 100 outliers scattered over a wider region, and uses
RANSAC to recover the true parameters.

```@example ransac_line
using ConsensusFitting
using Random
using CairoMakie

Random.seed!(42)

# ── Generate synthetic data ────────────────────────────────────────────────
a_true, b_true = 2.0, 3.0
n_inliers  = 100
n_outliers = 100

x_in = collect(range(-10.0, 10.0; length=n_inliers))
y_in = a_true .* x_in .+ b_true .+ randn(n_inliers)

x_out = -10.0 .+ 20.0 .* rand(n_outliers)
y_out = -25.0 .+ 50.0 .* rand(n_outliers)

# Pack into a 2 × N matrix (each column is one data point [x; y])
data = [vcat(x_in, x_out)'; vcat(y_in, y_out)']

# ── Define the fitting and distance functions ──────────────────────────────

# Fit a line y = a*x + b through exactly 2 points.
function fit_line(pts)
    x1, y1 = pts[1, 1], pts[2, 1]
    x2, y2 = pts[1, 2], pts[2, 2]
    isapprox(x1, x2; atol=1e-10) && return []   # vertical line → degenerate
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return [a, b]
end

# Classify points using their vertical (y-direction) residual.
function line_dist(M, x, t)
    a, b    = M[1], M[2]
    resid   = abs.(x[2, :] .- (a .* x[1, :] .+ b))
    inliers = findall(resid .< t)
    return inliers, M
end

# ── Run RANSAC ─────────────────────────────────────────────────────────────
M, inliers = ransac(data, fit_line, line_dist, 2, 2.0)

println("Recovered slope:     ", round(M[1]; digits=4), "  (true: $a_true)")
println("Recovered intercept: ", round(M[2]; digits=4), "  (true: $b_true)")
println("Inliers identified:  ", length(inliers), " / $(size(data, 2))")
```

### Visualising the result

```@example ransac_line
outlier_mask = trues(size(data, 2))
outlier_mask[inliers] .= false

fig = Figure(size=(500, 500))
ax  = Axis(fig[1, 1];
           xlabel="x", ylabel="y",
           title="RANSAC line fitting")

# All data points
scatter!(ax, data[1, outlier_mask], data[2, outlier_mask];
         color=(:tomato, 0.7), markersize=8, label="Outliers")
scatter!(ax, data[1, inliers], data[2, inliers];
         color=(:steelblue, 0.8), markersize=8, label="Inliers")

# True and recovered lines
x_plot = range(-10.0, 10.0; length=200)
lines!(ax, x_plot, M[1] .* x_plot .+ M[2];
       color=:orange, linewidth=2.5, label="RANSAC fit")
lines!(ax, x_plot, a_true .* x_plot .+ b_true;
       color=:black, linewidth=2, linestyle=:dash, label="True line")

axislegend(ax; position=:lt)
fig
```

Once the inlier set has been identified, a final fit can be performed to the set of all inliers if desired.

```@example ransac_line
using LinearAlgebra: dot

"""
    fit_line_overconstrained(data)

Fit a line y = a + b*x to data given as a 2×N matrix:
    data[1, :] = x
    data[2, :] = y

# Returns
  - `θ` :[a, b] (slope, intercept)
  - `Σθ`: covariance matrix of θ (2×2)
  - `σ²`: estimated residual variance
"""
function fit_line_overconstrained(data)
    @assert size(data, 1) == 2 "Input must be a 2×N matrix"
    x = view(data, 1, :)
    y = view(data, 2, :)
    N = length(x)
    @assert N ≥ 2 "Need at least two points"
    # Least-squares solution
    A = hcat(x, ones(N)) # Design matrix: [x 1]
    θ = A \ y   # equivalent to (A'A)^(-1)A'y but more stable
    r = y - A * θ # Residuals
    # Degrees of freedom: N - number of parameters (2)
    dof = N - 2
    @assert dof > 0 "Need more than 2 points for covariance estimate"
    # Residual variance estimate
    σ² = dot(r, r) / dof
    # Covariance matrix: σ² * (A'A)^(-1)
    Σθ = σ² * inv(A' * A)
    return θ, Σθ, σ²
end

μ, Σθ, σ² = fit_line_overconstrained(data[:, inliers])
println("Recovered slope:     ", round(μ[1]; digits=4), " ± ", round(sqrt(Σθ[1,1]); digits=4), "  (true: $a_true)")
println("Recovered intercept:     ", round(μ[2]; digits=4), " ± ", round(sqrt(Σθ[2,2]); digits=4), "  (true: $b_true)")
```

---

## References
This page cites the following references:

```@bibliography
Pages = ["ransac.md"]
Canonical = false
```