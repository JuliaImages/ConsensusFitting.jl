```@meta
CurrentModule = ConsensusFitting
```

# Optimal RANSAC

## Background

**Optimal RANSAC** is a repeatable variant of the RANSAC algorithm introduced
by [Hast2013](@citet).  Standard RANSAC is inherently non-deterministic: because
it selects random minimal samples, two runs on the same data can find different
inlier sets, which is problematic when the algorithm is embedded in a larger
pipeline that must be reproducible.  Optimal RANSAC addresses this by augmenting
the standard hypothesis-and-verify loop with three iterative refinement steps
that steer the search toward the *global* optimal inlier set, making repeated
convergence to the same result much more likely than in the standard RANSAC
algorithm.

### The algorithm

Optimal RANSAC shares its outer structure with standard RANSAC but, crucially,
replaces the single scoring step with three tightly coupled sub-procedures
whenever a candidate hypothesis yields more than a small number of tentative
inliers.

#### 1. Resample (Algorithm 2)

A random subset of up to a quarter of the current tentative-inlier set is
drawn and a new model is fitted to that subset.  The new model is then scored
**against the full dataset** using a (optionally wider) search tolerance
``t_{\mathrm{search}}``.  If this produces a larger inlier set the resampling
loop restarts from the expanded set; otherwise it tries up to 8 different
subsets before returning.  This is directly inspired by the Local-Optimisation
step of LO-RANSAC [Chum2003, Chum2004](@cite) but is triggered more aggressively: whenever
a promising hypothesis is found rather than only after a global best is
updated.

#### 2. Rescore (Algorithm 3)

Starting from the inlier set produced by resampling, the model is iteratively
re-estimated from *all* current inliers and scored against all data until the
inlier set stops changing or after at most 20 iterations.  Using all inliers
for fitting (rather than only the minimal sample ``s``) exploits the fact that
many problems admit a least-squares fit from more than ``s`` points, producing
a model that is better aligned to the true structure.

#### 3. Pruneset (Algorithm 4) — optional

When a wider search tolerance ``t_{\mathrm{search}} > t`` is used alongside a
user-supplied residual function, a final pruning pass removes any inlier whose
residual exceeds the tight tolerance ``t``.  The most extreme inlier is removed
one at a time and the model is re-estimated after each removal, so that the
final model is always consistent with the retained inliers.

### Stopping criterion

Instead of the adaptive ``N`` iterations formula used by standard RANSAC,
Optimal RANSAC terminates when the same inlier-set *size* is re-discovered a
specified number of times in succession (`min_consensus`, default 2).  The
refinement steps make it very unlikely to find the same size by chance unless
it corresponds to the true optimal set, so this simple criterion works
surprisingly well in practice.

### When to use Optimal RANSAC

Optimal RANSAC is preferable to standard RANSAC when

- **repeatability** is important regardless of RNG state (standard RANSAC
  is repeatible given the same seeded RNG, but generally not otherwise, while
  Optimal RANSAC is designed to be repeatible regardless of RNG state),
- the inlier fraction is **very low** (well below 5%), making 
  RANSAC's standard adaptive stopping criterion unreliable, or
- a **high-precision** final inlier set is needed (using the two-tolerance
  mode with `t_search > t` and `residualfn`).

It is less appropriate when

- the model fitting step (`fittingfn`) does not support over-determined input
  (required for the rescore step), or
- the data contains **multiple competing structures** of comparable size, in
  which case the algorithm may converge to a locally optimal set.

## API

```@docs
optimalransac
```

## References
This page cites the following references:

```@bibliography
Pages = ["optimalransac.md"]
Canonical = false
```