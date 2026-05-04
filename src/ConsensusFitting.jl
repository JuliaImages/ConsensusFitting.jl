module ConsensusFitting

using Random: AbstractRNG, default_rng, randperm

export ransac, optimalransac

include("RANSAC.jl")
include("OptimalRANSAC.jl")

end
