module ConsensusFitting

using Random: AbstractRNG, default_rng, randperm

export ransac, optimalransac, iusac

include("RANSAC.jl")
include("OptimalRANSAC.jl")
include("IUSAC.jl")

end
