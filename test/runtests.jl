using ConsensusFitting
using Random
using Test
using SafeTestsets: @safetestset

# Run doctests first
using Documenter: DocMeta, doctest
DocMeta.setdocmeta!(ConsensusFitting, :DocTestSetup, :(using ConsensusFitting); recursive=true)
doctest(ConsensusFitting)

@testset "ConsensusFitting.jl" verbose=true begin
    @safetestset "RANSAC tests" include("RANSAC_tests.jl")
    @safetestset "Optimal RANSAC tests" include("OptimalRANSAC_tests.jl")
end
