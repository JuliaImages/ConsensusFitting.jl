using ConsensusFitting
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style=:authoryear)

DocMeta.setdocmeta!(ConsensusFitting, :DocTestSetup, :(using ConsensusFitting); recursive=true)

makedocs(;
    modules=[ConsensusFitting],
    authors="cgarling <chris.t.garling@gmail.com> and contributors",
    sitename="ConsensusFitting.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical="https://JuliaAstro.org/ConsensusFitting.jl/stable/",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "RANSAC" => "ransac.md",
        "Optimal RANSAC" => "optimalransac.md",
        "Bibliography" => "refs.md",
    ],
    plugins=[bib],
    doctest=false,
    linkcheck=true,
    warnonly=[:missing_docs, :linkcheck],
)

deploydocs(;
    repo="github.com/JuliaAstro/ConsensusFitting.jl.git",
    devbranch="main",
)
