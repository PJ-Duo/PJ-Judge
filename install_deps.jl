import Pkg; 
Pkg.add([
    "DataFrames", 
    "CSV", 
    "LinearAlgebra", 
    "Crayons", 
    "BenchmarkTools", 
    "Word2Vec", 
    "TextAnalysis", 
    "StatsBase"
    ])

# Custom TextModels pkg fork
Pkg.add(url = "https://github.com/p0ryae/TextModels.jl")