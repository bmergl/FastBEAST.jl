using TestItems
using Test
using FastBEAST
using StaticArrays

@testitem "Minimal ACA" tags=[:minimal] begin
    include("./test_aca.jl")
end

@testitem "Minimal H-Matrix" tags=[:minimal] begin
    include("./test_hmatrix.jl")
end

@testitem "Minimal FMM" tags=[:minimal] begin
    include("./test_fmm.jl")
end

@testitem "Minimal BEAST" tags=[:minimal] begin
    include("./test_beast.jl")
end

@testitem "Minimal Tree" tags=[:minimal] begin
    include("./test_tree.jl")
end

