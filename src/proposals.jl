abstract type ProposalKernel end

# container struct
const Proposals = Dict{Symbol,Union{Vector{<:ProposalKernel},<:ProposalKernel}}
const Symmetric = Union{Normal,Uniform}

"""
    AdaptiveUvProposal(accepted::Int64, tuneinterval::Int64, kernel::Normal)

An adaptive Univariate proposal kernel. To be used with the `rw!()` (random
walk) and `scale!()` update functions.
"""
mutable struct AdaptiveUvProposal{T} <: ProposalKernel
    kernel::T
    tuneinterval::Int64
    accepted::Int64
end

AdaptiveRwProposal() = AdaptiveUvProposal(Normal())
AdaptiveUnProposal() = AdaptiveUvProposal(Uniform(-0.5, 0.5))
AdaptiveUvProposal(d::Normal, ti=50) = AdaptiveUvProposal{Normal}(d, ti, 0.)
AdaptiveUvProposal(d::Uniform, ti=50) = AdaptiveUvProposal{Uniform}(d, ti, 0.)

Base.rand(prop::ProposalKernel) = rand(prop.kernel)
Base.rand(prop::ProposalKernel, n::Int64) = rand(prop.kernel, n)
Base.getindex(spl::Proposals, s::Symbol, i::Int64) = spl[s][i]

function adapt!(x::AdaptiveUvProposal{T}, gen::Int64,
        target=0.2, bound=10., δmax=0.25) where T<:Distribution
    gen == 0 ? (return) : nothing
    δn = min(δmax, 1. /√(gen/x.tuneinterval))
    α = x.accepted / x.tuneinterval
    lσ = α > target ? log(hyperp(x)) + δn : log(hyperp(x)) - δn
    lσ = abs(lσ) > bound ? sign(lσ) * bound : lσ
    x.kernel = adapted(x.kernel, lσ)
    x.accepted = 0
end

consider_adaptation!(prop::ProposalKernel, generation::Int) =
    generation % prop.tuneinterval == 0 ? adapt!(prop, generation) : nothing

hyperp(x::AdaptiveUvProposal{Uniform}) = x.kernel.b
hyperp(x::AdaptiveUvProposal{Normal}) = x.kernel.σ
adapted(kernel::Uniform, lσ::Float64) = Uniform(-exp(lσ), exp(lσ))
adapted(kernel::Normal, lσ::Float64) = Normal(0., exp(lσ))

# Random walk proposals
rw(k::AdaptiveUvProposal{T}, x::Float64) where T<:Symmetric = x + rand(k), 0.
rw(k::AdaptiveUvProposal{T}, x::Vector{Float64}) where T<:Symmetric =
    x .+ rand(k), 0.

# scaling proposals
function scale(k::AdaptiveUvProposal{Uniform}, x::Float64)
    xp = x*exp(rand(k))
    return xp, log(xp) - log(x)
end

function scale(k::AdaptiveUvProposal{Uniform}, x::Vector{Float64})
    xp = x .* exp(rand(k))
    return xp, sum(log.(xp)) - sum(log.(x))
end

function reflect(x::Float64, a::Float64=0., b::Float64=1.)
    while !(a <= x <= b)
        x = x < a ? 2a - x : x
        x = x > b ? 2b - x : x
    end
    return x
end
