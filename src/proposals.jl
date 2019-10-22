abstract type ProposalKernel end

# container struct
const Proposals = Dict{Symbol,Union{Vector{<:ProposalKernel},<:ProposalKernel}}
const Symmetric = Union{Normal,Uniform}

"""
    AdaptiveUvProposal(accepted::Int64, tuneinterval::Int64, kernel::Normal,
        move::Function)

An adaptive Univariate proposal kernel.
"""
mutable struct AdaptiveUvProposal{T} <: ProposalKernel
    kernel::T
    tuneinterval::Int64
    accepted::Int64
    move::Function
    bounds::Tuple{Float64,Float64}
end

function (prop::AdaptiveUvProposal)(θ)
    return prop.move(prop, θ)
end

# constructors
AdaptiveUvProposal(d::Normal, ti=25, move=rw, bounds=(-Inf, Inf)) =
    AdaptiveUvProposal{Normal}(d, ti, 0., move, bounds)
AdaptiveUvProposal(d::Uniform, ti=25, move=rw, bounds=(-Inf, Inf)) =
    AdaptiveUvProposal{Uniform}(d, ti, 0., move, bounds)


# other helpful proposals
AdaptiveRwProposal(σ=1.0) = AdaptiveUvProposal(Normal(0., σ))
AdaptiveUnProposal(ϵ=0.5) = AdaptiveUvProposal(Uniform(-ϵ, ϵ))
AdaptiveScaleProposal(ϵ=0.5) =
    AdaptiveUvProposal(Uniform(-ϵ, ϵ), 25., scale, (0.,Inf))
AdaptiveUnitProposal(ϵ=0.2) =
    AdaptiveUvProposal(Uniform(-ϵ, ϵ), 25., rw, (0.,1.))


# coevol-like
CoevolProposals(σ=1.0, ti=25) = [
    AdaptiveUvProposal(Normal(0., σ), ti, m) for m in [rw, rwrandom, rwiid]]


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
function rw(k::AdaptiveUvProposal{T}, x::Float64) where T<:Symmetric
    xp = reflect(x + rand(k), k.bounds...)
    return xp, 0.
end

function rw(k::AdaptiveUvProposal{T}, x::Vector{Float64}) where T<:Symmetric
    xp = reflect.(x .+ rand(k), k.bounds...)
    return xp, 0.
end

function rwrandom(k::AdaptiveUvProposal{T}, x::Vector{Float64}) where
        T<:Symmetric
    i = rand(1:length(x))
    xp, r = rw(k, x[i])
    [x[1:i-1] ; xp; x[i+1:end]], r
end

function rwiid(k::AdaptiveUvProposal{T}, x::Vector{Float64}) where
        T<:Symmetric
    xp = reflect.(x .+ rand(k, length(x)), k.bounds...)
    xp, 0.
end


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
