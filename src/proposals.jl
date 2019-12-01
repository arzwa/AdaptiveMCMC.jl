abstract type ProposalKernel end

# container struct
const Symmetric = Union{Normal,Uniform}

"""
    AdaptiveUvProposal(kernel)

An adaptive Univariate proposal kernel.
"""
@with_kw mutable struct AdaptiveUvProposal{T,V} <: ProposalKernel
    kernel      ::T                       = Normal()
    move        ::V                       = rw
    tuneinterval::Int64                   = 25
    total       ::Int64                   = 0
    accepted    ::Int64                   = 0
    bounds      ::Tuple{Float64,Float64}  = (-Inf, Inf)
    δmax        ::Float64                 = 0.2
    logbound    ::Float64                 = 10.
    target      ::Float64                 = 0.36
    stop        ::Int64                   = 1000
end

function (prop::AdaptiveUvProposal)(θ)
    consider_adaptation!(prop)
    prop.total += 1
    return prop.move(prop, θ)
end

# other helpful constructors
AdaptiveRwProposal(σ=1.0, ti=25, stop=1000) = AdaptiveUvProposal(
    kernel=Normal(0., σ), tuneinterval=ti, stop=stop)
AdaptiveUnProposal(ϵ=0.5, ti=25, stop=1000) = AdaptiveUvProposal(
    kernel=Uniform(-ϵ, ϵ), tuneinterval=ti, stop=stop)
AdaptiveScaleProposal(ϵ=0.5, ti=25, stop=1000) = AdaptiveUvProposal(
    kernel=Uniform(-ϵ, ϵ), move=scale, bounds=(0.,Inf), tuneinterval=ti, stop=stop)
AdaptiveUnitProposal(ϵ=0.2, ti=25, stop=1000) = AdaptiveUvProposal(
    kernel=Uniform(-ϵ, ϵ), bounds=(0.,1.), tuneinterval=ti, stop=stop)

# coevol-like
CoevolRwProposals(σ=[1.0, 1.0, 1.0], ti=25, stop=1000) = AdaptiveUvProposal[
    AdaptiveUvProposal(move=m, kernel=Normal(0., s), tuneinterval=ti, stop=stop)
        for (m, s) in zip([rw, rwrandom, rwiid], σ)]
CoevolUnProposals(ϵ=[1.0, 1.0, 1.0], ti=25, stop=1000) = AdaptiveUvProposal[
    AdaptiveUvProposal(move=m, kernel=Uniform(-e, e),tuneinterval=ti, stop=stop)
        for (m, e) in zip([rw, rwrandom, rwiid], ϵ)]

# reversible jump Beluga
DecreaseProposal(δ=0.5, ti=25, stop=1000) = AdaptiveUvProposal(
    kernel=Uniform(-δ,δ), tuneinterval=ti, move=decrease, stop=stop)

# Base extensions
Base.rand(prop::ProposalKernel) = rand(prop.kernel)
Base.rand(prop::ProposalKernel, n::Int64) = rand(prop.kernel, n)

function adapt!(x::AdaptiveUvProposal{T}) where T<:Distribution
    @unpack total, tuneinterval, accepted, δmax, target, logbound = x
    total == 0 ? (return) : nothing
    δn = min(δmax, 1. /√(total/tuneinterval))
    α = accepted / tuneinterval
    lσ = α > target ? log(hyperp(x)) + δn : log(hyperp(x)) - δn
    lσ = abs(lσ) > logbound ? sign(lσ) * logbound : lσ
    x.kernel = adapted(x.kernel, lσ)
    x.accepted = 0
end

consider_adaptation!(prop::ProposalKernel) =
    (prop.total <= prop.stop && prop.total % prop.tuneinterval == 0) ?
        adapt!(prop) : nothing

hyperp(x::AdaptiveUvProposal{Uniform{T}}) where T<:Real = x.kernel.b
hyperp(x::AdaptiveUvProposal{Normal{T}}) where T<:Real = x.kernel.σ
adapted(kernel::Uniform, lσ::Float64) = Uniform(-exp(lσ), exp(lσ))
adapted(kernel::Normal, lσ::Float64) = Normal(0., exp(lσ))

# independence proposal
independent(k::AdaptiveUvProposal, x::Float64) = rand(k.kernel), 0.
independent(k::AdaptiveUvProposal) = rand(k.kernel), 0.

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

decrease(k::AdaptiveUvProposal, x::Float64) = x - abs(rand(k)), NaN

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
