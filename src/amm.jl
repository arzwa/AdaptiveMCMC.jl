"""
    AdaptiveMixtureProposal
"""
@with_kw mutable struct AdaptiveMixtureProposal <: ProposalKernel
    d  ::Int  # dimension
    Σ  ::Matrix{Float64} = ones(d,d) + I # covariance matrix
    σ  ::Float64         = 0.01 # fixed component σ
    β  ::Float64         = 0.1  # mixture weight
    Ex ::Vector{Float64} = zeros(d)
    Exx::Matrix{Float64} = zeros(d,d)
    n  ::Int64           = 0 # generation
    accepted::Int64      = 0
    start::Int64         = 50
end

AdaptiveMixtureProposal(d::Int64; σ=0.1, β=0.1, start=100) =
    AdaptiveMixtureProposal(d, zeros(d,d), σ,β, zeros(d), zeros(d,d), 0,0,start)

function (prop::AdaptiveMixtureProposal)(θ::Vector{Float64})
    # update covariance based on input vector
    update_covariance!(prop, θ)
    @unpack d, Σ, σ, β, Ex, Exx, n, accepted, start = prop
    x = rand(MvNormal(zeros(d), σ/d))
    if n > start
        x = β*x + (1. - β)*rand(MvNormal((2.38^2/d)*Σ))
    end
    θ + x
end

# NOTE: this algorithm is not numerically stable in general
function update_covariance!(p::AdaptiveMixtureProposal, θ::Vector{Float64})
    n = p.n + 1
    Exx = ((n-1)*p.Exx + θ*θ')/n
    Ex = ((n-1)*p.Ex + θ)/n
    p.Σ = n*(Exx - Ex*Ex')/(n-1)
    p.Ex = Ex
    p.Exx = Exx
    p.n = n
end

function amm_mcmc!(chain, p)
    prop = chain.proposals[p]
    x = prop(chain[p])
    lp = logpdf(chain, p=>x)
    mhr = lp - chain[:logp]
    if log(rand()) < mhr
        chain[p] = x
        chain[:logp] = lp
        prop.accepted += 1
    end
end
