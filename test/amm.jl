using AdaptiveMCMC
using Distributions
using Plots

mutable struct MvNormalChain <: Chain
    X::Array{Float64,2}
    prior::MvNormal
    state::State
    proposals::Proposals
    gen::Int64
    trace::Array{Float64,2}
end

function Distributions.logpdf(chain::MvNormalChain, args...)
    k, v = args[1]
    logpdf(chain.prior, v) + sum(logpdf(MvNormal(v, 1), chain.X))
end


d = 2
n = 1000
θ = rand(MvNormal([1 0.9 ; 0.9 1]), 5)
X = hcat([[rand(Normal(θ[1,i])), rand(Normal(θ[2,i]))] for i in 1:n]...)

p = Proposals(:θ=>AdaptiveMixtureProposal(d, start=100))
trace = zeros(0, d)
chain = MvNormalChain(X, MvNormal(d,5),
    State(:θ=>rand(d), :logp=>-Inf), p, 0, trace)

for i=1:10000
    amm_mcmc!(chain, :θ)
    chain.trace = [chain.trace ; chain[:θ]']
end

histogram(chain.trace[1000:end,1])
histogram!(chain.trace[1000:end,2])
