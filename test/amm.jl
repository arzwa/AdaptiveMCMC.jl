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


d = 20
S = cov(randn(d, d))
x = randn(d)
target = MvNormal(x, S)
θ = rand(target, 1000)
X = hcat([rand(MvNormal(θ[:,i], 1)) for i in 1:size(θ)[2]]...)

p = Proposals(:θ=>AdaptiveMixtureProposal(d, start=100))
trace = zeros(0, d)
chain = MvNormalChain(X, MvNormal(d,5), State(:θ=>x, :logp=>-Inf), p, 0, trace)

for i=1:10000
    amm_mcmc!(chain, :θ)
    chain.trace = [chain.trace ; chain[:θ]']
end

plot(chain.trace[:, 1:5])
hline!(θ[1:5], linewidth=5, alpha=0.5)
