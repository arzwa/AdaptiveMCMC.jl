module AdaptiveMCMC

    using Distributions
    using Parameters

    include("proposals.jl")
    include("amm.jl")

    """
        Chain

    Subtypes of Chain are assumed to have a `state` and `proposals` field, and
    should have a `logpdf` method that can take a variable number of pairs as
    arguments.
    """
    abstract type Chain end

    Base.getindex(w::Chain, s::Symbol) = w.state[s]
    Base.getindex(w::Chain, s::Symbol, i::Int64) = w.state[s][i]
    Base.setindex!(w::Chain, x, s::Symbol) = w.state[s] = x
    Base.setindex!(w::Chain, x, s::Symbol, i::Int64) = w.state[s][i] = x
    Base.display(io::IO, w::Chain) = print("$(typeof(w))($(w.state))")
    Base.show(io::IO, w::Chain) = write(io, "$(typeof(w))($(w.state))")

    export
        AdaptiveUvProposal, AdaptiveRwProposal, AdaptiveUnProposal,
        AdaptiveUnitProposal, AdaptiveScaleProposal, adapt!, rw, scale, Chain,
        reflect, consider_adaptation!, AdaptiveMixtureProposal, amm_mcmc!,
        CoevolUnProposals, CoevolRwProposals

end
