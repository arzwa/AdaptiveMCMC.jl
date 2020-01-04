module AdaptiveMCMC

    using Distributions
    using Parameters
    using LinearAlgebra

    include("proposals.jl")
    include("amm.jl")

    export
        AdaptiveUvProposal,
        AdaptiveRwProposal,
        AdaptiveUnProposal,
        AdaptiveUnitProposal,
        AdaptiveScaleProposal,
        AdaptiveMixtureProposal,
        CoevolUnProposals,
        CoevolRwProposals,
        WgdProposals
end
