module AdaptiveMCMC

    using Distributions

    include("proposals.jl")

    export
        Proposals, AdaptiveUvProposal, AdaptiveRwProposal, AdaptiveUnProposal
        adapt!, rw, scale

end
