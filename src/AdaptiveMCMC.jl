module AdaptiveMCMC

using Distributions

include("proposals.jl")

"""
    Chain

Subtypes of Chain are assumed to have a `state` and `proposals` field, and
should have a `logpdf` method that can take a variable number of pairs as
arguments.
"""
abstract type Chain end


export
    Proposals, AdaptiveUvProposal, AdaptiveRwProposal, AdaptiveUnProposal
    adapt!, rw, scale, reflect

end
