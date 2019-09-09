# Metropolis-within-Gibbs
"""
    Chain

Subtypes of Chain are assumed to have a `state` and `proposals` field, and
should have a logpdf method.
"""
abstract type Chain end

# Assume the state is some dict with keys being symbols, and values scalars/vecs
"""
    mhgibbs(chain::Chain, order::Array{Int64}, p::Symbol; move=rw)

Metropolis-within-Gibbs update for a vector of parameters.
"""
function mhgibbs(chain::Chain, order::Array{Int64}, p::Symbol; move=rw)
    x = deepcopy(chain[p])
    for i in order
        mhr = 0.
        prop = chain.proposals[p, i]
        x[p, i], mhr_ = move(chain[p, i])
        mhr += mhr_
        lp = logpdf(chain, p=>x)
        mhr += lp - chain[:logp]
        if log(rand()) < mhr
            chain[:logp] = lp
            prop.accepted += 1
        else
            x[p, i] = chain[p, i]
        end
        chain.gen % prop.tuninterval == 0 ?
            adapt!(prop) : nothing
    end
    chain[p] = x
end

mhgibbs(chain::Chain, p::Symbol; move=rw) = mhgibbs(
    chain, 1:length(chain[:p]), move=move)
