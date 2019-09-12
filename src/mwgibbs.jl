# Metropolis-within-Gibbs

# Assume the state is some dict with keys being symbols, and values scalars/vecs
"""
    mwgibbs(chain::Chain, order::Array{Int64}, p::Symbol; move=rw)

Metropolis-within-Gibbs update for a vector of parameters. Chain shoudl
implement `logpdf(chain, p=>x, :i=>i)`
"""
function mwgibbs(chain::Chain, order::Array{Int64}, p::Symbol; move=rw)
    x = deepcopy(chain[p])
    for i in order
        mhr = 0.
        prop = chain.proposals[p, i]
        x[p, i], mhr_ = move(chain[p, i])
        mhr += mhr_
        lp = logpdf(chain, p=>x, :i=>i)
        mhr += lp - chain[:logp]
        if log(rand()) < mhr
            chain[:logp] = lp
            prop.accepted += 1
        else
            x[p, i] = chain[p, i]
        end
        consider_adaptation!(prop)
    end
    chain[p] = x
end

mwgibbs(chain::Chain, p::Symbol; move=rw) = mhgibbs(
    chain, 1:length(chain[:p]), move=move)
