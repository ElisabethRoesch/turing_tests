using Turing, LinearAlgebra, Plots, StatsPlots, DataFrames, Optim, DynamicHMC;

# Include additional files
include("/Users/mcoomer/phd/2d_random_walks/code/calculations.jl")
include("../code/bprw_2d.jl")

# SET GLOBAL PARAMETERS
steps = 50
walks = 20
bias = [80.23,75.22]

###############################################################################
# GENERATE DATA
###############################################################################
# BRW (W=0.9, P=1, B=40)
###############################################################################
params = [0.9, 5, 40]
data, sources, persists = [], [], []
for i in 1:walks
    v = bprw_2d(steps, bias, params)
    trajectory = v[3]
    s = v[9]
    ps = v[10]
    append!(data, trajectory)
    append!(sources, s)
    append!(persists, ps)
end
###############################################################################

###############################################################################
# 1. LOG LIKELIHOOD FUNCTION (MCMC)
function loglikelihood_2d(data, sources, persists, w, p, b)
    kb = b
    kp = p
    numofdatapoints = length(data)
    result = 0.0
    for d = 2:numofdatapoints
        result += logpdf.(MixtureModel(VonMises, [(sources[d], kb), (persists[d-1], kp)], [w, (1-w)]), data[d])
    end
    return result
end
###############################################################################
# 2. DEFINE THE MCMC MODEL
@model mcmc_bprw_2d(data, sources, persists) = begin
    # Define prior for b
    b ~ Distributions.Uniform(eps(), 50.0)
    # Define prior for p
    p ~ Distributions.Uniform(eps(), 50.0)
    # Define prior for w
    w ~ Distributions.Uniform(0.0, 1.0)

    # Define distribution that data is sampled from in bprw_2d model
    for d = 2:length(data)
        data[d] ~ MixtureModel(VonMises, [(sources[d], b), (persists[d-1], p)], [w, (1-w)])
    end

    Turing.@addlogprob! loglikelihood_2d(data, sources, persists, w, p, b)
end
###############################################################################

###############################################################################
# 3. RUN MCMC USING NUTS SAMPLER
###############################################################################
num_chains = 4
chain = mapreduce(
    c -> sample(mcmc_bprw_2d(data,sources,persists), NUTS(200, 0.5), 10_000, discard_adapt=false),
    chainscat,
    1:num_chains);
gelmandiag(chain)
println(mean(chain[:acceptance_rate]))
StatsPlots.plot(chain)
###############################################################################
