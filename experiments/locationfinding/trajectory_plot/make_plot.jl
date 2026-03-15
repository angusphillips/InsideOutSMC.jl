using InsideOutSMC

using Random
using Distributions
using LinearAlgebra

import Flux

include("../environment.jl")
include("../dad_policy.jl")

using .LocationFindingEnvironment: T
using .LocationFindingEnvironment: xdim, udim
using .LocationFindingEnvironment: init_state
using .LocationFindingEnvironment: dynamics

using JLD2
using DelimitedFiles


train_seed = parse(Int, get(ENV, "TRAIN_SEED", "1"))
plot_seed = parse(Int, get(ENV, "PLOT_SEED", "1"))
Random.seed!(plot_seed)

policy_path = "./experiments/locationfinding/data/locationfinding_ibis_csmc_ctl_seed$(train_seed).jld2"
policy = load(policy_path)["ctl"]
Flux.reset!(policy)

nb_steps = parse(Int, get(ENV, "NB_STEPS", string(T)))

trajectory = Array{Float64}(undef, xdim + udim, nb_steps + 1)
trajectory[:, 1] = init_state

Flux.reset!(policy)
for t in 1:nb_steps
    state = trajectory[1:xdim, t]
    action = policy_mean(policy, trajectory[:, t])
    next_state = dynamics_sample(dynamics, state, action)
    trajectory[:, t + 1] = vcat(next_state, action)
end

time_steps = 1:nb_steps+1
seed_csv_path = "./experiments/locationfinding/data/locationfinding_ibis_csmc_trajectory_seed$(train_seed).csv"

writedlm(
    seed_csv_path,
    hcat(time_steps, trajectory'),
    ',',
)

if train_seed == 1
    writedlm(
        "./experiments/locationfinding/data/locationfinding_ibis_csmc_trajectory.csv",
        hcat(time_steps, trajectory'),
        ',',
    )
end

println("Saved trajectory CSV to $(seed_csv_path)")
