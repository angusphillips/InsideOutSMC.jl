using InsideOutSMC

using Random
using Distributions
using Statistics
using LinearAlgebra
using StatsBase

import Flux

include("../environment.jl")
include("../dad_policy.jl")

using .LocationFindingEnvironment: T, d, k
using .LocationFindingEnvironment: xdim, udim
using .LocationFindingEnvironment: init_state
using .LocationFindingEnvironment: ibis_dynamics
using .LocationFindingEnvironment: param_prior
using .LocationFindingEnvironment: param_proposal
using .LocationFindingEnvironment: design_sigma

using .LocationFindingDADPolicy: DADGaussianPolicy
using .LocationFindingDADPolicy: build_dad_network

using JLD2


train_seed = parse(Int, get(ENV, "TRAIN_SEED", "1"))
Random.seed!(train_seed)

input_dim = xdim + udim
output_dim = udim

encoder_shapes = [256]
encoding_dim = 16
emitter_shapes = Int[]
empty_value = 0.01

ctl_encoder_fn, ctl_emitter_fn = build_dad_network(
    input_dim,
    output_dim;
    encoder_shapes = encoder_shapes,
    encoding_dim = encoding_dim,
    emitter_shapes = emitter_shapes,
)

learner_log_std = fill(log(design_sigma), udim)
evaluator_log_std = fill(-20.0, udim)

learner_policy = DADGaussianPolicy(
    xdim,
    udim,
    encoding_dim,
    ctl_encoder_fn,
    ctl_emitter_fn,
    learner_log_std;
    empty_value = empty_value,
)

evaluator_policy = DADGaussianPolicy(
    xdim,
    udim,
    encoding_dim,
    ctl_encoder_fn,
    ctl_emitter_fn,
    evaluator_log_std;
    empty_value = empty_value,
)

learner_loop = IBISClosedLoop(
    ibis_dynamics,
    learner_policy,
)

evaluator_loop = IBISClosedLoop(
    ibis_dynamics,
    evaluator_policy,
)

action_penalty = parse(Float64, get(ENV, "ACTION_PENALTY", "0.0"))
slew_rate_penalty = parse(Float64, get(ENV, "SLEW_RATE_PENALTY", "0.0"))
tempering = parse(Float64, get(ENV, "TEMPERING", "0.25"))

nb_steps = parse(Int, get(ENV, "NB_STEPS", string(T)))
nb_trajectories = parse(Int, get(ENV, "NB_TRAJECTORIES", "256"))
nb_particles = parse(Int, get(ENV, "NB_PARTICLES", "1024"))

nb_ibis_moves = parse(Int, get(ENV, "NB_IBIS_MOVES", "3"))
nb_csmc_moves = parse(Int, get(ENV, "NB_CSMC_MOVES", "1"))

nb_iter = parse(Int, get(ENV, "NB_ITER", "20"))
batch_size = parse(Int, get(ENV, "BATCH_SIZE", "64"))
learning_rate = parse(Float64, get(ENV, "LR", "5e-4"))

opt_state = Flux.setup(Flux.Optimise.Adam(learning_rate), learner_loop)

reset_ibis_profiling!()
set_ibis_profiling_active!(true)
set_ibis_profiling_phase!(:reference_smc)

Flux.reset!(learner_loop.ctl)
state_struct, param_struct = smc_with_ibis_marginal_dynamics(
    nb_steps,
    nb_trajectories,
    nb_particles,
    init_state,
    learner_loop,
    param_prior,
    param_proposal,
    nb_ibis_moves,
    action_penalty,
    slew_rate_penalty,
    tempering,
)

idx = rand(Categorical(state_struct.weights))
reference = IBISReference(
    state_struct.trajectories[:, :, idx],
    param_struct.particles[:, :, :, idx],
    param_struct.weights[:, :, idx],
    param_struct.log_weights[:, :, idx],
    param_struct.log_likelihoods[:, :, idx],
)

learner_loop, all_returns = markovian_score_climbing_with_ibis_marginal_dynamics(
    nb_iter,
    opt_state,
    batch_size,
    nb_steps,
    nb_trajectories,
    nb_particles,
    init_state,
    learner_loop,
    evaluator_loop,
    param_prior,
    param_proposal,
    nb_ibis_moves,
    action_penalty,
    slew_rate_penalty,
    tempering,
    reference,
    nb_csmc_moves,
    true,
)

set_ibis_profiling_active!(false)
set_ibis_profiling_phase!(:done)

seed_output_path = "./experiments/locationfinding/data/locationfinding_ibis_csmc_ctl_seed$(train_seed).jld2"
jldsave(seed_output_path; ctl = evaluator_loop.ctl, returns = all_returns)

if train_seed == 1
    jldsave(
        "./experiments/locationfinding/data/locationfinding_ibis_csmc_ctl.jld2";
        ctl = evaluator_loop.ctl,
        returns = all_returns,
    )
end

println("Saved trained policy to $(seed_output_path)")
