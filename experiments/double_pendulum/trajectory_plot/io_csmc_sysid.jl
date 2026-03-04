using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

import Zygote
import Flux
import Bijectors

include("../environment.jl")

using .DoublePendulumEnvironment: xdim, udim
using .DoublePendulumEnvironment: init_state
using .DoublePendulumEnvironment: dynamics
using .DoublePendulumEnvironment: ibis_dynamics
using .DoublePendulumEnvironment: param_prior
using .DoublePendulumEnvironment: param_proposal
using .DoublePendulumEnvironment: ctl_shift, ctl_scale
using .DoublePendulumEnvironment: ctl_feature_fn

using JLD2


train_seed = parse(Int, get(ENV, "TRAIN_SEED", "1"))
Random.seed!(train_seed)


input_dim = 4
output_dim = 2
recur_size = 64
dense_size = 256

ctl_encoder_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(input_dim, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, recur_size),
        # Flux.LSTM(recur_size, recur_size),
        # Flux.LSTM(recur_size, recur_size),
        Flux.GRU(recur_size => recur_size),
        Flux.GRU(recur_size => recur_size),
    ),
)

ctl_mean_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

ctl_log_std = @. log(sqrt([1.0, 1.0]))

ctl_bijector = (
    Bijectors.Shift(ctl_shift)
    ∘ Bijectors.Scale(ctl_scale)
    ∘ Tanh()
)

learner_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    ctl_log_std,
    ctl_bijector,
)

evaluator_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    [-20.0, -20.0],
    ctl_bijector,
)

learner_loop = IBISClosedLoop(
    ibis_dynamics, learner_policy
)

evaluator_loop = IBISClosedLoop(
    ibis_dynamics, evaluator_policy
)

action_penalty = 0.0
slew_rate_penalty = 0.1
tempering = 0.25

nb_steps = 50
nb_trajectories = 256
nb_particles = 128

nb_ibis_moves = 3
nb_csmc_moves = 1

nb_iter = 25
opt_state = Flux.setup(Flux.Optimise.Adam(5e-4), learner_loop)
batch_size = 64

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
    param_struct.log_likelihoods[:, :, idx]
)

learner_loop, _ = markovian_score_climbing_with_ibis_marginal_dynamics(
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
    true
)

set_ibis_profiling_active!(false)
set_ibis_profiling_phase!(:done)

ibis_stats = get_ibis_profiling_stats()

function phase_total_ibis(stats, phase)
    return get(stats.reweight_likelihood_evals_by_phase, phase, 0) + get(stats.move_likelihood_evals_by_phase, phase, 0)
end

function phase_total(stats, phase)
    return phase_total_ibis(stats, phase) + get(stats.trajectory_reweight_likelihood_evals_by_phase, phase, 0)
end

reference_total = phase_total(ibis_stats, :reference_smc)
markovian_init_total = phase_total(ibis_stats, :markovian_init_csmc)
markovian_loop_total = phase_total(ibis_stats, :markovian_loop_csmc)
csmc_total = markovian_init_total + markovian_loop_total
training_total = ibis_stats.total_likelihood_evals
ibis_total = ibis_stats.ibis_total_likelihood_evals
trajectory_reweight_total = ibis_stats.trajectory_reweight_likelihood_evals

check_mechanism_partition = training_total == ibis_total + trajectory_reweight_total
check_ibis_partition = ibis_total == ibis_stats.reweight_likelihood_evals + ibis_stats.move_likelihood_evals
check_phase_partition = training_total == reference_total + markovian_init_total + markovian_loop_total
check_csmc_partition = csmc_total == markovian_init_total + markovian_loop_total

println("\n=== IBIS training profiling (evaluation loops excluded) ===")
println("reweight calls: ", ibis_stats.reweight_calls)
println("move triggers total: ", ibis_stats.move_triggers_total)
println("move kernel calls: ", ibis_stats.move_kernel_calls)
println("trajectory reweight calls: ", ibis_stats.trajectory_reweight_calls)
println("likelihood evals (IBIS reweight): ", ibis_stats.reweight_likelihood_evals)
println("likelihood evals (IBIS move): ", ibis_stats.move_likelihood_evals)
println("likelihood evals (trajectory reweight SMC/CSMC): ", trajectory_reweight_total)
println("likelihood evals (IBIS total): ", ibis_total)
println("likelihood evals (total training): ", training_total)
println("likelihood evals (reference SMC): ", reference_total)
println("likelihood evals (markovian init CSMC): ", markovian_init_total)
println("likelihood evals (markovian loop CSMC): ", markovian_loop_total)
println("likelihood evals (CSMC training calls total): ", csmc_total)
println("sanity checks: mechanism_partition=", check_mechanism_partition, ", ibis_partition=", check_ibis_partition, ", phase_partition=", check_phase_partition, ", csmc_partition=", check_csmc_partition)
println("phase totals:")
for phase in sort!(collect(keys(merge(
    ibis_stats.reweight_likelihood_evals_by_phase,
    ibis_stats.move_likelihood_evals_by_phase,
    ibis_stats.trajectory_reweight_likelihood_evals_by_phase,
))); by=string)
    reweight_phase = get(ibis_stats.reweight_likelihood_evals_by_phase, phase, 0)
    move_phase = get(ibis_stats.move_likelihood_evals_by_phase, phase, 0)
    trajectory_reweight_phase = get(ibis_stats.trajectory_reweight_likelihood_evals_by_phase, phase, 0)
    total_phase = reweight_phase + move_phase + trajectory_reweight_phase
    println("  ", phase, " => total=", total_phase, ", ibis_reweight=", reweight_phase, ", ibis_move=", move_phase, ", trajectory_reweight=", trajectory_reweight_phase)
end
println("move-trigger counts by time index t:")
for (t, c) in zip(ibis_stats.move_trigger_times, ibis_stats.move_trigger_counts)
    println("  t=", t, " => ", c)
end

seed_output_path = "./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl_seed$(train_seed).jld2"
jldsave(seed_output_path; evaluator_loop.ctl)

if train_seed == 1
    jldsave("./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl.jld2"; evaluator_loop.ctl)
end

println("Saved trained policy to $(seed_output_path)")
