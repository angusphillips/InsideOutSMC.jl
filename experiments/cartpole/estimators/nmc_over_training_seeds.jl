using InsideOutSMC

using Random
using Distributions
using Statistics

import Flux

using Printf: @printf
using DelimitedFiles

include("../environment.jl")

using .CartpoleEnvironment: init_state
using .CartpoleEnvironment: ibis_dynamics
using .CartpoleEnvironment: param_prior

using JLD2


seed_list_str = get(ENV, "TRAIN_SEEDS", "1,2,3,4,5")
train_seeds = parse.(Int, split(seed_list_str, ','))

nb_runs_per_seed = parse(Int, get(ENV, "NB_NMC_RUNS", get(ENV, "NB_SPCE_RUNS", "32")))
nb_steps = parse(Int, get(ENV, "NB_STEPS", "50"))
nb_outer_samples = parse(Int, get(ENV, "NB_OUTER_SAMPLES", "64"))
nb_inner_samples = parse(Int, get(ENV, "NB_INNER_SAMPLES", "100000"))

csv_path = get(
    ENV,
    "NMC_CSV_PATH",
    "./experiments/cartpole/data/cartpole_nmc_over_training_seeds.csv"
)

seed_means = zeros(length(train_seeds))
seed_stds = zeros(length(train_seeds))

for (i, train_seed) in enumerate(train_seeds)
    policy_path = "./experiments/cartpole/data/cartpole_ibis_csmc_ctl_seed$(train_seed).jld2"

    if !isfile(policy_path)
        error("Missing policy file $(policy_path). Train it first with TRAIN_SEED=$(train_seed).")
    end

    policy = load(policy_path)["ctl"]
    closedloop = IBISClosedLoop(ibis_dynamics, policy)

    nmc_runs = zeros(nb_runs_per_seed)
    for k in 1:nb_runs_per_seed
        Random.seed!(10_000 * train_seed + k)
        Flux.reset!(closedloop.ctl)
        nmc_runs[k] = compute_NMC(
            closedloop,
            param_prior,
            init_state,
            nb_steps,
            nb_outer_samples,
            nb_inner_samples,
        )
    end

    seed_means[i] = mean(nmc_runs)
    seed_stds[i] = std(nmc_runs)

    @printf(
        "train_seed: %i, NMC: %0.4f ± %0.4f\n",
        train_seed,
        seed_means[i],
        seed_stds[i],
    )
end

overall_mean = mean(seed_means)
overall_std = std(seed_means)

@printf("Across training seeds NMC: %0.4f ± %0.4f\n", overall_mean, overall_std)

writedlm(
    csv_path,
    hcat(train_seeds, seed_means, seed_stds),
    ','
)

println("Saved per-seed NMC summary to $(csv_path)")
