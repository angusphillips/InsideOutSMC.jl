using Random
using Distributions


mutable struct IBISProfilingStats
    active::Bool
    phase::Symbol
    reweight_likelihood_evals::Int
    move_likelihood_evals::Int
    trajectory_reweight_likelihood_evals::Int
    reweight_calls::Int
    move_kernel_calls::Int
    trajectory_reweight_calls::Int
    move_triggers_total::Int
    move_triggers_by_t::Dict{Int,Int}
    reweight_likelihood_evals_by_phase::Dict{Symbol,Int}
    move_likelihood_evals_by_phase::Dict{Symbol,Int}
    trajectory_reweight_likelihood_evals_by_phase::Dict{Symbol,Int}
    reweight_calls_by_phase::Dict{Symbol,Int}
    move_kernel_calls_by_phase::Dict{Symbol,Int}
    trajectory_reweight_calls_by_phase::Dict{Symbol,Int}
    move_triggers_by_t_phase::Dict{Symbol,Dict{Int,Int}}
end


const _ibis_profile_lock = ReentrantLock()
const _ibis_profile_stats = Ref(
    IBISProfilingStats(
        false,
        :unspecified,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        Dict{Int,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Int}(),
        Dict{Symbol,Dict{Int,Int}}(),
    )
)


function reset_ibis_profiling!()
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        stats.active = false
        stats.phase = :unspecified
        stats.reweight_likelihood_evals = 0
        stats.move_likelihood_evals = 0
        stats.trajectory_reweight_likelihood_evals = 0
        stats.reweight_calls = 0
        stats.move_kernel_calls = 0
        stats.trajectory_reweight_calls = 0
        stats.move_triggers_total = 0
        empty!(stats.move_triggers_by_t)
        empty!(stats.reweight_likelihood_evals_by_phase)
        empty!(stats.move_likelihood_evals_by_phase)
        empty!(stats.trajectory_reweight_likelihood_evals_by_phase)
        empty!(stats.reweight_calls_by_phase)
        empty!(stats.move_kernel_calls_by_phase)
        empty!(stats.trajectory_reweight_calls_by_phase)
        empty!(stats.move_triggers_by_t_phase)
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function set_ibis_profiling_active!(active::Bool)
    lock(_ibis_profile_lock)
    try
        _ibis_profile_stats[].active = active
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function set_ibis_profiling_phase!(phase::Symbol)
    lock(_ibis_profile_lock)
    try
        _ibis_profile_stats[].phase = phase
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function get_ibis_profiling_stats()
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        move_triggers_by_t = copy(stats.move_triggers_by_t)
        move_trigger_times = sort(collect(keys(move_triggers_by_t)))
        move_trigger_counts = [move_triggers_by_t[t] for t in move_trigger_times]
        ibis_total_likelihood_evals = stats.reweight_likelihood_evals + stats.move_likelihood_evals
        total_likelihood_evals = ibis_total_likelihood_evals + stats.trajectory_reweight_likelihood_evals

        return (
            active=stats.active,
            reweight_calls=stats.reweight_calls,
            move_kernel_calls=stats.move_kernel_calls,
            trajectory_reweight_calls=stats.trajectory_reweight_calls,
            move_triggers_total=stats.move_triggers_total,
            move_triggers_by_t=move_triggers_by_t,
            move_trigger_times=move_trigger_times,
            move_trigger_counts=move_trigger_counts,
            reweight_likelihood_evals=stats.reweight_likelihood_evals,
            move_likelihood_evals=stats.move_likelihood_evals,
            ibis_total_likelihood_evals=ibis_total_likelihood_evals,
            trajectory_reweight_likelihood_evals=stats.trajectory_reweight_likelihood_evals,
            total_likelihood_evals=total_likelihood_evals,
            phase=stats.phase,
            reweight_likelihood_evals_by_phase=copy(stats.reweight_likelihood_evals_by_phase),
            move_likelihood_evals_by_phase=copy(stats.move_likelihood_evals_by_phase),
            trajectory_reweight_likelihood_evals_by_phase=copy(stats.trajectory_reweight_likelihood_evals_by_phase),
            reweight_calls_by_phase=copy(stats.reweight_calls_by_phase),
            move_kernel_calls_by_phase=copy(stats.move_kernel_calls_by_phase),
            trajectory_reweight_calls_by_phase=copy(stats.trajectory_reweight_calls_by_phase),
            move_triggers_by_t_phase=Dict(
                phase => copy(counts)
                for (phase, counts) in stats.move_triggers_by_t_phase
            ),
        )
    finally
        unlock(_ibis_profile_lock)
    end
end


function _record_reweight_likelihood_evals!(nb_particles::Int)
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        if stats.active
            phase = stats.phase
            stats.reweight_calls += 1
            stats.reweight_likelihood_evals += nb_particles
            stats.reweight_calls_by_phase[phase] = get(stats.reweight_calls_by_phase, phase, 0) + 1
            stats.reweight_likelihood_evals_by_phase[phase] = get(stats.reweight_likelihood_evals_by_phase, phase, 0) + nb_particles
        end
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function _record_move_trigger!(time_idx::Int)
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        if stats.active
            stats.move_triggers_total += 1
            stats.move_triggers_by_t[time_idx] = get(stats.move_triggers_by_t, time_idx, 0) + 1
            phase = stats.phase
            if !haskey(stats.move_triggers_by_t_phase, phase)
                stats.move_triggers_by_t_phase[phase] = Dict{Int,Int}()
            end
            phase_dict = stats.move_triggers_by_t_phase[phase]
            phase_dict[time_idx] = get(phase_dict, time_idx, 0) + 1
        end
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function _record_move_likelihood_evals!(history_steps::Int, nb_particles::Int)
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        if stats.active
            stats.move_kernel_calls += 1
            stats.move_likelihood_evals += history_steps * nb_particles
            phase = stats.phase
            stats.move_kernel_calls_by_phase[phase] = get(stats.move_kernel_calls_by_phase, phase, 0) + 1
            stats.move_likelihood_evals_by_phase[phase] = get(stats.move_likelihood_evals_by_phase, phase, 0) + history_steps * nb_particles
        end
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function _record_trajectory_reweight_likelihood_evals!(nb_likelihood_evals::Int, nb_calls::Int)
    lock(_ibis_profile_lock)
    try
        stats = _ibis_profile_stats[]
        if stats.active
            phase = stats.phase
            stats.trajectory_reweight_calls += nb_calls
            stats.trajectory_reweight_likelihood_evals += nb_likelihood_evals
            stats.trajectory_reweight_calls_by_phase[phase] = get(stats.trajectory_reweight_calls_by_phase, phase, 0) + nb_calls
            stats.trajectory_reweight_likelihood_evals_by_phase[phase] = get(stats.trajectory_reweight_likelihood_evals_by_phase, phase, 0) + nb_likelihood_evals
        end
    finally
        unlock(_ibis_profile_lock)
    end
    return nothing
end


function batch_ibis!(
    trajectories::AbstractArray{Float64,3},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct
) where {T<:Function}

    _, _, nb_trajectories = size(trajectories)
    for traj_idx = 1:nb_trajectories
        ibis!(
            view(trajectories, :, :, traj_idx),
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            view_struct(param_struct, traj_idx),
        )
    end
end


function ibis!(
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct
) where {T<:Function}

    _, nb_steps_p_1 = size(trajectory)
    nb_steps = nb_steps_p_1 - 1

    for time_idx = 1:nb_steps
        ibis_step!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            param_struct,
        )
    end
end


function batch_ibis_step!(
    time_idx::Int,
    trajectories::AbstractArray{Float64,3},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}

    _, _, nb_trajectories = size(trajectories)
    for traj_idx = 1:nb_trajectories
        ibis_step!(
            time_idx,
            view(trajectories, :, :, traj_idx),
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            view_struct(param_struct, traj_idx),
        )
    end
end


function ibis_step!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}

    # 1. Reweight
    reweight_params!(
        time_idx,
        trajectory,
        dynamics,
        param_struct,
    )

    # 2. Resample-move if necessary
    weights = @view param_struct.weights[time_idx+1, :]
    if effective_sample_size(weights) < 0.75 * param_struct.nb_particles
        _record_move_trigger!(time_idx)
        resample_params!(
            time_idx,
            param_struct
        )
        move!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            param_struct,
        )
    end
end


function reweight_params!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_struct::IBISParamStruct,
)
    xdim = dynamics.xdim

    # Get the log weight increments
    log_weight_increments = ibis_conditional_dynamics_logpdf(
        dynamics,
        param_struct.particles[:, time_idx, :],
        trajectory[begin:xdim, time_idx],  # state
        trajectory[xdim+1:end, time_idx+1],    # action
        trajectory[begin:xdim, time_idx+1],    # next_state
        param_struct.scratch
    )  # Assumes parameter-independent noise
    _record_reweight_likelihood_evals!(param_struct.nb_particles)

    # Copy over particles and weights to next time step
    param_struct.particles[:, time_idx+1, :] = param_struct.particles[:, time_idx, :]
    param_struct.weights[time_idx+1, :] = param_struct.weights[time_idx, :]
    param_struct.log_weights[time_idx+1, :] = param_struct.log_weights[time_idx, :]
    param_struct.log_likelihoods[time_idx+1, :] = param_struct.log_likelihoods[time_idx, :]

    # Update log_weights and log_likelihoods
    param_struct.log_weights[time_idx+1, :] += log_weight_increments
    param_struct.log_likelihoods[time_idx+1, :] += log_weight_increments

    # Normalize the updated weights
    normalize_weights!(time_idx+1, param_struct)
end


function resample_params!(
    time_idx::Int,
    param_struct::IBISParamStruct,
)
    systematic_resampling!(time_idx+1, param_struct)

    param_struct.weights[time_idx+1, :] .= 1 / param_struct.nb_particles
    param_struct.log_weights[time_idx+1, :] .= 0.0
    param_struct.particles[:, time_idx+1, :] .= param_struct.particles[:, time_idx+1, param_struct.resampled_idx]
    param_struct.log_likelihoods[time_idx+1, :] .= param_struct.log_likelihoods[time_idx+1, param_struct.resampled_idx]
end


function move!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}
    for j = 1:nb_moves
        kernel!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            param_struct,
        )
    end
end


function kernel!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    param_struct::IBISParamStruct,
) where {T<:Function}

    history = @view trajectory[:, 1:time_idx+1]
    particles = @view param_struct.particles[:, time_idx+1, :]
    weights = @view param_struct.weights[time_idx+1, :]
    prop_particles = param_proposal(particles, weights)

    # populate uniforms
    rand!(param_struct.rvs)

    prop_log_likelihood = accumulate_likelihood(
        history,
        prop_particles,
        dynamics,
        param_prior,
        param_struct.scratch
    )

    acceptance_rate = 0.0
    @views @inbounds for m = 1:param_struct.nb_particles
        log_ratio = prop_log_likelihood[m] - param_struct.log_likelihoods[time_idx+1, m]
        if log(param_struct.rvs[m]) < log_ratio
            param_struct.particles[:, time_idx+1, m] = prop_particles[:, m]
            param_struct.log_likelihoods[time_idx+1, m] = prop_log_likelihood[m]
            acceptance_rate += 1.0
        end
    end
    # println(acceptance_rate / param_struct.nb_particles)
end


function accumulate_likelihood(
    history::AbstractMatrix{Float64},
    particles::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    scratch::AbstractMatrix{Float64},
)
    lls = logpdf(param_prior, particles)

    xdim = dynamics.xdim
    _, nb_steps_p_1 = size(history)
    nb_particles = size(particles, 2)
    _record_move_likelihood_evals!(nb_steps_p_1 - 1, nb_particles)

    @views @inbounds for t in 1:nb_steps_p_1 - 1
        lls += ibis_conditional_dynamics_logpdf(
            dynamics,
            particles,
            history[begin:xdim, t],    # state
            history[xdim+1:end, t+1],  # action
            history[begin:xdim, t+1],  # next_state
            scratch
        )  # Assumes parameter-independent noise
    end
    return lls
end
