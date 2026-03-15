module LocationFindingEnvironment

    using Random
    using Distributions
    using LinearAlgebra
    using StatsBase

    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: IBISDynamics

    # Strict-parity defaults from policy_learning/config/bed_model/location_finding.yaml
    T = 30
    d = 2
    k = 2

    sigma = 0.5
    design_sigma = 1.0
    alpha = 1.0
    max_signal = 1e-4
    base_intensity = 1e-1

    xdim = 1
    udim = d

    step_size = 1.0
    init_state = [0.0, 0.0, 0.0]

    # Fixed source configuration used for trajectory export only.
    true_params = [-0.75, 0.75, 0.75, -0.75]

    function unpack_theta(
        p::AbstractVector{Float64}
    )::Matrix{Float64}
        return reshape(p, k, d)
    end

    function intensity_function(
        u::AbstractVector{Float64},
        theta::AbstractMatrix{Float64}
    )::Float64
        intensity = base_intensity
        @inbounds for source_idx in 1:k
            source = view(theta, source_idx, :)
            sq_dist = sum((u .- source) .^ 2)
            intensity += alpha / (max_signal + sq_dist)
        end
        return intensity
    end

    function log_intensity_mean(
        p::AbstractVector{Float64},
        u::AbstractVector{Float64}
    )::Float64
        theta = unpack_theta(p)
        return log(intensity_function(u, theta))
    end

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )::Vector{Float64}
        mean_y = log_intensity_mean(p, u)
        return [mean_y - x[1]]
    end

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        mean_y = log_intensity_mean(p, u)
        xn[1] = mean_y - x[1]
    end

    function diffusion_fn(
        args::AbstractVector{Float64}...,
    )::Vector{Float64}
        return [sigma]
    end

    param_prior = MvNormal(
        zeros(k * d),
        Matrix{Float64}(I, k * d, k * d),
    )

    function param_proposal(
        particles::AbstractMatrix{Float64},
        weights::AbstractVector{Float64},
        constant::Float64 = 1.0,
    )::Matrix{Float64}
        covar = cov(Matrix(particles), AnalyticWeights(weights), 2)
        eig_vals, eig_vecs = eigen(covar)
        sqrt_eig_vals = @. sqrt(max(eig_vals, 1e-8))
        sqrt_covar = eig_vecs * Diagonal(sqrt_eig_vals)
        return particles .+ constant .* sqrt_covar * randn(size(particles))
    end

    dynamics = StochasticDynamics(
        xdim,
        udim,
        (x, u) -> drift_fn(true_params, x, u),
        diffusion_fn,
        step_size,
    )

    ibis_dynamics = IBISDynamics(
        xdim,
        udim,
        drift_fn!,
        diffusion_fn,
        step_size,
    )

    ctl_shift, ctl_scale = [0.0, 0.0], [design_sigma, design_sigma]

    function ctl_feature_fn(
        z::AbstractVector{Float64}
    )::Vector{Float64}
        return z[1:xdim+udim]
    end

end
