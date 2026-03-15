module LocationFindingDADPolicy

    using Random
    using LinearAlgebra

    import DistributionsAD as Distributions

    import Flux

    using InsideOutSMC: StochasticPolicy
    import InsideOutSMC: policy_mean
    import InsideOutSMC: policy_sample
    import InsideOutSMC: policy_logpdf
    import InsideOutSMC: policy_entropy
    import InsideOutSMC: policy_resample_state!

    import Flux: reset!


    mutable struct DADGaussianPolicy{T<:Flux.Chain,S<:Flux.Chain} <: StochasticPolicy
        xdim::Int
        udim::Int
        encoding_dim::Int
        empty_value::Float64
        encoder::T
        emitter::S
        log_std::Vector{Float64}
        encoding_state::Union{Nothing, Matrix{Float64}}
        call_idx::Int
    end


    function build_dad_network(
        input_dim::Int,
        output_dim::Int;
        encoder_shapes::Vector{Int} = [256],
        encoding_dim::Int = 16,
        emitter_shapes::Vector{Int} = Int[],
    )::Tuple{Flux.Chain, Flux.Chain}
        encoder_layers = Any[]
        prev_dim = input_dim
        for hidden_dim in encoder_shapes
            push!(encoder_layers, Flux.Dense(prev_dim, hidden_dim, Flux.relu))
            prev_dim = hidden_dim
        end
        push!(encoder_layers, Flux.Dense(prev_dim, encoding_dim))
        encoder = Flux.f64(Flux.Chain(encoder_layers...))

        emitter_layers = Any[]
        prev_dim = encoding_dim
        for hidden_dim in emitter_shapes
            push!(emitter_layers, Flux.Dense(prev_dim, hidden_dim, Flux.relu))
            prev_dim = hidden_dim
        end
        push!(emitter_layers, Flux.Dense(prev_dim, output_dim))
        emitter = Flux.f64(Flux.Chain(emitter_layers...))

        return encoder, emitter
    end


    function DADGaussianPolicy(
        xdim::Int,
        udim::Int,
        encoding_dim::Int,
        encoder::Flux.Chain,
        emitter::Flux.Chain,
        log_std::AbstractVector{<:Real};
        empty_value::Float64 = 0.01,
    )
        return DADGaussianPolicy(
            xdim,
            udim,
            encoding_dim,
            empty_value,
            encoder,
            emitter,
            Float64.(collect(log_std)),
            nothing,
            1,
        )
    end


    function _ensure_encoding_state!(
        sp::DADGaussianPolicy,
        nb_samples::Int,
    )
        if sp.encoding_state === nothing
            sp.encoding_state = fill(sp.empty_value, sp.encoding_dim, nb_samples)
            return nothing
        end

        @assert sp.encoding_state !== nothing
        if size(sp.encoding_state, 2) != nb_samples
            sp.encoding_state = sp.encoding_state[:, 1:nb_samples]
        end
        return nothing
    end


    function _update_encoding_state!(
        sp::DADGaussianPolicy,
        zs::AbstractMatrix{Float64},
    )
        if sp.call_idx == 1
            return nothing
        end

        prev_y = view(zs, 1:sp.xdim, :)
        prev_x = view(zs, sp.xdim+1:sp.xdim+sp.udim, :)
        enc_input = vcat(prev_y, prev_x)

        @assert sp.encoding_state !== nothing
        sp.encoding_state = sp.encoding_state + sp.encoder(enc_input)
        return nothing
    end


    function _current_action_mean!(
        sp::DADGaussianPolicy,
        zs::AbstractMatrix{Float64},
    )::Matrix{Float64}
        nb_samples = size(zs, 2)
        _ensure_encoding_state!(sp, nb_samples)
        _update_encoding_state!(sp, zs)

        @assert sp.encoding_state !== nothing
        return sp.emitter(sp.encoding_state)
    end


    function policy_mean(
        sp::DADGaussianPolicy,
        z::AbstractVector{Float64},
    )::Vector{Float64}
        means = _current_action_mean!(sp, reshape(z, :, 1))
        sp.call_idx += 1
        return vec(means)
    end


    function policy_sample(
        sp::DADGaussianPolicy,
        zs::AbstractMatrix{Float64},
    )::Matrix{Float64}
        means = _current_action_mean!(sp, zs)
        std = exp.(sp.log_std)

        nb_samples = size(zs, 2)
        noise = randn(sp.udim, nb_samples)
        action = means + std .* noise

        sp.call_idx += 1
        return action
    end


    function policy_logpdf(
        sp::DADGaussianPolicy,
        zs::AbstractMatrix{Float64},
        us::AbstractMatrix{Float64},
    )::Vector{Float64}
        means = _current_action_mean!(sp, zs)
        std = exp.(sp.log_std)

        variances = std .^ 2
        lls = map(eachcol(us), eachcol(means)) do u, m
            dist = Distributions.TuringDenseMvNormal(m, Diagonal(variances))
            Distributions.logpdf(dist, u)
        end

        sp.call_idx += 1
        return collect(lls)
    end


    function policy_entropy(
        sp::DADGaussianPolicy,
    )::Float64
        covar = Diagonal(exp.(sp.log_std) .^ 2)
        return (
            0.5 * sp.udim * log(2 * pi * exp(1))
            + 0.5 * logdet(covar)
        )
    end


    function reset!(
        sp::DADGaussianPolicy,
    )
        sp.encoding_state = nothing
        sp.call_idx = 1
        return nothing
    end


    function policy_resample_state!(
        sp::DADGaussianPolicy,
        resampled_idx::AbstractVector{Int},
    )
        if sp.encoding_state !== nothing
            sp.encoding_state = sp.encoding_state[:, resampled_idx]
        end
        return nothing
    end


    Flux.@functor DADGaussianPolicy (encoder, emitter, log_std)

end
