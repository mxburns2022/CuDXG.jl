export annealed_sinkhorn_log, annealed_sinkhorn_euclidean

function annealed_sinkhorn_log(
    r::AbstractArray{T},
    c::AbstractArray{T},
    W::AbstractMatrix{T},
    args::EOTArgs{T},
    frequency::Int=50,
) where {T<:Real}
    eta_init = args.eta_p
    eta_final = args.epsilon
    eta = max(eta_init, eta_final)
    original_eta = args.eta_p
    time_start = time_ns()
    φ = log.(r)
    ψ = log.(c)
    φws = copy(φ)
    ψws = copy(ψ)
    φprev = copy(φws)
    ψprev = copy(ψws)
    objective = T(NaN)
    q = args.anneal_mult
    preveta = one(T)
    verbose = args.verbose
    args.verbose = false
    total_iters = 0
    if verbose
        println("time(s),iter,infeas,ot_objective,eta,solver")
    end
    while true
        args.eta_p = eta
        _, φ, ψ, objective, residual_val, num_iter = sinkhorn_log(
            r,
            c,
            W,
            args,
            frequency;
            return_state=true,
            φ0=φws,
            ψ0=ψws,
        )
        total_iters += num_iter
        if verbose
            elapsed_time = (time_ns() - time_start) / 1e9
            @printf "%.6e,%d,%.14e,%.14e,%.14e,annealed_sinkhorn\n" elapsed_time total_iters residual_val objective eta
        end
        if eta == eta_final
            break
        end
        φws .= φ + (eta / preveta) .* (φ - φprev)
        ψws .= ψ + (eta / preveta) .* (ψ - ψprev)
        φprev .= φ
        ψprev .= ψ
        preveta = eta
        eta = max(q * eta, eta_final)
    end

    args.verbose = verbose
    args.eta_p = original_eta
    return exp.(-W ./ eta_final .+ φ .+ ψ'), φ, ψ
end

function annealed_sinkhorn_euclidean(
    marginal1::CuArray{T},
    marginal2::CuArray{T},
    location1::CuArray{T},
    location2::CuArray{T},
    out1::String,
    out2::String,
    potentials::String,
    args::EOTArgs,
    frequency::Int,
    p::R,
) where {T<:Real,R}
    eta_init = args.eta_p
    eta_final = args.epsilon
    eta = max(eta_init, eta_final)
    original_eta = args.eta_p
    time_start = time_ns()
    φ = log.(marginal1)
    ψ = log.(marginal2)
    φws = copy(φ)
    ψws = copy(ψ)
    φprev = copy(φws)
    ψprev = copy(ψws)
    objective = T(NaN)
    q = args.anneal_mult
    preveta = one(T)
    verbose = args.verbose
    args.verbose = false
    total_iters = 0
    if verbose
        println("time(s),iter,infeas,ot_objective,eta,solver")
    end
    while true
        args.eta_p = eta
        φ, ψ, objective, residual_val, num_iter = sinkhorn_color_transfer(
            location1,
            location2,
            marginal1,
            marginal2,
            args,
            frequency,
            p;
            return_cuda=true,
            return_assignments=false,
            φ0=φws,
            ψ0=ψws,
        )
        total_iters += num_iter
        if verbose
            elapsed_time = (time_ns() - time_start) / 1e9
            @printf "%.6e,%d,%.14e,%.14e,%.14e,annealed_sinkhorn_kernel\n" elapsed_time total_iters residual_val objective eta
        end
        if eta == eta_final
            break
        end
        φws .= φ + (eta / preveta) .* (φ - φprev)
        ψws .= ψ + (eta / preveta) .* (ψ - ψprev)
        φprev .= φ
        ψprev .= ψ
        preveta = eta
        eta = max(q * eta, eta_final)
    end

    args.verbose = verbose
    args.eta_p = original_eta

    φ_out = Array(φ)
    ψ_out = Array(ψ)
    assignments1 = CUDA.zeros(T, 3, size(location1, 2))
    assignments2 = CUDA.zeros(T, 3, size(location2, 2))
    threads = 256
    naive_blocks1 = div(size(location1, 2), threads, RoundUp)
    naive_blocks2 = div(size(location2, 2), threads, RoundUp)
    residual_cache = CUDA.zeros(T, size(location1, 2))
    @cuda threads = threads blocks = div(size(location1, 2), div(threads, 32, RoundDown), RoundUp) max_logsumexp_spp_ct!(residual_cache, location1, location2, p)
    W∞ = maximum(residual_cache)
    @cuda threads = threads blocks = naive_blocks1 naive_findmaxindex_ct!(assignments1, location1, location2, φ, ψ, eta_final, W∞, p)
    @cuda threads = threads blocks = naive_blocks2 naive_findmaxindex_ct!(assignments2, location2, location1, ψ, φ, eta_final, W∞, p)

    if potentials != ""
        open(potentials * ".row", "w") do outfile
            for φi in φ_out
                println(outfile, φi)
            end
        end
        open(potentials * ".col", "w") do outfile
            for ψj in ψ_out
                println(outfile, ψj)
            end
        end
    end
    if out1 != ""
        assignments1_out = Array(assignments1)
        open(out1, "w") do outfile
            for index in eachindex(assignments1_out)
                println(outfile, "$(assignments1_out),$(index)")
            end
        end
    end
    if out2 != ""
        assignments2_out = Array(assignments2)
        open(out2, "w") do outfile
            for index in eachindex(assignments2_out)
                println(outfile, "$(assignments2_out),$(index)")
            end
        end
    end

    return φ_out, ψ_out, objective
end
