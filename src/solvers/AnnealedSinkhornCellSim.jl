function annealed_sinkhorn_cellsim(
    data1::CuArray{T},
    data2::CuArray{T},
    scale::T,
    metric::Symbol,
    marginal1::CuArray{T},
    marginal2::CuArray{T},
    args::EOTArgs{T},
    frequency::Int=100,
) where T<:Real
    eta_init = args.eta_p
    eta_final = args.epsilon
    eta = max(eta_init, eta_final)
    original_eta = args.eta_p
    time_start = time_ns()
    φ = log.(marginal1)
    ψ = log.(marginal2)
    φws = log.(marginal1)
    ψws = log.(marginal2)
    φprev = copy(φws)
    ψprev = copy(ψws)
    objective = T(NaN)
    q = args.anneal_mult
    preveta = 1.0
    verbose = args.verbose
    args.verbose = false
    total_iters = 0
    if verbose
        println("time(s),iter,infeas,ot_objective,eta,solver")
    end
    while true
        args.eta_p = eta
        φ, ψ, objective, residual_val, num_iter = sinkhorn_cellsim(data1,
                                          data2, 
                                          scale, 
                                          metric, 
                                          marginal1, 
                                          marginal2, 
                                          args, 
                                          frequency;
                                          return_cuda=true,
                                          φ0=φws,
                                          ψ0=ψws)
        total_iters += num_iter
        if verbose
            elapsed_time = (time_ns() - time_start) / 1e9
            @printf "%.6e,%d,%.14e,%.14e,%.14e,annealed_sinkhorn_cellsim\n" elapsed_time total_iters residual_val objective eta
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
    return Array(φ), Array(ψ), objective
end
