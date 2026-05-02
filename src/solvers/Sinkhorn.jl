using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools


function sinkhorn_log(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    return_state::Bool=false,
    φ0::Union{AbstractArray{R}, Nothing}=nothing,
    ψ0::Union{AbstractArray{R}, Nothing}=nothing) where {R}
    # input 
    # WScaled = W
    n = size(r, 1)
    K = -W ./ args.eta_p

    if isa(W, CuArray)
        φ = isnothing(φ0) ? CUDA.zeros(R, n) : copy(φ0)
        ψ = isnothing(ψ0) ? CUDA.zeros(R, n) : copy(ψ0)
        cache1 = CUDA.zeros(R, n)
        cache2 = CUDA.zeros(R, n)
        maxcache = CUDA.zeros(R, n)
    else
        φ = isnothing(φ0) ? zeros(R, n) : copy(φ0)
        ψ = isnothing(ψ0) ? zeros(R, n) : copy(ψ0)
        cache1 = zeros(R, n)
        cache2 = zeros(R, n)
        maxcache = zeros(R, n)
    end
    if args.verbose
        println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    end
    time_start = time_ns()
    num_iter = 0
    println("REEEE")
    for i in 1:args.itermax
        # logsumexp!(cache1', maxcache', K .+ φ, 1)
        # logsumexp!(cache2, maxcache, K .+ (log.(c) - cache1)', 2)
        # φ .= log.(r) - cache2
        # (log.(c) - logsumexp(K .+ φ, 1)')', 2)
        ψ = reshape(log.(c) - logsumexp(K .+ φ, 1)', n)
        φ = log.(r) - logsumexp(K .+ ψ', 2)
        if i % 2000 == 0
            p = exp.(K .+ φ .+ ψ')
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
        end
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        # println()
        if (i - 1) % frequency == 0
            p = exp.(K .+ φ .+ ψ')
            # pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            # println(ψ'c, " ", φ'r)
            obj = dot(p, W)
            pobj = obj + args.eta_p * sum(neg_entropy(p))
            # println()
            dobj = -args.eta_p * (sum(-logsumexp(K .+ φ .+ ψ')) - c'ψ - sum(r'φ))
            # pdgap = -pobj + dobj
            if args.verbose
                @printf "%.6g,%d,%.14e,%.14e,%.14e,%.14e,sinkhorn\n" elapsed_time i feas obj pobj dobj
            end
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end

        num_iter += 1
    end
    ψ = log.(c) - logsumexp(K .+ φ, 1)'
    φ = log.(r) - logsumexp(K .+ ψ', 2)
    p = exp.(K .+ φ .+ ψ')
    if return_state
        feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
        obj = dot(p, W)
        return p, φ, ψ, obj, feas, num_iter
    end
    return p, φ, ψ
end
