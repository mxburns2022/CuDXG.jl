using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools


function sinkhorn_log(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50) where {R}
    # input 
    # WScaled = W
    n = size(r, 1)
    K = -W ./ args.eta_p
    if isa(W, CuArray)
        φ = CUDA.zeros(R, n)
        cache1 = CUDA.zeros(R, n)
        cache2 = CUDA.zeros(R, n)
        maxcache = CUDA.zeros(R, n)
    else
        φ = zeros(R, n)
        cache1 = zeros(R, n)
        cache2 = zeros(R, n)
        maxcache = zeros(R, n)
    end
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    for i in 1:args.itermax
        # logsumexp!(cache1', maxcache', K .+ φ, 1)
        # logsumexp!(cache2, maxcache, K .+ (log.(c) - cache1)', 2)
        # φ .= log.(r) - cache2
        # (log.(c) - logsumexp(K .+ φ, 1)')', 2)
        φ = log.(r) - logsumexp(K .+
                                (log.(c) - logsumexp(K .+ φ, 1)')', 2)
        if i % 2000 == 0
            ψ = reshape(log.(c) - logsumexp(K .+ φ, 1)', (n,))
            p = exp.(K .+ φ .+ ψ')
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
        end
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end

        if args.verbose && (i - 1) % frequency == 0
            ψ = reshape(log.(c) - logsumexp(K .+ φ, 1)', n)
            p = exp.(K .+ φ .+ ψ')
            # pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            # println(ψ'c, " ", φ'r)
            obj = dot(p, W)
            pobj = obj + args.eta_p * sum(neg_entropy(p))
            # println()
            dobj = args.eta_p * (sum(-logsumexp(K .+ φ .+ ψ')) - c'ψ - sum(r'φ))
            # pdgap = -pobj + dobj
            @printf "%.6g,%d,%.14e,%.14e,%.14e,%.14e,sinkhorn\n" elapsed_time i feas obj pobj dobj
            if pobj + dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end


    end
    ψ = log.(c) - logsumexp(K .+ φ, 1)'
    φ = log.(r) - logsumexp(K .+ ψ', 2)
    return exp.(K .+ φ .+ ψ'), φ, ψ
end
