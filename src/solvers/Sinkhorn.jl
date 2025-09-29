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
    K = -W ./ args.ηp
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
    for i in 1:args.tmax
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
        if args.verbose && (i - 1) % frequency == 0
            ψ = reshape(log.(c) - logsumexp(K .+ φ, 1)', (n,))
            p = exp.(K .+ φ .+ ψ')
            pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            # println(ψ'c, " ", φ'r)
            pobj = dot(pr, W)
            # pdgap = -pobj + dobj
            @printf "%d,%.14e,%.14e,-1,sinkhorn\n" i feas pobj
        end

    end
    ψ = log.(c) - logsumexp(K .+ φ, 1)'
    φ = log.(r) - logsumexp(K .+ ψ', 2)
    return exp.(K .+ φ .+ ψ'), φ, ψ
end
