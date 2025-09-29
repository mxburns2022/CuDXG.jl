using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools


function accelerated_sinkhorn(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50) where R
    # input 
    # WScaled = W
    n = size(r, 1)
    K = -W ./ args.ηp
    if isa(W, CuArray)
        u = CUDA.zeros(R, n)
        v = CUDA.zeros(R, n)
        ǔ = CUDA.zeros(R, n)
        v̌ = CUDA.zeros(R, n)
        ũ = CUDA.zeros(R, n)
        ṽ = CUDA.zeros(R, n)
        residual_r = CUDA.zeros(R, n)
        residual_c = CUDA.zeros(R, n)
    else
        u = zeros(R, n)
        v = zeros(R, n)
        ǔ = zeros(R, n)
        v̌ = zeros(R, n)
        ũ = zeros(R, n)
        ṽ = zeros(R, n)
        residual_r = zeros(R, n)
        residual_c = zeros(R, n)
    end
    θ = 1
    function residual!(u, v)
        normv = sum(softmax(K .+ u .+ v'; normalize_values=true))
        residual_r .= softmax(K .+ u .+ v'; dims=2, normalize_values=true) ./ normv - r
        residual_c .= softmax(K .+ u .+ v'; dims=1, normalize_values=true)' ./ normv - c
    end
    function dual_value(u, v)
        return -sum(r'u) - sum(c'v) + sum(logsumexp(K .+ u .+ v'))
    end
    for i in 1:args.tmax
        if (i - 1) % frequency == 0
            p = exp.(K .+ ǔ .+ v̌')
            pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            # println(ψ'c, " ", φ'r)
            pobj = dot(pr, W)
            # pdgap = -pobj + dobj
            @printf "%d,%.14e,%.14e,-1,accelerated_sinkhorn\n" i feas pobj
        end

        ū = (1 - θ) .* ǔ + θ .* ũ
        v̄ = (1 - θ) .* v̌ + θ .* ṽ
        residual!(ū, v̄)
        ũ .-= 1 / 2θ .* residual_r
        ṽ .-= 1 / 2θ .* residual_c
        û = ū - 1 / 2 * (residual_r)
        v̂ = v̄ - 1 / 2 * (residual_c)
        if i % 2 == 0
            û = û + log.(r) - logsumexp(K .+ û .+ v̂', 2)
        else
            v̂ = v̂ + log.(c) - logsumexp(K .+ û .+ v̂', 1)'
        end
        if dual_value(ǔ, v̌) < dual_value(û, v̂)
            u .= ǔ
            v .= v̌
        else
            u .= û
            v .= v̂
        end
        if i % 2 == 0
            ǔ = u + log.(r) - logsumexp(K .+ u .+ v', 2)
            v̌ .= v
        else
            v̌ = v + log.(c) - logsumexp(K .+ u .+ v', 1)'
            ǔ .= u
        end
        θ = θ * (sqrt(θ^2 + 4) - θ) / 2
    end

    p = exp.(K .+ u .+ v')
    return p, u, v
end