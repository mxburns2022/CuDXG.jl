using CuTransferEOT
# include("src/Utilities.jl")
using Random
using PythonOT
using CUDA
using ProgressBars
using Test
using LinearAlgebra

function generate_random_ot(N, rng)
    r = normalize(rand(rng, N), 1)
    c = normalize(rand(rng, N), 1)
    W = abs.(randn(rng, N, N))
    optimum = emd2(r, c, W)
    return r, c, W, optimum
end
function logsumexp(x::AbstractArray{T}, dims=Nothing) where T<:Number
    if dims == Nothing
        dims = 1:length(size(x))
    end
    maxx = maximum(x, dims=dims)
    v1 = log.(sum(exp.(x .- maxx), dims=dims)) .+ maxx
    return v1
end
function softmax(x::AbstractArray{T}; normalize_values=true, dims=[], norm_dims=Nothing) where T<:Real
    if norm_dims == Nothing
        norm_dims = [1:ndims(x)...]
    end

    if !normalize_values
        return sum(exp.(x), dims=dims)
    else
        maxx = maximum(x, dims=norm_dims)
        v1 = sum(exp.(x .- maxx), dims=dims)
        return v1 ./ sum(v1, dims=norm_dims)
    end
end
η = 1e-3
function weighted_primal_kl(p, p2, r)
    hp = dot(r .* p, log.(p ./ p2))
end
function entropy(p)
    return dot(p, log.(p .+ 1e-20))
end
function weighted_kl(μ, ν, W, s, r, c)
    # c = colvec(ν, W, s, r)
    # @test sum(c) ≈ 1.0 atol = 1e-10
    cv = c #.+ 0.01 / size(μ, 1)
    p = softmax(-(0.5 * s * W .+ (2μ .- 1)') ./ η, norm_dims=2)
    p2 = softmax(-(0.5 * s * W .+ (2ν .- 1)') ./ η, norm_dims=2)
    hp = dot(r .* p, log.(p ./ p2))
    hp2 = sum([r[i] .* dot(p[i, 1:end], log.(p[i, 1:end] ./ p2[i, 1:end])) for i in 1:size(r, 1)])
    println(hp)
    println(hp2)

    return hp .+ dot(μ .* c, log.(μ ./ ν)) + dot((1 .- μ) .* c, log.((1 .- μ) ./ (1 .- ν)))
end
function g(μ, W, s, r)
    return η * dot(r, logsumexp(-(0.5 * s .* W .+ μ' .- (1 .- μ)') ./ η, 2))
end

function colvec(ν, W, s, r)
    p = r .* softmax(-(0.5 * s .* W .+ ν' .- (1 .- ν)') ./ η, norm_dims=2)
    c = reshape(sum(p', dims=2), size(r, 1))
    return c
end
function linearization(μ, ν, W, s, r)
    c = colvec(ν, W, s, r)
    return g(μ, W, s, r) - g(ν, W, s, r) + 2dot(c, μ - ν)
end
function project(μ)
    mup = max.(μ, exp(-1) * max(μ, 1 .- μ))
    mun = max.((1 .- μ), exp(-1) * max(μ, 1 .- μ))
    # println(mup ./ mun)
    return mup ./ (mup + mun)
end
function test_val()
    rng = Xoshiro(0)

    N = 50
    for i in 1:100
        r, c, W, optimum = generate_random_ot(N, rng)
        # r, c, W = map(CuArray, [r, c, W])
        for i in ProgressBar(1:5000)
            normalize!(W, Inf)
            μ = project(rand(rng, N))
            ν = project(rand(rng, N))
            s = rand() * 0.01
            cμ = colvec(μ, W, s, r)
            # cν = colvec(ν, W, s, r)
            eigvals2 = eigvals(diagm(cμ - c ./ μ) - cμ * cμ')
            v = argmax(cμ - c ./ μ)
            println(cμ[v], " ", c[v] / μ[v], " ", cμ[v] / (c[v] / μ[v]))
            # println(μ, ν)
            # μ = zeros(N)
            ind = Int(Base.round(rand(rng) * N, RoundUp))
            # μ[ind] = 1
            # μ = normalize(μ .+ 1e-7, 1)
            # println(μ)
            @test dot(c, μ) + dot(c, 1 .- μ) ≈ 1.0 atol = 1e-10
            @test linearization(μ, c .* ν, W, s, r) <= 1 / η * weighted_kl(c .* μ, c .* ν, W, s, r, c)
            # @test weighted_kl(μ, ν, W, s, r, c) <= 4(exp(1))
        end
    end
end
rng = Xoshiro(0)
N = 10
r, c, W, optimum = generate_random_ot(N, rng)
normalize!(W, Inf)
mu = rand(rng, N) ./ 10 .+ 0.4
nu = rand(rng, N) ./ 10 .+ 0.4
args = EOTArgs(B=1.0, eta_p=1e-6, eta_mu=0.0, itermax=2000, epsilon=1e-10)
