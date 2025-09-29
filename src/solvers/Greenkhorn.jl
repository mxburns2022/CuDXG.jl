using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools


function greenkhorn_log(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50) where {R}
    n = size(r, 1)
    K = exp(-W ./ args.ηp)

    φ = ones(R, n) / n
    ψ = ones(R, n) / n

    @inline function ρ(a, b)
        return b - a + a .* log.(max.(a ./ b, 1e-6))
    end
    rW = reshape(sum(ψ' .* K .* φ, dims=2) - r, n)
    cW = reshape(sum((ψ' .* K .* φ)', dims=2) - c, n)
    # println(size(rW))
    ρr = ρ(r, rW)
    ρc = ρ(c, cW)
    for k in 1:args.tmax
        i = argmax(abs.(rW))
        j = argmax(abs.(cW))
        # @test norm(cW - (sum((ψ' .* K .* φ)', dims=2) - c)) ≈ 0.0 atol = 1e-8
        # @test norm(rW - (sum((ψ' .* K .* φ), dims=2) - r)) ≈ 0.0 atol = 1e-8
        if rW[i] > cW[j]
            # println("φ update $(i)")
            φnew = r[i] / sum(K[i, :] .* ψ)
            cW += K[i, :] .* (φnew - φ[i]) .* ψ
            rW[i] = sum(K[i, :] .* (φnew) .* ψ) - r[i]
            φ[i] = φnew
            # println((diagm(φ)*K*diagm(ψ))[i, 1:end])
            # println(K[i, :] .* ψ)
            # println("φ $(rW[i]) $(i) update $(φnew) $((sum(ψ' .* K .* φ, dims=2) - r)[i]) $(norm(sum(ψ' .* K .* φ, dims=2) - r))")
        else
            ψnew = c[j] / sum(K[:, j] .* φ)
            rW += K[:, j] .* (ψnew - ψ[j]) .* φ
            cW[j] = sum(K[:, j] .* (ψnew) .* φ) - c[j]
            # ρc[j] = ρ(c[j], cW[j])
            ψ[j] = ψnew
            # println("ψ $(cW[j]) $(j) update $(ψnew) $(norm(cW))")
        end
        # println(exp.(K[:, j] .+ ψnew .+ φ) - exp.(K[:, j] .+ ψold .+ φ))
        # @test norm(rW - softmax(K .+ φ .+ ψ', dims=2; normalize_values=false)) ≈ 0.0 atol = 1e-8
        # println(exp.(K[i, :] .+ φnew .+ ψ) - exp.(K[i, :] .+ φold .+ ψ))
        # @test norm(cW - softmax(K .+ φ .+ ψ', dims=1; normalize_values=false)') ≈ 0.0 atol = 1e-8
        # if k % 2000 == 0
        # p = ψ' .* K .* φ
        # feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
        # end
        if args.verbose && (k - 1) % frequency == 0
            p = ψ' .* K .* φ
            # println(p)
            # pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            pobj = dot(p, W)
            @printf "%d,%.14e,%.14e,-1,greenkhorn\n" k feas pobj
        end

    end
    return ψ' .* K .* φ, φ, ψ
end
