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
    K = exp.(-W ./ args.ηp)

    φ = ones(R, n) / n
    ψ = ones(R, n) / n


    rW = reshape(sum(ψ' .* K .* φ, dims=2) - r, n)
    cW = reshape(sum((ψ' .* K .* φ)', dims=2) - c, n)

    for k in 1:args.tmax*n
        i = argmax(abs.(rW))
        j = argmax(abs.(cW))
        if abs(rW[i]) > abs(cW[j])
            # Rescale the ith element to ensure the row marginal element is correct
            φnew = r[i] / sum(K[i, :] .* ψ)
            cW += K[i, :] .* (φnew - φ[i]) .* ψ
            rW[i] = sum(K[i, :] .* (φnew) .* ψ) - r[i]
            φ[i] = φnew

        else
            # Rescale the jth element to ensure the column marginal element is correct
            ψnew = c[j] / sum(K[:, j] .* φ)
            rW += K[:, j] .* (ψnew - ψ[j]) .* φ
            cW[j] = sum(K[:, j] .* (ψnew) .* φ) - c[j]
            ψ[j] = ψnew
        end
        if args.verbose && (k - 1) % frequency == 0
            p = ψ' .* K .* φ
            pr = round(p, r, c)
            feas = norm(sum(pr, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            pobj = dot(p, W)
            @printf "%d,%.14e,%.14e,-1,greenkhorn\n" k feas pobj
        end

    end
    return round(ψ' .* K .* φ, r, c), φ, ψ
end
