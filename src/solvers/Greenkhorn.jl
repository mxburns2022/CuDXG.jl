using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools


function greenkhorn_log(r::TA,
    c::TA,
    W::TM,
    args::EOTArgs{R},
    frequency::Int=50) where {TA,TM,R}
    n = size(r, 1)
    K = exp.(-W ./ args.eta_p)

    φ = TA(ones(R, n)) / n
    ψ = TA(ones(R, n)) / n


    rW = reshape(sum(ψ' .* K .* φ, dims=2) - r, n)
    cW = reshape(sum((ψ' .* K .* φ)', dims=2) - c, n)
    time_start = time_ns()
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    for k in 1:args.itermax*n
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
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (k - 1) % frequency == 0
            p = ψ' .* K .* φ
            # pr = round(p, r, c)
            feas = norm(sum(p, dims=1)' .- c, 1) + norm(sum(p, dims=2) .- r, 1)
            obj = dot(p, W)
            pobj = obj + args.eta_p * sum(neg_entropy(p, dims=[1, 2]))
            dobj = -(args.eta_p * (-sum(log.(sum(p))) - sum(c'log.(ψ)) - sum(r'log.(φ))))
            @printf "%.6g,%d,%.14e,%.14e,%.14e,%.14e,greenkhorn\n" elapsed_time k feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end

    end
    return round(ψ' .* K .* φ, r, c), φ, ψ
end
