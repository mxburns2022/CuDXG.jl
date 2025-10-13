using CUDA
using Printf
using LinearAlgebra




function HPD(r::TA,
    c::TA,
    _W::TM,
    args::EOTArgs{R},
    frequency::Int=50) where {TA,TM,R}
    N = size(r, 1)
    L = sqrt(2)
    W = _W ./ norm(_W, Inf)
    P = TM(ones(N, N)) ./ (N * N)
    P̃ = TM(zeros(N, N))
    u = TA(zeros(N))
    v = TA(zeros(N))
    ū = TA(zeros(N))
    v̄ = TA(zeros(N))
    ũ = TA(zeros(N))
    ṽ = TA(zeros(N))
    λ = 2norm(W, Inf)
    β = 10000log(N) / (N * λ^2)
    η = args.eta_p
    θ = η * sqrt(β) / L
    τ = 1 / sqrt(β * L)
    ρ = 0.5
    time_start = time_ns()
    P̂ = TM(zeros(N, N))
    û = τ * θ * copy(u)
    v̂ = τ * θ * copy(v)
    τθ = τ * θ
    T = 0
    # niter = 
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    for k in 1:args.itermax
        τₖ = τ * sqrt(1 + θ) / ρ
        βₖ = β / (1 + η * β * τ)
        while true
            τₖ *= ρ
            if τₖ <= 1e-20
                break
            end
            θₖ = τₖ / τ
            σₖ = βₖ * τₖ
            ū .= ũ + θₖ * (ũ - u)
            v̄ .= ṽ + θₖ * (ṽ - v)
            P̃ .= softmax(
                (1 / (1 + σₖ * η)) .* (log.(P) - σₖ * (W .- ū .- v̄')))
            u_new = reshape(clamp.(ũ + τₖ * (r - sum(P̃, dims=2)), -λ, λ), N)
            v_new = reshape(clamp.(ṽ + τₖ * (c - sum(P̃', dims=2)), -λ, λ), N)
            distance = 0.5 * norm(u_new - ū)^2 + 0.5 * norm(v_new - v̄)^2
            divergence = dot(P̃, log.(P̃ ./ P)) / βₖ
            innerprod = τₖ * sum((u_new .- ū .+ v_new' .- v̄') .* (P̃ - P))
            if distance + divergence + innerprod >= -1e-10
                copy!(u, ũ)
                copy!(v, ṽ)
                copy!(ũ, u_new)
                copy!(ṽ, v_new)
                copy!(P, P̃)
                T += τₖ
                P̂ += τₖ * P̃
                û += τₖ * ū
                v̂ += τₖ * v̄
                β = βₖ
                τ = τₖ
                θ = θₖ
                break
            end
        end
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (k - 1) % frequency == 0
            obj = dot(round(P̂ ./ T, r, c), W)
            pobj = dot(P̂ ./ T, W) + η * sum(neg_entropy(P̂ ./ T))
            avg_u = û ./ (τθ + T)
            avg_v = v̂ ./ (τθ + T)
            dobj = -η * sum(logsumexp((-W .+avg_u .+ avg_v') / η)) + dot(avg_v, c) + dot(avg_u, r)

            infeas = norm(sum(P̂ ./ T, dims=2) - r, 1) + norm(sum(P̂' ./ T, dims=2) - c, 1)
            @printf "%.6g,%d,%.14e,%.14e,%.14e,%.14e,HPD\n" elapsed_time k infeas obj pobj dobj
            if pobj-dobj < args.epsilon / 6 && infeas < args.epsilon / 6
                break
            end
        end
    end
end
