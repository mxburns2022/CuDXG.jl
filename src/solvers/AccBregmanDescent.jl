
function accelerated_bregman_descent(
    r::AbstractArray{R},
    c::AbstractArray{R},
    _W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50
) where R
    W∞ = norm(_W, Inf)
    ηp = args.eta_p / 2 / W∞
    W = _W ./ W∞
    n = size(r, 1)

    μ⁺ = ones(R, n) * 0.5

    sumvals = zeros(R, n)
    residual_storage = zeros(R, n)
    maxvals = zeros(R, n)

    μ⁻ = copy(μ⁺)
    ν⁺ = copy(μ⁺)
    ν⁻ = copy(μ⁺)
    γ⁺ = copy(μ⁺)
    γ⁻ = copy(μ⁺)
    ημ = 1.0 ./ (c .+ args.C3 / n)
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(μ⁺, μ⁻)
        maximum!(maxvals, -(W * 0.5 .+ μ⁺' .- (1. .- μ⁺)') ./ ηp)
        sum!(sumvals, exp.(-(W * 0.5 .+ μ⁺' .- (1. .- μ⁺)') ./ ηp .- maxvals))
        sum!(residual_storage', exp.(-(W * 0.5 .+ μ⁺' .- (1. .- μ⁺)') ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    function dual_value(μ⁺)
        return dot(c, μ⁺ - (1. .- μ⁺)) + ηp * dot(r, logsumexp(-(W * 0.5 .+ μ⁺' .- (1. .- μ⁺)') ./ ηp, 2))
    end
    function bregman(μ⁺, μ⁻, ν⁺, ν⁻)
        return dot(μ⁺ ./ ημ, log.(μ⁺ ./ ν⁺)) + dot((1. .- μ⁺) ./ ημ, log.((1. .- μ⁺) ./ (1. .- ν⁺)))
    end
    function linearization(μ⁺, μ⁻, ν⁺, ν⁻, ∇dν)
        dual_value(μ⁺) - dual_value(ν⁺) - dot(∇dν, μ⁺ - ν⁺ - (μ⁻ - ν⁻))
    end
    ρ = 2
    γ = 1.5
    Gmin = exp(-args.B) / 10
    G = 1
    function ΔBProject!(μ⁺, μ⁻)
        normv = (μ⁺ + μ⁻)

        μ⁻ .= μ⁻ ./ normv
        μ⁺ .= μ⁺ ./ normv
        μ⁻adjust = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
        μ⁺adjust = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
        normv = (μ⁺adjust + μ⁻adjust)

        μ⁻ .= μ⁻adjust ./ normv
        μ⁺ .= μ⁺adjust ./ normv
    end
    θ = 1
    for k in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (k - 1) % frequency == 0
            p = softmax(-(W * 0.5 .+ μ⁺' .- (1 .- μ⁺)') ./ ηp, norm_dims=2)
            pr = r .* p
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = dot(round(pr, r, c), _W)
            pobj = primalv(p, _W, W∞, ηp, r, c)
            dobj = dualv(μ⁺, μ⁻, _W, W, W∞, ηp, r, c)
            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,dual_extrap\n" elapsed_time i feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        Mₖ = max(G / ρ, Gmin)
        Gₖ = Mₖ
        niter_inner = 0
        while true
            a = Gₖ
            b = (G * θ)
            _c = -(G * θ)
            θₖ = polyroot(a, b, _c, γ)
            # println(θₖ, θ)
            if θₖ <= 1e-20
                θ = 1e-20
                break
            end
            γ⁺ .= (1 - θₖ) * μ⁺ + θₖ * ν⁺
            γ⁻ .= (1 - θₖ) * μ⁻ + θₖ * ν⁻
            infeas(γ⁺, γ⁻)
            ν⁺_new = ν⁺ .* exp.(ημ * ηp ./ (θₖ^(γ - 1) * Gₖ) .* (residual_storage - c))
            ν⁻_new = ν⁻ .* exp.(-ημ * ηp ./ (θₖ^(γ - 1) * Gₖ) .* (residual_storage - c))
            ΔBProject!(ν⁺_new, ν⁻_new)
            μ⁺_new = (1 - θₖ) * μ⁺ + θₖ * ν⁺_new
            μ⁻_new = (1 - θₖ) * μ⁻ + θₖ * ν⁻_new
            if linearization(μ⁺_new, μ⁻_new, γ⁺, γ⁻, c - residual_storage) <= θₖ^γ * Gₖ ./ ηp * bregman(ν⁺_new, ν⁻_new, ν⁺, ν⁻)
                copy!(μ⁺, μ⁺_new)
                copy!(μ⁻, μ⁻_new)
                copy!(ν⁺, ν⁺_new)
                copy!(ν⁻, ν⁻_new)
                θ = θₖ
                G = Gₖ
                break
            end
            niter_inner += 1
            Gₖ *= ρ
        end
        if θ <= 1e-20
            break
        end
        G = Gₖ
    end
end