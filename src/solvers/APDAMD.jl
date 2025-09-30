using LinearAlgebra

@kwdef mutable struct APDAMDState{TA,TM,R}
    p::TM
    p⁺::TM = copy(p)
    λ::TA
    μ::TA = copy(λ)
    z::TA = copy(λ)
    λ⁺::TA = copy(λ)
    μ⁺::TA = copy(λ)
    z⁺::TA = copy(λ)
    grad_cache::TA = copy(λ)
    L::R
    M::R = L
    a::R = 0.
    ā::R = 0.
    infeas::R = 0.
    dobj::R = 0.
end


function mirror_descent_step(state::APDAMDState, prob::EOTProblem)
    state.M *= 2
    a⁺ = (1 + √(1 + 4 * prob.N * state.M * state.ā)) / (2 * state.M * prob.N)
    ā⁺ = a⁺ + state.ā
    state.μ⁺ .= a⁺ / ā⁺ .* state.z + state.ā / ā⁺ .* state.λ
    dual_gradient!(state.grad_cache, state.μ⁺, prob)
    state.z⁺ .= state.z - prob.N * a⁺ * state.grad_cache
    state.λ⁺ .= a⁺ / ā⁺ .* state.z⁺ + state.ā / ā⁺ .* state.λ
    state.dobj = φ(state.λ⁺, prob)
    if state.dobj - φ(state.μ⁺, prob) - (state.λ⁺ - state.μ⁺)' * state.grad_cache <= state.M / 2 * norm(state.λ⁺ - state.μ⁺, Inf)^2
        state.a = a⁺
        state.ā = ā⁺
        copy!(state.z, state.z⁺)
        copy!(state.λ, state.λ⁺)
        copy!(state.μ, state.μ⁺)
        state.p⁺ = get_p(state.μ, prob)
        α = a⁺ / ā⁺
        axpby!(1 - α, state.p⁺, α, state.p)
        state.L = state.M / 2
        state.infeas = norm(sum(state.p, dims=2) - prob.r, 1) + norm(sum(state.p, dims=1)' - prob.c, 1)
        return true
    end
    return false
end


function bisection_search(state::APDAMDState, prob::EOTProblem)
    state.M = state.L / 2
    while !mirror_descent_step(state, prob)
        # Will continue until termination
    end
end


function APDAMD(r::TA,
    c::TA,
    W::TM,
    args::EOTArgs{R},
    frequency::Int=50) where {TA,TM,R}
    prob = EOTProblem(η=args.eta_p, r=r, c=c, W=W)
    p = sofitermax(-W ./ prob.η)
    λ0 = TA(zeros(2prob.N))
    state = APDAMDState(p=p, λ=λ0, L=4. / prob.η)
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    for i in 1:args.itermax
        bisection_search(state, prob)
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (i - 1) % frequency == 0
            obj = dot(round(state.p, r, c), W)
            pobj = obj + prob.η * sum(neg_entropy(p))
            dobj = prob.η * state.dobj
            @printf "%.6g,%d,%.14e,%.14e,%.14e,%.14e,APDAMD\n" elapsed_time i state.infeas obj pobj dobj
            if pobj + state.dobj < args.epsilon / 6 && state.infeas < args.epsilon / 6
                break
            end
        end
    end
    return state.p, state.μ[1:prob.N], state.μ[prob.N+1:end]
end