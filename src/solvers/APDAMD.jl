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
end

function dual_gradient!(output::TA, x::TA, prob::EOTProblem) where TA
    grad_cache1 = softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]', dims=2)
    grad_cache2 = softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]', dims=1)'
    output .= vcat(grad_cache1, grad_cache2)
    output .-= prob.b
end

function φ(x, prob::EOTProblem)
    return sum(logsumexp(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]')) - dot(prob.b, x)
end

function get_p(x, prob::EOTProblem)
    return softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]')
end

function mirror_descent_step(state::APDAMDState, prob::EOTProblem)
    state.M *= 2
    a⁺ = (1 + √(1 + 4 * prob.N * state.M * state.ā)) / (2 * state.M * prob.N)
    ā⁺ = a⁺ + state.ā
    state.μ⁺ .= a⁺ / ā⁺ .* state.z + state.ā / ā⁺ .* state.λ
    dual_gradient!(state.grad_cache, state.μ⁺, prob)
    state.z⁺ .= state.z - prob.N * a⁺ * state.grad_cache
    state.λ⁺ .= a⁺ / ā⁺ .* state.z⁺ + state.ā / ā⁺ .* state.λ
    if φ(state.λ⁺, prob) - φ(state.μ⁺, prob) - (state.λ⁺ - state.μ⁺)' * state.grad_cache <= state.M / 2 * norm(state.λ⁺ - state.μ⁺, Inf)^2
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
    prob = EOTProblem(η=args.ηp, r=r, c=c, W=W)
    p = softmax(-W ./ prob.η)
    λ0 = TA(zeros(2prob.N))
    state = APDAMDState(p=p, λ=λ0, L=4. / prob.η)
    for i in 1:args.tmax
        bisection_search(state, prob)
        if args.verbose && (i - 1) % frequency == 0
            obj = dot(round(state.p, r, c), W)
            @printf "%d,%.14e,%.14e,-1,dual_extrap\n" i state.infeas obj
        end
        if state.infeas <= args.ε
            break
        end
    end
    return state.p, state.μ[1:prob.N], state.μ[prob.N+1:end]
end