include("src/Utilities.jl")
using LinearAlgebra
using Random
using CUDA
using Test
using PythonOT
using Printf


function update_μ(μ⁺::AbstractArray{R}, μ⁻::AbstractArray{R}, μ⁺_1::AbstractArray{R}, μ⁻_1::AbstractArray{R}, μ⁺_2::AbstractArray{R}, μ⁻_2::AbstractArray{R}, η) where R
    μ⁺ .= μ⁺_1 + η * (μ⁺_2 - μ⁺_1)
    μ⁻ .= μ⁻_1 + η * (μ⁻_2 - μ⁻_1)
    normv = (μ⁺ + μ⁻)
    μ⁺ ./= normv
    μ⁻ ./= normv
end

function update_μ(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺_1::CuDeviceArray{R}, μ⁻_1::CuDeviceArray{R}, μ⁺_2::CuDeviceArray{R}, μ⁻_2::CuDeviceArray{R}, η) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    new_μplus = μ⁺_1[tid] + η * (μ⁺_2[tid] - μ⁺_1[tid])
    new_μminus = μ⁻_1[tid] + η * (μ⁻_2[tid] - μ⁻_1[tid])
    normv = new_μplus + new_μminus
    new_μplus /= normv
    new_μminus /= normv
    μ⁺[tid] = new_μplus
    μ⁻[tid] = new_μminus

    return
end


function update_μ_residual(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺a::CuDeviceArray{R}, μ⁻a::CuDeviceArray{R}, residual::CuDeviceArray{R}, eta_muᵢ::CuDeviceArray{R}, eta_mu::R, B::R, adjust::Bool, W∞::R) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    eta_muᵢ_tid = eta_muᵢ[tid]
    difference = residual[tid]
    maxval = eta_muᵢ_tid * max(-difference, difference)
    new_μ⁺ = μ⁺a[tid]^(1 - eta_mu) * exp(W∞ * eta_muᵢ_tid * (difference) - maxval)
    new_μ⁻ = μ⁻a[tid]^(1 - eta_mu) * exp(-W∞ * eta_muᵢ_tid * (difference) - maxval)
    normv = new_μ⁺ + new_μ⁻
    new_μ⁺ /= normv
    new_μ⁻ /= normv

    μ⁺[tid] = new_μ⁺
    μ⁻[tid] = new_μ⁻
    if adjust
        new_μ⁻a = max(new_μ⁻, exp(-B) * max(new_μ⁺, new_μ⁻))
        new_μ⁺a = max(new_μ⁺, exp(-B) * max(new_μ⁺, new_μ⁻))
        normv = (new_μ⁻a + new_μ⁺a)
        new_μ⁺a /= normv
        new_μ⁻a /= normv
        μ⁺a[tid] = new_μ⁺a
        μ⁻a[tid] = new_μ⁻a
        μ⁺[tid] = new_μ⁺a
        μ⁻[tid] = new_μ⁻a
    end

    return
end


function warp_logsumexp!(output::CuDeviceVector{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(W, 1)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        maxval = -Inf
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * W[i+local_id, tid_x] * st + W∞ * (μ⁺[i+local_id] - μ⁻[i+local_id])) / reg
            maxval = max(value, maxval)
        end
        maxval = CUDA.reduce_warp(max, maxval)
        if local_id == 0
            output[tid_x] = maxval
        end
        # return
        sync_warp()
        maxval = output[tid_x]

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * W[i+local_id, tid_x] * st + W∞ * (μ⁺[i+local_id] - μ⁻[i+local_id])) / reg
            local_acc += exp(value - maxval)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = (log(local_acc2) + maxval)
        end
        tid_x += nwarps
    end
    return
end


# Fused, coalesced variant: expects Wᵗ (i.e., permutedims(W, (2,1))) so that loads across lanes are coalesced.
# Performs a single pass using an associative log-sum-exp pair reduction to avoid re-reading memory.
@inline function _lse_pair_combine(a::NTuple{2,T}, b::NTuple{2,T}) where {T}
    m1, s1 = a
    m2, s2 = b
    m = ifelse(m1 > m2, m1, m2)
    # s = s1*exp(m1-m) + s2*exp(m2-m)
    return (m, s1 * exp(m1 - m) + s2 * exp(m2 - m))
end

function warp_logsumexp_t_fused!(output::CuDeviceVector{T}, Wt::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, reg::T, st::T, W∞::T,) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(Wt, 1)  # Wt is transposed: first dim iterates columns j
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    # Precompute scalars to avoid divisions in the inner loop
    α = (0.5 * st) / reg
    invreg = W∞ / reg
    @inbounds for _ in 1:N_outer
        if tid_x > N
            return
        end
        # Online log-sum-exp per thread
        m_local = T(-Inf)
        s_local = zero(T)
        @inbounds for j in 1:step:N
            jj = j + local_id
            if jj > N
                break
            end
            # Access W as W[jj, tid_x] because we pass Wᵗ for coalesced loads
            w = Wt[jj, tid_x]
            δ = (μ⁺[jj] - μ⁻[jj]) * invreg
            v = -(α * w + δ)
            # online LSE update
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end

        # Warp-level reduction of (m,s)
        m = shfl_down_sync(0xffffffff, m_local, 16)
        s = shfl_down_sync(0xffffffff, s_local, 16)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 8)
        s = shfl_down_sync(0xffffffff, s_local, 8)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 4)
        s = shfl_down_sync(0xffffffff, s_local, 4)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 2)
        s = shfl_down_sync(0xffffffff, s_local, 2)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 1)
        s = shfl_down_sync(0xffffffff, s_local, 1)
        m, s = _lse_pair_combine((m_local, s_local), (m, s))

        if local_id == 0
            output[tid_x] = s
        end
        tid_x += nwarps
    end
    return
end


function residual_c!(output::CuDeviceVector{T}, c::CuDeviceArray{T}, r::CuDeviceArray{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T},
    logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(c, 1)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        diff = W∞ * (μ⁺[tid_x] - μ⁻[tid_x])

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * st * W[i+local_id, tid_x] + diff) / reg - logZi[i+local_id]
            local_acc += r[i+local_id] * exp(value)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        sync_warp()
        if local_id == 0
            output[tid_x] = local_acc2 - c[tid_x]
        end
        tid_x += nwarps
    end
    return
end


function update_r!(output::CuDeviceVector{T}, logZi::CuDeviceVector{T}, wₖ::T) where T
    N = size(output, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end

    output[tid] += logZi[tid] * wₖ
    return
end


function extragradient_barycenter_dual(
    c::AbstractArray{TA},
    W::AbstractArray{TW},
    args::EOTArgs{R},
    _w::TA=Nothing,
    frequency::Int=50;
    s0::Float64=0.0
) where {R,TW<:CuArray,TA<:CuArray}
    st = s0
    m = size(c, 1)
    W∞ = CuArray([norm(W[i], Inf) for i in 1:m])
    ηp = args.eta_p / 2
    n = size(W[1], 1)
    if _w == Nothing
        _w = CuArray(ones(R, n))
    end
    w = normalize(_w, 1)
    μ⁺ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    μ⁻ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    μt⁺ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    μt⁻ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    ν⁺ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    ν⁻ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    νt⁺ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    νt⁻ = [CUDA.ones(R, n) * 0.5 for i in 1:m]
    r = CUDA.ones(R, n) / n#reshape(normalize(exp.(sum(hcat([w[k] * logsumexp(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, 2) for k in 1:m]...), dims=2)), 1), n)
    r̄ = CUDA.zeros(R, n)
    sumvals = CUDA.zeros(R, n)
    maxvals = CUDA.zeros(R, n)
    residual_storage = CUDA.zeros(R, n)
    eta_mu = [args.C2 ./ (c[k] .+ args.C3 / n) for k in 1:m]

    threads = 256
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    Wt = [permutedims(Wₖ, (2, 1)) for Wₖ in W]
    time_start = time_ns()
    @inline function infeas(cₖ, μ⁺, μ⁻, Wₖᵀ, Wₖ, W∞, r)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp!(sumvals, Wₖᵀ, μ⁺, μ⁻, ηp, st, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage, cₖ, r, Wₖ, μ⁺, μ⁻, sumvals, ηp, st, W∞)
    end
    # pr
    # ηp = 0.1
    prevobj = 1e-4
    for i in 1:args.itermax
        for k in 1:m
            infeas(c[k], ν⁺[k], ν⁻[k], Wt[k], W[k], W∞[k], r)
            @cuda threads = threads blocks = linear_blocks update_μ_residual(μt⁺[k], μt⁻[k], μ⁺[k], μ⁻[k], residual_storage, eta_mu[k], args.eta_mu, args.B, false, W∞[k])
            @cuda threads = threads blocks = linear_blocks update_μ(νt⁺[k], νt⁻[k], ν⁺[k], ν⁻[k], μ⁺[k], μ⁻[k], ηp)
        end
        CUDA.synchronize()
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if i % frequency == 0
            p = [r .* softmax(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
            obj = dot(w, dot(p[k], W[k]) for k in 1:m)
            feas = sum(norm(sum(p[k], dims=1)' - c[k], 1) for k in 1:m)
            println("$(obj) $(feas)")
            # sleep(0.1)
            if feas < 1e-12
                break
            end
        end
        # st = (1 - ηp^(2 / 3)) * st + ηp^(2 / 3)
        st = (1 - ηp) * st + ηp
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp!(sumvals, Wt[k], νt⁺[k], νt⁻[k], ηp, st, W∞[k])
            CUDA.synchronize()
            r̄ += sumvals * w[k]
        end
        rmax = maximum(r̄)
        r̄ = normalize(exp.(r̄ .- rmax), 1)
        for k in 1:m
            infeas(c[k], νt⁺[k], νt⁻[k], Wt[k], W[k], W∞[k], r̄)
            @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺[k], μ⁻[k], μ⁺[k], μ⁻[k], residual_storage, eta_mu[k], args.eta_mu, args.B, true, W∞[k])
            @cuda threads = threads blocks = linear_blocks update_μ(ν⁺[k], ν⁻[k], ν⁺[k], ν⁻[k], μt⁺[k], μt⁻[k], ηp)
        end
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp!(sumvals, Wt[k], ν⁺[k], ν⁻[k], ηp, st, W∞[k])
            CUDA.synchronize()
            r̄ += sumvals * w[k]
        end
        rmax = maximum(r̄)
        r = normalize(exp.(r̄ .- rmax), 1)
        # ηp *= 0.99
    end
    p = [r .* softmax(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
    r = reshape(r, n)
    return p, r, μ⁺, μ⁻, st
end

function test_update_r()
    n = 5
    m = 4
    rng = Xoshiro(0)
    ηp = 1e-2
    st = 0.124
    c = [CuArray(normalize(rand(rng, n), 1)) for i in 1:m]
    W = CuArray(rand(rng, n, n))
    Wk = [W, W, W, W]
    Wt = [permutedims(Wₖ, (2, 1)) for Wₖ in Wk]
    W∞ = [norm(W, Inf) for i in 1:m]
    threads = 256

    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    r = CUDA.ones(Float64, n) / n
    r̄ = CUDA.zeros(Float64, n)
    ν⁺ = [CuArray(rand(rng, n)) for i in 1:m]
    ν⁻ = [1 .- ν⁺[i] for i in 1:m]
    sumvals = CUDA.zeros(Float64, n)
    w = normalize(rand(rng, m), 1)
    fill!(r̄, 0.0)
    for k in 1:m
        @cuda threads = threads blocks = warp_blocks warp_logsumexp!(sumvals, Wt[k], ν⁺[k], ν⁻[k], ηp, st, W∞[k])
        CUDA.synchronize()
        # println(sumvals, -(Wt[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ (ηp))
        r̄ += w[k] * sumvals
    end
    rmax = maximum(r̄)
    r = normalize(exp.(r̄ .- rmax), 1)

    println(r̄)
    println(r)
    println(r2)
end

function extragradient_ot_dual(
    c::AbstractVector{TA},
    W::AbstractVector{TM},
    args::EOTArgs{R},
    w::TW=Nothing,
    frequency::Int=50;
    s0::Float64=0.0
) where {TA,TM,TW,R}
    st = s0
    m = size(c, 1)
    W∞ = TA([norm(W[i], Inf) for i in 1:m])
    ηp = args.eta_p / 2
    n = size(W[1], 1)
    if w == Nothing
        w = TA(ones(R, n))
    end
    μ⁺ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    μ⁻ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    μt⁺ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    μt⁻ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    ν⁺ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    ν⁻ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    νt⁺ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    νt⁻ = [TA(ones(R, n)) * 0.5 for i in 1:m]
    Σw⁻¹ = 1 / sum(w)
    r = normalize(exp.(Σw⁻¹ * sum(hcat([w[k] * logsumexp(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, 2) for k in 1:m]...), dims=2)), 1)

    sumvals = TA(zeros(R, n))
    residual_storage = TA(zeros(R, n))
    maxvals = TA(zeros(R, n))

    eta_mu = [args.C2 ./ ((c[k] .+ args.C3 / n)) for k in 1:m]

    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(μ⁺, μ⁻, Wₖ, W∞, r)

        maximum!(maxvals, -(Wₖ * 0.5 * st .+ W∞ * (μ⁺' .- μ⁻')) ./ ηp)

        sum!(sumvals, exp.(-(Wₖ * 0.5 * st .+ W∞ * (μ⁺' .- μ⁻')) ./ ηp .- maxvals))
        sum!(residual_storage', exp.(-(Wₖ * 0.5 * st .+ W∞ * (μ⁺' .- μ⁻')) ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        for k in 1:m
            infeas(ν⁺[k], ν⁻[k], W[k], W∞[k], r)
            μt⁺[k] = μ⁺[k] .^ (1 - args.eta_mu) .* exp.(eta_mu[k] .* W∞[k] .* (residual_storage - c[k]))
            μt⁻[k] = μ⁻[k] .^ (1 - args.eta_mu) .* exp.(-eta_mu[k] .* W∞[k] .* (residual_storage - c[k]))
            normv = (μt⁺[k] + μt⁻[k])
            μt⁺[k] ./= normv
            μt⁻[k] ./= normv
            νt⁺[k] = ν⁺[k] + ηp * (μ⁺[k] - ν⁺[k])
            νt⁻[k] = ν⁻[k] + ηp * (μ⁻[k] - ν⁻[k])
            normv = (νt⁺[k] + νt⁻[k])
            νt⁺[k] ./= normv
            νt⁻[k] ./= normv
        end

        st = (1 - ηp) * st + ηp
        num = Σw⁻¹ * sum(hcat([w[k] * logsumexp(-(W[k] * 0.5 * st .+ W∞[k] * (νt⁺[k]' .- νt⁻[k]')) ./ (ηp), 2) for k in 1:m]...), dims=2)
        num .-= maximum(num)
        r̄ = normalize(exp.(num), 1)
        for k in 1:m
            infeas(νt⁺[k], νt⁻[k], W[k], W∞[k], r̄)
            μ⁺[k] = μ⁺[k] .^ (1 - args.eta_mu) .* exp.(eta_mu[k] .* W∞[k] .* (residual_storage - c[k]))
            μ⁻[k] = μ⁻[k] .^ (1 - args.eta_mu) .* exp.(-eta_mu[k] .* W∞[k] .* (residual_storage - c[k]))
            normv = (μ⁺[k] + μ⁻[k])
            μ⁺[k] ./= normv
            μ⁻[k] ./= normv

            ν⁺[k] = ν⁺[k] + ηp * (μt⁺[k] - ν⁺[k])
            ν⁻[k] = ν⁻[k] + ηp * (μt⁻[k] - ν⁻[k])
            normv = (ν⁺[k] + ν⁻[k])
            ν⁺[k] ./= normv
            ν⁻[k] ./= normv

            μt⁻[k] = max.(μ⁻[k], exp(-args.B) .* max.(μ⁺[k], μ⁻[k]))
            μt⁺[k] = max.(μ⁺[k], exp(-args.B) .* max.(μ⁺[k], μ⁻[k]))
            normv = (μt⁻[k] + μt⁺[k])
            μ⁺[k] .= μt⁺[k] ./ normv
            μ⁻[k] .= μt⁻[k] ./ normv

        end
        num = Σw⁻¹ * sum(hcat([w[k] * logsumexp(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ (ηp), 2) for k in 1:m]...), dims=2)
        num .-= maximum(num)
        r = normalize(exp.(num), 1)
        if i % frequency == 0
            p = [r .* softmax(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
            obj = dot(w, dot(p[k], W[k]) for k in 1:m)
            feas = sum(norm(sum(p[k], dims=1)' - c[k], 1) for k in 1:m)
            println("$(obj) $(feas)")
            # sleep(0.1)
            if feas < 1e-10
                break
            end
        end

    end
    p = [r .* softmax(-(W[k] * 0.5 * st .+ W∞[k] * (ν⁺[k]' .- ν⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
    r = reshape(r, n)
    return p, r, μ⁺, μ⁻, st
end


m = 6
n = 400
gen = Xoshiro(0)
C = [CuArray(normalize(rand(gen, n), 1)) for _ in 1:m]
W = CuArray(rand(gen, n, n))
open("matrix.txt", "w") do outfile
    for i in axes(W, 1)
        for j in axes(W, 2)
            print(outfile, "$(W[i, j]) ")
        end
        println(outfile)
    end
end
open("dists.txt", "w") do outfile
    for i in axes(C, 1)
        for j in axes(C[i], 1)
            print(outfile, "$(C[i][j]) ")
        end
        println(outfile)
    end
end
# w = CuArray(normalize(ones(m), 1))#rand(gen, m), 1))
# rt = barycenter(hcat(C...), W, 1e-2, stopThr=1e-8, method="sinkhorn", verbose=true, numItermax=1000000)
WK = [(copy(W)) for _ in 1:m]
args = EOTArgs(eta_p=1e-8, itermax=5000)
# args = EOTArgs(eta_p=1e-8, itermax=10000)
c1, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1004.csv")
c2, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1007.csv")
c3, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1010.csv")
c5, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1008.csv")
c6, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1009.csv")
c4, h, w = read_dotmark_data("/home/matt/Documents/bench/DOTmark_1.0/Data/Shapes/data64_1003.csv")
W = CuArray(normalize(get_euclidean_distance(h, w), Inf))
C = [CuArray(normalize(c1 .+ 1e-8, 1)), CuArray(normalize(c2 .+ 1e-8, 1)), CuArray(normalize(c5 .+ 1e-8, 1))]
m = size(C, 1)
WK = [W for i in 1:m]
w = CUDA.ones(Float64, m)
# w = CuArray([0.333333333333333333, 0.333333333333333333, 0.333333333333333333, 0.333333333333333333, 0.333333333333333333])
# rt = barycenter(hcat(C...), W, 1e-2, stopThr=1e-8, method="sinkhorn", verbose=true, numItermax=1000000)
# emdcost = sum(emd2(rt, C[i], W) for i in 1:m)
# p, r, _, _, _ = extragradient_ot_dual(C, WK, args, w, 50)
# open("r.txt", "w") do outfile
#     for i in axes(r, 1)
#         print(outfile, "$(r[i]) ")
#     end
# end
args = EOTArgs(eta_p=1e-7, itermax=10000)
p, r, _, _ = extragradient_barycenter_dual(C, WK, args, w, 100)
open("r64_sparse.txt", "w") do outfile
    for i in axes(r, 1)
        print(outfile, "$(r[i]) ")
    end
end
# emdcost = sum(emd2(rt, C[i], W) for i in 1:m)
# p, r2, _, _, _ = extragradient_ot_dual(C, WK, args, w, 500)
# emdcost = dot(Array(w), emd2(Array(rt), Array(C[i]), Array(W)) for i in 1:m)
# emdcost2 = dot(Array(w), emd2(Array(r), Array(C[i]), Array(W)) for i in 1:m)
# # using JuMP
# using Gurobi
# if n < 200
# model = Model(Gurobi.Optimizer)
# @variable(model, pm[1:m, 1:n, 1:n])
# @variable(model, rmv[1:n])
# for i in 1:m
#     @constraint(model, sum(pm[i, :, :]', dims=2) == Array(C[i]))
#     @constraint(model, sum(pm[i, :, :], dims=2) == rmv)
# end
# @constraint(model, pm .>= 0)
# @constraint(model, rmv .>= 0)
# @constraint(model, sum(rmv) == 1.)
# @objective(model, MIN_SENSE, dot(Array(w), [dot(pm[k, :, :], Array(WK[k])) for k in 1:m]))
# # end
