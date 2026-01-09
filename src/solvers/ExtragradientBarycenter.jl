using LinearAlgebra
using Random
using CUDA
using Test
# using PythonOT
using Printf




function warp_logsumexp_t_fused_otb!(output::CuDeviceVector{T}, Wt::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(Wt, 1)  # Wt is transposed: first dim iterates columns j
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    # Precompute scalars to avoid divisions in the inner loop
    α = (0.5 * st) / (reg)
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
            δ = (2μ⁺[jj] - 1) * invreg
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
            output[tid_x] = log(s) + m
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


function update_μ_residual_otb(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺a::CuDeviceArray{R}, μ⁻a::CuDeviceArray{R}, residual::CuDeviceArray{R}, c::CuDeviceArray{R}, eta_muᵢ::CuDeviceArray{R}, eta_mu::R, B::R, adjust::Bool, W∞::R) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    eta_muᵢ_tid = eta_muᵢ[tid]
    difference = residual[tid] - c[tid]
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


function residual_c_otb!(output::CuDeviceVector{T}, c::CuDeviceArray{T}, r::CuDeviceArray{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T},
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
        diff = W∞ * (2μ⁺[tid_x] - 1.0)

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
            output[tid_x] = local_acc2# - c[tid_x]
        end
        tid_x += nwarps
    end
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
    eta_mu = [1 ./ (args.C2 ./ (c[k] .+ args.C3 / n)) for k in 1:m]

    threads = 256
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    Wt = [permutedims(Wₖ, (2, 1)) for Wₖ in W]
    time_start = time_ns()
    @inline function infeas(ck, μ⁺, Wₖᵀ, Wₖ, W∞, r)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals, Wₖᵀ, μ⁺, ηp, st, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage, r, Wₖ, μ⁺, sumvals, ηp, st, W∞)
    end
    # pr
    # ηp = 0.1
    for i in 1:args.itermax
        for k in 1:m
            infeas(c[k], ν⁺[k], Wt[k], W[k], W∞[k] * W∞_multiplier, r)
            @cuda threads = threads blocks = linear_blocks update_μ_residual(μt⁺[k], μt⁻[k], μ⁺[k], μ⁻[k], residual_storage, c[k], eta_mu[k], args.eta_mu, args.B, false)
            @cuda threads = threads blocks = linear_blocks update_μ(νt⁺[k], νt⁻[k], ν⁺[k], ν⁻[k], μ⁺[k], μ⁻[k], ηp)

        end
        CUDA.synchronize()
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            p = [r .* softmax(-(W[k] * 0.5 * st ./ W∞[k] .+ (νt⁺[k]' .- νt⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
            obj = dot(w, dot(p[k], W[k]) for k in 1:m)
            feas = sum(norm(sum(p[k], dims=1)' - c[k], 1) for k in 1:m)
            @printf "%.6e,%d,%.14e,%.14e,extragradient_barycenter_kernel\n" elapsed_time i feas obj
            break
        end
        if i % frequency == 0
            p = [r .* softmax(-(W[k] * 0.5 * st ./ W∞[k] .+ (νt⁺[k]' .- νt⁻[k]')) ./ ηp, norm_dims=2) for k in 1:m]
            obj = dot(w, dot(p[k], W[k]) for k in 1:m)
            feas = sum(norm(sum(p[k], dims=1)' - c[k], 1) for k in 1:m)
            @printf "%.6e,%d,%.14e,%.14e,extragradient_barycenter_kernel\n" elapsed_time i feas obj
            # sleep(0.1)
            if feas < 1e-13
                @printf "%.6e,%d,%.14e,%.14e,extragradient_barycenter_kernel\n" elapsed_time i feas obj
                break
            end
        end
        # st = (1 - ηp^(2 / 3)) * st + ηp^(2 / 3)
        st = (1 - ηp) * st + ηp
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals, Wt[k], νt⁺[k], ηp, st, W∞[k])
            CUDA.synchronize()
            r̄ += sumvals * w[k]
        end
        rmax = maximum(r̄)
        r̄ = normalize(exp.(r̄ .- rmax), 1)
        for k in 1:m
            infeas(c[k], νt⁺[k], Wt[k], W[k], W∞[k], r̄)
            # display(residual_storage)
            # display(c[k])
            @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺[k], μ⁻[k], μ⁺[k], μ⁻[k], residual_storage, c[k], eta_mu[k], args.eta_mu, args.B, true)
            # @cuda threads = threads blocks = linear_blocks update_μ(ν⁺[k], ν⁻[k], ν⁺[k], ν⁻[k], μt⁺[k], μt⁻[k], ηp)
            @cuda threads = threads blocks = linear_blocks update_μ_other(ν⁺[k], ν⁻[k], ν⁺[k], ν⁻[k], μt⁺[k], μt⁻[k], νt⁺[k], νt⁻[k], ηp)
        end
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals, Wt[k], ν⁺[k], ηp, st, W∞[k])
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

function ipb_kernel(
    c::AbstractArray{TA},
    loc1::TW,
    loc2::TW,
    W∞::R,
    args::EOTArgs{R},
    _w::Tw=Nothing,
    frequency::Int=50,
    p::Int=2
) where {R,TW<:CuArray,TA<:CuArray,Tw}
    m = size(c, 1)
    η = args.eta_p
    n = size(c[1], 1)
    if _w == Nothing
        _w = ones(R, n)
    end
    w = normalize(_w, 1)
    φ = [CUDA.zeros(R, n) for i in 1:m]
    ψ = [CUDA.zeros(R, n) for i in 1:m]
    rcache = CUDA.zeros(R, n)
    residual_cache = CUDA.zeros(R, n)
    cost_cache = CUDA.zeros(R, n)
    r = CUDA.ones(R, n) / n

    threads = 256
    blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    println("time(s),iter,infeas,ot_objective,dual,sinkhorn")
    time_start = time_ns()
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        fill!(rcache, 0.0)
        for k in 1:m
            # @cuda threads = threads blocks = blocks warp_logsumexp_ct_opt!(φ[k], loc1, loc2, r, ψ[k], η, W∞)
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_opt!(ψ[k], loc1, loc2, c[k], φ[k], η, W∞, p)
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_opt!(φ[k], loc1, loc2, r, ψ[k], η, W∞, p)
            φ[k] = (φ[k] - log.(r))
            rcache -= w[k] .* φ[k]
        end
        for k in 1:m
            φ[k] = rcache + φ[k]
        end

        rcache .-= maximum(rcache)
        r .= exp.(rcache)
        r ./= sum(r)
        # println(r)

        CUDA.synchronize()
        if args.verbose && (i - 1) % frequency == 0
            ot_objective = 0.0
            residual_r = 0.0
            objective = 0.0
            for k in 1:m
                @cuda threads = threads blocks = blocks residual_opt!(residual_cache, cost_cache, loc1, loc2, r, φ[k], ψ[k], η, W∞, p)
                # CUDA.synchronize()
                residual_r += w[k] * norm(residual_cache, 1)
                ot_objective += w[k] * sum(cost_cache)
                @cuda threads = threads blocks = blocks residual_opt!(residual_cache, cost_cache, loc1, loc2, c[k], ψ[k], φ[k], η, W∞, p)
                residual_r += w[k] * norm(residual_cache, 1)
                objective += w[k] * dot(ψ[k], c[k]) + w[k] * dot(φ[k], r)
            end
            @printf "%.6e,%d,%.14e,%.14e,%.14e,sinkhorn_kernel\n" elapsed_time i residual_r ot_objective objective
            if residual_r <= args.epsilon / 2
                break
            end
        end
    end
    return r, φ, ψ
end
# Compute barycenters defined on a regular grid
function extragradient_barycenter_kernel(
    c::AbstractArray{TA},
    loc1::TW,
    loc2::TW,
    W∞::R,
    args::EOTArgs{R},
    _w::Tw=Nothing,
    frequency::Int=50, p::Int=2;
    s0::Float64=0.0
) where {R,TW<:CuArray,TA<:CuArray,Tw}
    st = s0
    m = size(c, 1)
    ηp = args.eta_p / 2
    n = size(c[1], 1)
    if _w == Nothing
        _w = ones(R, n)
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
    cost_storage = CUDA.zeros(R, n)
    residual_storage = CUDA.zeros(R, n)
    eta_mu = [1 ./ (args.C2 ./ (c[k] .+ args.C3 / n)) for k in 1:m]

    threads = 256
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    time_start = time_ns()
    @inline function infeas(μ⁺, r)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, loc1, loc2, μ⁺, ηp, st, W∞, p)
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_storage, cost_storage, loc1, loc2, r, μ⁺, sumvals, ηp, st, W∞, p)
    end
    # pr
    # ηp = 0.1
    println("time(s),iter,infeas,ot_objective,primal,dual")
    for i in 1:args.itermax
        cost_value = 0
        primal_value = 0
        feas_value = 0
        # hr = dot(r, log.(r .+ 1e-20))
        for k in 1:m
            infeas(ν⁺[k], r)
            CUDA.synchronize()
            feas_value += norm(c[k] - residual_storage, 1) * w[k]
            cost_value += sum(cost_storage) * w[k]

            @cuda threads = threads blocks = linear_blocks update_μ_residual(
                μt⁺[k], μt⁻[k], μ⁺[k], μ⁻[k],
                residual_storage, c[k], eta_mu[k],
                args.eta_mu, args.B, false)
            @cuda threads = threads blocks = linear_blocks update_μ(
                νt⁺[k], νt⁻[k], ν⁺[k], ν⁻[k], μ⁺[k], μ⁻[k], ηp)
            primal_value += (dot(r, sumvals) * ηp + dot(2ν⁺[k] .- 1, c[k])) * w[k]

        end
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if i % frequency == 0
            dual_value = 0.0
            for k in 1:m
                @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, loc1, loc2, μ⁺[k], ηp, 1.0, W∞, p)

                dual_value = ηp * dot(r, sumvals) + dot(c[k], 2μ⁺[k] .- 1)
            end
            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragradient_barycenter_kernel\n" elapsed_time i feas_value cost_value primal_value dual_value
            flush(stdout)
            if primal_value - dual_value < args.epsilon / 6 && feas_value < args.epsilon / 6
                break
            end
        end
        # st = (1 - ηp^(2 / 3)) * st + ηp^(2 / 3)
        st = (1 - ηp) * st + ηp
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, loc1, loc2, νt⁺[k], ηp, st, W∞, p)
            CUDA.synchronize()
            r̄ += sumvals * w[k]
        end
        rmax = maximum(r̄)
        r̄ = exp.(r̄ .- rmax)
        r̄ /= sum(r̄)
        for k in 1:m
            infeas(νt⁺[k], r̄)
            @cuda threads = threads blocks = linear_blocks update_μ_residual(
                μ⁺[k], μ⁻[k], μ⁺[k], μ⁻[k],
                residual_storage, c[k], eta_mu[k],
                args.eta_mu, args.B, true)
            @cuda threads = threads blocks = linear_blocks update_μ(
                ν⁺[k], ν⁻[k], ν⁺[k], ν⁻[k], μt⁺[k], μt⁻[k], ηp)
        end
        fill!(r̄, 0.0)
        for k in 1:m
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, loc1, loc2, ν⁺[k], ηp, st, W∞, p)
            CUDA.synchronize()
            r̄ += sumvals * w[k]
        end
        rmax = maximum(r̄)
        r = exp.(r̄ .- rmax)
        r /= sum(r)
        # ηp *= 0.99
    end

    return r, μ⁺, μ⁻, st
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
        @cuda threads = threads blocks = warp_blocks warp_logsumexp!(sumvals, Wt[k], ν⁺[k], ηp, st, W∞[k])
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

function extragradient_ot_barycenter(
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
