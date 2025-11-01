using CUDA
using Printf
using LinearAlgebra
using Random
using Test

function dualv(μ⁺, μ⁻, st, W, W∞, ηp, r, c)
    pμ = softmax(-(0.5W * st .+ (W∞ .* (μ⁺ .- μ⁻))') ./ ηp, norm_dims=2)
    return (dot(W, r .* pμ) +
            2 * dot((sum(r .* pμ, dims=1)' - c), W∞ .* (μ⁺ - μ⁻))
            # 2W∞ * norm((sum(r .* pμ, dims=1)' - c), 1) 
            + 2ηp * dot(r, neg_entropy(pμ, dims=2)))
end
function primalv(μ⁺p::TA, μ⁻p::TA, st::R, W::TM, W∞::TWinf, ηp::R, r::TA, c::TA) where {TA,TM,R,TWinf}
    pμ = softmax(-(0.5W * st .+ (W∞ .* (μ⁺p .- μ⁻p))') ./ ηp, norm_dims=2)
    return (dot(W, r .* pμ)
            + 2 .* sum(W∞ .* abs.(sum(r .* pμ, dims=1)' - c))
            + 2ηp * dot(r, neg_entropy(pμ, dims=2)))
end

function primalv(p::TM, W::TM, W∞::TWinf, ηp::R, r::TA, c::TA) where {TA,TM,R,TWinf}
    return (dot(W, r .* p) +
            2 .* sum(W∞ .* abs.(sum(r .* p, dims=1)' - c)) +
            2 * ηp * dot(r, neg_entropy(p, dims=2)))
end

function extragradient_ot(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    adjust::Bool=true,
    p0=Nothing
) where {R}
    # input 
    # W∞ = maximum(W', dims=2)
    W∞ = maximum(W)
    ηp = args.eta_p / 2
    n = size(r, 1)
    if isa(W, CuArray)
        μ⁺ = 0.5 * CUDA.ones(R, (n, 1))
    else
        μ⁺ = 0.5 * ones(R, (n, 1))
    end

    μ⁻ = copy(μ⁺)
    ν⁺ = copy(μ⁺)
    ν⁻ = copy(μ⁺)
    μ⁻t = copy(μ⁺)
    μ⁺t = copy(μ⁺)
    # eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3)
    eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3 / n)
    ηπ = args.C2 / sqrt(args.B) ./ r
    if p0 == Nothing
        if isa(W, CuArray)
            p = CUDA.ones(R, (n, n)) ./ (n)
        else
            p = ones(R, (n, n)) ./ (n)
        end
    else
        if isa(W, CuArray)
            p = CuArray(p0) ./ r
        else
            p = (p0) ./ r
        end
    end

    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (i - 1) % frequency == 0
            pr = r .* p
            feas = norm(sum(pr', dims=2) - c, 1)
            obj = dot(pr, W)
            pobj = primalv(p, W, W∞, ηp, r, c)
            dobj = dualv(μ⁺, μ⁻, 1.0, W, W∞, ηp, r, c)

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragrad_primal\n" elapsed_time i feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        # eta_mu = 1 ./ sum(r .* p, dims=1)'
        arg = (sum(r .* p, dims=1)' - c) .* eta_mu
        maxval = max.(arg, -arg)
        μ⁻t = μ⁻ .^ (1 - args.eta_mu) .* exp.(-arg - maxval)
        μ⁺t = μ⁺ .^ (1 - args.eta_mu) .* exp.(arg - maxval)
        normv = (μ⁻t + μ⁺t)
        μ⁻t = μ⁻t ./ normv
        μ⁺t = μ⁺t ./ normv
        pt = p .^ (1 - ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 / W∞ .+ (μ⁺ .- μ⁻)'))
        pt ./= sum(pt, dims=2)
        # eta_mu = 1 ./ sum(r .* pt, dims=1)'
        arg = (sum(r .* pt, dims=1)' - c) .* eta_mu
        maxval = max.(arg, -arg)
        μ⁻ = μ⁻ .^ (1 - args.eta_mu) .* exp.(-arg - maxval)
        μ⁺ = μ⁺ .^ (1 - args.eta_mu) .* exp.(arg - maxval)

        normv = (μ⁻ + μ⁺)
        μ⁻ ./= normv
        μ⁺ ./= normv

        p = p .^ (1 - ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 / W∞ .+ (μ⁺t .- μ⁻t)'))
        p ./= sum(p, dims=2)
        if adjust
            μ⁻a = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
            μ⁺a = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
            normv = (μ⁻a + μ⁺a)
            μ⁻ = μ⁻a ./ normv
            μ⁺ = μ⁺a ./ normv
        else
            copy!(μ⁺a, μ⁺)
            copy!(μ⁻a, μ⁻)
        end
    end
    return round(r .* p, r, c), μ⁺, μ⁻
end


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


function update_μ_residual(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺a::CuDeviceArray{R}, μ⁻a::CuDeviceArray{R}, residual::CuDeviceArray{R}, c::CuDeviceArray{R}, eta_muᵢ::CuDeviceArray{R}, eta_mu::R, B::R, adjust::Bool) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    eta_muᵢ_tid = 1 / eta_muᵢ[tid]
    difference = residual[tid] - c[tid]
    maxval = eta_muᵢ_tid * max(-difference, difference)
    new_μ⁺ = μ⁺a[tid]^(1 - eta_mu) * exp(eta_muᵢ_tid * (difference) - maxval)
    new_μ⁻ = μ⁻a[tid]^(1 - eta_mu) * exp(-eta_muᵢ_tid * (difference) - maxval)
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
    end

    return
end


function warp_logsumexp!(output::CuDeviceVector{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
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
            value = -(0.5 * W[i+local_id, tid_x] / W∞ * st + (2μ⁺[i+local_id] - 1)) / reg
            maxval = max(value, maxval)
        end
        maxval = CUDA.reduce_warp(max, maxval)
        sync_warp()
        if local_id == 0
            output[tid_x] = maxval
        end
        # return
        maxval = output[tid_x]

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * W[i+local_id, tid_x] / W∞ * st + (2μ⁺[i+local_id] - 1)) / reg
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

function warp_logsumexp_t_fused!(output::CuDeviceVector{T}, Wt::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(Wt, 1)  # Wt is transposed: first dim iterates columns j
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    # Precompute scalars to avoid divisions in the inner loop
    α = (0.5 * st) / (reg * W∞)
    invreg = one(T) / reg
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


function residual_c!(output::CuDeviceVector{T}, r::CuDeviceArray{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(r, 1)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        diff = (2μ⁺[tid_x] - 1)

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * st * W[i+local_id, tid_x] ./ W∞ + diff) / reg - logZi[i+local_id]
            local_acc += r[i+local_id] * exp(value)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = local_acc2
        end
        tid_x += nwarps
    end
    return
end

function test_warp_logsumexp!()
    rng = Xoshiro(0)
    N = 20
    r = CuArray(normalize!(rand(rng, N), 1))
    c = CuArray(normalize!(rand(rng, N), 1))
    μ⁺ = CuArray(rand(rng, N))
    μ⁻ = CuArray(rand(rng, N))
    maxvals = CUDA.zeros(Float64, N)
    sumvals = CUDA.zeros(Float64, N)
    residual_storage = CUDA.zeros(Float64, N)
    residual_storage2 = CUDA.zeros(Float64, N)
    normv = (μ⁺ + μ⁻)
    μ⁺ ./= normv
    μ⁻ ./= normv

    W = CuArray(rand(rng, N, N))
    ηp = 1e-3
    st = 0.02
    maximum!(maxvals, -(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundUp), RoundUp)
    sum!(sumvals, (exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals)))
    sumvals = log.(sumvals) + maxvals
    # sumvals = maxvals
    sumvals2 = CUDA.zeros(Float64, N)
    # Use transposed W for coalesced memory access in the fused kernel
    Wt = permutedims(W, (2, 1))
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals2, Wt, μ⁺, ηp, st, W∞)
    CUDA.synchronize()
    println(sumvals)
    println(sumvals2)
    @test norm(sumvals - sumvals2) ≈ 0 atol = 1e-8

    sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- sumvals .+ log.(r)))
    # @test sum(residual_storage) ≈ 1.0 atol = 1e-8
    @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    CUDA.synchronize()
    println(residual_storage2)
    println(residual_storage)
    @test norm(residual_storage2 - residual_storage) ≈ 0 atol = 1e-8

    println(CUDA.@elapsed begin
        maximum!(maxvals, -(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp)
        sum!(sumvals, (exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals)))
        sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
    end)

    println(CUDA.@elapsed begin
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals2, Wt, μ⁺, ηp, st, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    end)


end

# +-------------------------+
# |   This one brings joy   |
# +-------------------------+
#       (◠‿◠)
#     ノ( ♥ )
function extragradient_ot_dual(r::CuArray{R},
    c::CuArray{R},
    W::CuArray{R},
    args::EOTArgs{R},
    frequency::Int=50;
    _μ⁺=Nothing,
    _μ⁻=Nothing,
    s0=0.0
) where {R}
    W∞ = norm(W, Inf)
    # ε /= W∞
    n = size(r, 1)
    if _μ⁺ == Nothing
        μ⁺ = CUDA.ones(R, n) * 0.5
        μ⁻ = copy(μ⁺)
    else
        # μ⁺ = CUDA.ones(R, n) * 0.5
        μ⁺ = copy(_μ⁺)
        μ⁻ = copy(_μ⁻)
    end
    ηp = args.eta_p / 2 / W∞
    ηstep = ηp * W∞
    sumvals = CUDA.zeros(R, n)
    residual_storage = CUDA.zeros(R, (n))

    μ⁻a = copy(μ⁻)
    μ⁺a = copy(μ⁺)
    μ⁻t = copy(μ⁻)
    μ⁺t = copy(μ⁺)
    eta_mu = (c .+ args.C3 / n) ./ (args.C2)
    μ⁺p = copy(μ⁺)
    μ⁻p = copy(μ⁻)
    μ⁺pa = copy(μ⁺)
    μ⁻pa = copy(μ⁻)
    st = s0

    threads = 256
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    Wt = permutedims(W, (2, 1))
    @inline function infeas(μ⁺, μ⁻)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals, Wt, μ⁺, ηp, st, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage, r, W, μ⁺, sumvals, ηp, st, W∞)
    end
    # ηp = 0.1
    hr = sum(neg_entropy(r))
    for i in 1:args.itermax
        infeas(μ⁺p, μ⁻p)
        @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺t, μ⁻t, μ⁺, μ⁻, residual_storage, c, eta_mu, args.eta_mu, args.B, false)
        @cuda threads = threads blocks = linear_blocks update_μ(μ⁺pa, μ⁻pa, μ⁺p, μ⁻p, μ⁺, μ⁻, ηstep)
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            p = softmax(-(W * 0.5 * st / W∞ .+ (μ⁺p' .- μ⁻p')) ./ ηp, norm_dims=2)
            obj = dot(W, r .* p)
            # pr = r .* p
            primal_value = primalv(μ⁺p, μ⁻p, st, W, W∞, ηp, r, c) + 2ηp * hr
            dual_value = dualv(μ⁺, μ⁻, 1.0, W, W∞, ηp, r, c) + 2ηp * hr
            # feas = norm(sum(pr, dims=1)' - c, 1)
            feas = norm(c - residual_storage, 1)

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragrad_dual_cuda\n" elapsed_time i feas obj primal_value dual_value
            if feas < args.epsilon / 2 && primal_value - dual_value < args.epsilon / 2
                break
            end
        end
        # st = (1 - ηp^(2 / 3)) * st + ηp^(2 / 3)
        st = (1 - ηstep) * st + ηstep
        infeas(μ⁺pa, μ⁻pa)
        @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺, μ⁻, μ⁺, μ⁻, residual_storage, c, eta_mu, args.eta_mu, args.B, true)
        @cuda threads = threads blocks = linear_blocks update_μ(μ⁺p, μ⁻p, μ⁺p, μ⁻p, μ⁺t, μ⁻t, ηstep)
        # ηp *= 0.99
    end
    p = softmax(-(W * 0.5 * st ./ W∞ .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2)
    return r .* p, μ⁺, μ⁻, st
end


# +------------------------------+
# |   This one also brings joy   |
# +------------------------------+
#       (◠‿◠)
#     ノ( ♥ )
function extragradient_ot_dual(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    s0::Float64=0.0
) where {R}
    W∞ = norm(W, Inf)
    ηp = args.eta_p / 2W∞
    n = size(r, 1)

    μ⁺ = ones(R, n) * 0.5

    sumvals = zeros(R, n)
    residual_storage = zeros(R, n)
    maxvals = zeros(R, n)


    μ⁻ = copy(μ⁺)
    μ⁻a = copy(μ⁻)
    μ⁺a = copy(μ⁺)
    μ⁻t = copy(μ⁻)
    μ⁺t = copy(μ⁺)
    eta_mu = (c .+ args.C3 / n) ./ (args.C2)
    μ⁺p = copy(μ⁺)
    μ⁻p = copy(μ⁻)
    ηpstep = 0.8
    μ⁺pa = copy(μ⁺)
    μ⁻pa = copy(μ⁻)
    μ⁻a = copy(μ⁻)
    st = s0
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(μ⁺, μ⁻)

        maximum!(maxvals, -(0.5W * st / W∞ .+ (μ⁺' .- μ⁻')) ./ ηp)

        sum!(sumvals, exp.(-(0.5W * st / W∞ .+ (μ⁺' .- μ⁻')) ./ ηp .- maxvals))
        sum!(residual_storage', exp.(-(0.5W * st ./ W∞ .+ (μ⁺' .- μ⁻')) ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            p = softmax(-(0.5W * st / W∞ .+ (μ⁺p' .- μ⁻p')) ./ ηp, norm_dims=2)
            # println(p)
            # sleep(1)
            pr = r .* p
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = dot(round(pr, r, c), W)
            pobj = primalv(p, W, W∞, ηp, r, c)
            dobj = dualv(μ⁺, μ⁻, 1.0, W, W∞, ηp, r, c)
            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,dual_extrap\n" elapsed_time i feas obj pobj dobj
            if (pobj - dobj) < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        infeas(μ⁺p, μ⁻p)
        # copy!(eta_mu, residual_storage)
        μ⁻t = μ⁻a .^ (1 - args.eta_mu) .* exp.(-(residual_storage - c) ./ eta_mu)
        μ⁺t = μ⁺a .^ (1 - args.eta_mu) .* exp.((residual_storage - c) ./ eta_mu)
        normv = (μ⁻t + μ⁺t)
        μ⁻t = μ⁻t ./ normv
        μ⁺t = μ⁺t ./ normv

        μ⁺pa .= μ⁺p + ηpstep * ηp * (μ⁺a - μ⁺p)
        μ⁻pa .= μ⁻p + ηpstep * ηp * (μ⁻a - μ⁻p)
        normv = (μ⁺pa + μ⁻pa)
        μ⁺pa ./= normv
        μ⁻pa ./= normv

        st = (1 - ηpstep * ηp) * st + ηpstep * ηp




        infeas(μ⁺pa, μ⁻pa)
        μ⁻ = μ⁻a .^ (1 - args.eta_mu) .* exp.(-(residual_storage - c) ./ eta_mu)#(sum(r .* pt, dims=1)' - c))
        μ⁺ = μ⁺a .^ (1 - args.eta_mu) .* exp.((residual_storage - c) ./ eta_mu)
        normv = (μ⁻ + μ⁺)
        μ⁻ ./= normv
        μ⁺ ./= normv

        μ⁺p = μ⁺p + ηpstep * ηp * (μ⁺t - μ⁺p)
        μ⁻p = μ⁻p + ηpstep * ηp * (μ⁻t - μ⁻p)

        normv = (μ⁺p + μ⁻p)
        μ⁺p ./= normv
        μ⁻p ./= normv

        μ⁻a = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
        μ⁺a = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
        normv = (μ⁻a + μ⁺a)

        μ⁻a = μ⁻a ./ normv
        μ⁺a = μ⁺a ./ normv
        # println(norm(μ⁺a - μ⁺) + norm(μ⁻a - μ⁻))

    end
    p = softmax(-(0.5W / W∞ .+ (μ⁺' .- μ⁻')) ./ ηp, norm_dims=2)
    sumvals = logsumexp(-(0.5W / W∞ .+ (μ⁺' .- μ⁻')) ./ ηp, 2)
    return r .* p, μ⁺, μ⁻, sumvals
end

# +-----------------------------+
# | This one does not bring joy |
# +-----------------------------+
#          (¬_¬)
#        /(_____)\
function extragradient_ot_full_dual(r::AbstractArray{R},
    c::AbstractArray{R},
    _W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    s0::Float64=0.0
) where {R}
    W∞ = norm(_W, Inf)
    ηp = args.eta_p / 2 / W∞
    W = _W ./ W∞
    n = size(r, 1)

    μ⁺ = ones(R, n) * 0.5

    sumvals = zeros(R, n)
    residual_storage = zeros(R, n)
    maxvals = zeros(R, n)


    μ⁻ = copy(μ⁺)
    μ⁻t = copy(μ⁻)
    μ⁺t = copy(μ⁺)
    eta_mu = ηp #* args.C2 ./ (c .+ args.C3 / n)
    st = s0
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(μ⁺, μ⁻)

        maximum!(maxvals, -(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp)

        sum!(sumvals, exp.(-(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp .- maxvals))
        sum!(residual_storage', exp.(-(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    function ΔBProject!(μ⁺, μ⁻)
        normv = (μ⁺ + μ⁺)

        μ⁻ .= μ⁻ ./ normv
        μ⁺ .= μ⁺ ./ normv
        μ⁻adjust = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
        μ⁺adjust = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
        normv = (μ⁺adjust + μ⁻adjust)

        μ⁻ .= μ⁻adjust ./ normv
        μ⁺ .= μ⁺adjust ./ normv
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            p = softmax(-(W * 0.5 / W∞ .+ μ⁺' .- μ⁻') ./ ηp, norm_dims=2)
            pr = r .* p
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = dot(round(pr, r, c), _W)
            pobj = primalv(p, _W, W∞, ηp, r, c)
            dobj = dualv(μ⁺, μ⁻, W, W∞, ηp, r, c)
            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,dual_extrap\n" elapsed_time i feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end

        infeas(μ⁺, μ⁻)
        μ⁻t = μ⁻ .^ (1 - args.eta_mu) .* exp.(-eta_mu .* (residual_storage - c))
        μ⁺t = μ⁺ .^ (1 - args.eta_mu) .* exp.(eta_mu .* (residual_storage - c))
        ΔBProject!(μ⁺t, μ⁻t)
        infeas(μ⁺t, μ⁻t)
        μ⁻ = μ⁻ .^ (1 - args.eta_mu) .* exp.(-eta_mu .* (residual_storage - c))
        μ⁺ = μ⁺ .^ (1 - args.eta_mu) .* exp.(eta_mu .* (residual_storage - c))
        ΔBProject!(μ⁺, μ⁻)
        # println(norm(μ⁺ - μ⁻))

    end
    p = softmax(-(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp, norm_dims=2)
    return r .* p, μ⁺, μ⁻
end



