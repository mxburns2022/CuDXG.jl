using CUDA
using Printf
using LinearAlgebra
using Random
using Test

function dualv(μ⁺, μ⁻, _W, W, W∞, ηp, r, c)
    pμ = softmax(-(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp, norm_dims=2)
    return 0.5 * dot(_W, r .* pμ) + W∞ * dot((sum(r .* pμ, dims=1)' - c), μ⁺ - μ⁻) + ηp * dot(r, neg_entropy(pμ, dims=2))
end
function primalv(μ⁺p::TA, μ⁻p::TA, st::R, _W::TM, W::TM, W∞::R, ηp::R, r::TA, c::TA) where {TA,TM,R}
    pμ = softmax(-(W * st * 0.5 .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2)
    return 0.5 * dot(_W, r .* pμ) + W∞ * norm((sum(r .* pμ, dims=1)' - c), 1) + ηp * dot(r, neg_entropy(pμ, dims=2))
end

function primalv(p::TM, _W::TM, W∞::R, ηp::R, r::TA, c::TA) where {TA,TM,R}
    return 0.5 * dot(_W, r .* p) + W∞ * norm((sum(r .* p, dims=1)' - c), 1) + ηp * dot(r, neg_entropy(p, dims=2))
end


function extragradient_ot(r::AbstractArray{R},
    c::AbstractArray{R},
    _W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    adjust::Bool=true,
    p0=Nothing
) where {R}
    # input 
    W∞ = norm(_W, Inf)
    ηp = args.eta_p / 2W∞
    eta_mu_scale = args.eta_mu
    W = _W ./ W∞
    epsilon = args.epsilon / W∞
    n = size(r, 1)
    if isa(W, CuArray)
        μ⁺ = 0.5 * CUDA.ones(R, (n, 1))
    else
        μ⁺ = 0.5 * ones(R, (n, 1))
    end

    μ⁻ = copy(μ⁺)
    μ⁻a = copy(μ⁺)
    μ⁺a = copy(μ⁺)
    μ⁻t = copy(μ⁺)
    μ⁺t = copy(μ⁺)
    # eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3)
    eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3 / n)
    ηπ = args.C2 / sqrt(args.B) ./ r
    if p0 == Nothing
        p = softmax(-r .* (W * 0.5 .+ (μ⁺' .- μ⁻')) ./ ηπ; norm_dims=2)
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
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = W∞ * dot(p, W)
            pobj = primalv(p, W, W∞, ηp, r, c)
            dobj = dualv(μ⁺a, μ⁻a, _W, W, W∞, ηp, r, c)

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragrad_primal\n" elapsed_time i feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        arg = eta_mu .* (sum(r .* p, dims=1)' - c)
        maxval = max.(arg, -arg)
        μ⁻t = μ⁻a .^ (1 - args.eta_mu) .* exp.(-arg - maxval)
        μ⁺t = μ⁺a .^ (1 - args.eta_mu) .* exp.(arg - maxval)
        normv = (μ⁻t + μ⁺t)
        μ⁻t = μ⁻t ./ normv
        μ⁺t = μ⁺t ./ normv
        pt = p .^ (1 - ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 .+ (μ⁺a .- μ⁻a)'))
        pt ./= sum(pt, dims=2)
        arg = eta_mu .* (sum(r .* pt, dims=1)' - c)
        maxval = max.(arg, -arg)
        μ⁻ = μ⁻a .^ (1 - args.eta_mu) .* exp.(-arg - maxval)
        μ⁺ = μ⁺a .^ (1 - args.eta_mu) .* exp.(arg - maxval)

        normv = (μ⁻ + μ⁺)
        μ⁻ ./= normv
        μ⁺ ./= normv

        p = p .^ (1 - ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 .+ (μ⁺t .- μ⁻t)'))
        p ./= sum(p, dims=2)
        if adjust
            μ⁻a = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
            μ⁺a = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
            normv = (μ⁻a + μ⁺a)
            μ⁻a ./= normv
            μ⁺a ./= normv
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
    eta_muᵢ_tid = eta_muᵢ[tid]
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


function warp_logsumexp!(output::CuDeviceVector{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, reg::T, st::T) where T
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
            value = -(0.5 * W[i+local_id, tid_x] * st + μ⁺[i+local_id] - μ⁻[i+local_id]) / reg
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
            value = -(0.5 * W[i+local_id, tid_x] * st + μ⁺[i+local_id] - μ⁻[i+local_id]) / reg
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

function warp_logsumexp_t_fused!(output::CuDeviceVector{T}, Wt::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, reg::T, st::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(Wt, 1)  # Wt is transposed: first dim iterates columns j
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    # Precompute scalars to avoid divisions in the inner loop
    α = (0.5 * st) / reg
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
            output[tid_x] = log(s) + m
        end
        tid_x += nwarps
    end
    return
end


function residual_c!(output::CuDeviceVector{T}, c::CuDeviceArray{T}, r::CuDeviceArray{T}, W::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T) where T
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
        diff = μ⁺[tid_x] - μ⁻[tid_x]

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
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals2, Wt, μ⁺, μ⁻, ηp, st)
    CUDA.synchronize()
    println(sumvals)
    println(sumvals2)
    @test norm(sumvals - sumvals2) ≈ 0 atol = 1e-8

    sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- sumvals .+ log.(r)))
    # @test sum(residual_storage) ≈ 1.0 atol = 1e-8
    @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, c, r, W, μ⁺, μ⁻, sumvals, ηp, st)
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
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals2, Wt, μ⁺, μ⁻, ηp, st)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, c, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    end)


end

# +-------------------------+
# |   This one brings joy   |
# +-------------------------+
#       (◠‿◠)
#     ノ( ♥ )
function extragradient_ot_dual(r::CuArray{R},
    c::CuArray{R},
    _W::CuArray{R},
    args::EOTArgs{R},
    frequency::Int=50;
    _μ⁺=Nothing,
    _μ⁻=Nothing,
    s0=0.0
) where {R}
    W∞ = norm(_W, Inf)
    ηp = args.eta_p / 2 / W∞
    W = _W ./ W∞
    n = size(r, 1)
    if _μ⁺ == Nothing
        μ⁺ = CUDA.ones(R, n) * 0.5
        μ⁻ = copy(μ⁺)
    else
        # μ⁺ = CUDA.ones(R, n) * 0.5
        μ⁺ = copy(_μ⁺)
        μ⁻ = copy(_μ⁻)
    end

    sumvals = CUDA.zeros(R, n)
    residual_storage = CUDA.zeros(R, (n))
    maxvals = CUDA.zeros(R, (n))

    μ⁻a = copy(μ⁻)
    μ⁺a = copy(μ⁺)
    μ⁻t = copy(μ⁻)
    μ⁺t = copy(μ⁺)
    eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3 / n)
    μ⁺p = copy(μ⁺)
    μ⁻p = copy(μ⁻)
    μ⁺pa = copy(μ⁺)
    μ⁻pa = copy(μ⁻)
    μ⁻a = copy(μ⁻)
    st = s0

    # function infeas(μ⁺, μ⁻)
    #     p = softmax(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp, norm_dims=2)
    #     return sum(r .* p, dims=1)' - c
    # end
    # maxj = reshape(maximum(W, dims=2), n)
    # maximum!(maxvals, -(W * 0.5) ./ ηp)
    threads = 256
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    Wt = permutedims(W, (2, 1))
    time_start = time_ns()
    @inline function infeas(μ⁺, μ⁻)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_t_fused!(sumvals, Wt, μ⁺, μ⁻, ηp, st)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage, c, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    end
    # ηp = 0.1
    for i in 1:args.itermax
        infeas(μ⁺p, μ⁻p)
        @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺t, μ⁻t, μ⁺a, μ⁻a, residual_storage, c, eta_mu, args.eta_mu, args.B, false)
        @cuda threads = threads blocks = linear_blocks update_μ(μ⁺pa, μ⁻pa, μ⁺p, μ⁻p, μ⁺a, μ⁻a, ηp)
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            obj = dot(W, r .* softmax(-(W * 0.5 * st .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2))
            # pr = r .* p
            # feas = norm(sum(pr, dims=1)' - c, 1)
            feas = norm(c - residual_storage, 1)
            pobj = primalv(μ⁺p, μ⁻p, st, _W, W, W∞, ηp, r, c)
            dobj = dualv(μ⁺a, μ⁻a, _W, W, W∞, ηp, r, c)

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragrad_dual_cuda\n" elapsed_time i feas obj pobj dobj
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        # st = (1 - ηp^(2 / 3)) * st + ηp^(2 / 3)
        st = (1 - ηp) * st + ηp
        infeas(μ⁺pa, μ⁻pa)
        @cuda threads = threads blocks = linear_blocks update_μ_residual(μ⁺, μ⁻, μ⁺a, μ⁻a, residual_storage, c, eta_mu, args.eta_mu, args.B, true)
        @cuda threads = threads blocks = linear_blocks update_μ(μ⁺p, μ⁻p, μ⁺p, μ⁻p, μ⁺t, μ⁻t, ηp)
        # ηp *= 0.99
    end
    p = softmax(-(W * 0.5 * st .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2)
    return r .* p, μ⁺, μ⁻, st
end


# +------------------------------+
# |   This one also brings joy   |
# +------------------------------+
#       (◠‿◠)
#     ノ( ♥ )
function extragradient_ot_dual(r::AbstractArray{R},
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
    μ⁻a = copy(μ⁻)
    μ⁺a = copy(μ⁺)
    μ⁻t = copy(μ⁻)
    μ⁺t = copy(μ⁺)
    eta_mu = args.C2 ./ (c .+ args.C3 / n)
    μ⁺p = copy(μ⁺)
    μ⁻p = copy(μ⁻)
    μ⁺pa = copy(μ⁺)
    μ⁻pa = copy(μ⁻)
    μ⁻a = copy(μ⁻)
    st = s0
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(μ⁺, μ⁻)

        maximum!(maxvals, -(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp)

        sum!(sumvals, exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals))
        sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            p = softmax(-(W * 0.5 * st .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2)
            pr = r .* p
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = dot(round(pr, r, c), _W)
            pobj = primalv(p, _W, W∞, ηp, r, c)
            dobj = dualv(μ⁺a, μ⁻a, _W, W, W∞, ηp, r, c)
            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,%.14e,dual_extrap\n" elapsed_time i feas obj pobj dobj norm(μ⁺a - μ⁺) + norm(μ⁻a - μ⁻)
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end

        infeas(μ⁺p, μ⁻p)
        μ⁻t = μ⁻a .^ (1 - args.eta_mu) .* exp.(-eta_mu .* (residual_storage - c))
        μ⁺t = μ⁺a .^ (1 - args.eta_mu) .* exp.(eta_mu .* (residual_storage - c))

        normv = (μ⁻t + μ⁺t)
        μ⁻t = μ⁻t ./ normv
        μ⁺t = μ⁺t ./ normv
        μ⁻ta = max.(μ⁻t, exp(-args.B) .* max.(μ⁺t, μ⁻t))
        μ⁺ta = max.(μ⁺t, exp(-args.B) .* max.(μ⁺t, μ⁻t))
        normv = (μ⁻ta + μ⁺ta)

        μ⁻t = μ⁻ta ./ normv
        μ⁺t = μ⁺ta ./ normv

        μ⁺pa .= μ⁺p + ηp * (μ⁺a - μ⁺p)
        μ⁻pa .= μ⁻p + ηp * (μ⁻a - μ⁻p)
        normv = (μ⁺pa + μ⁻pa)
        μ⁺pa ./= normv
        μ⁻pa ./= normv

        st = (1 - ηp) * st + ηp

        infeas(μ⁺pa, μ⁻pa)
        μ⁻ = μ⁻a .^ (1 - args.eta_mu) .* exp.(-eta_mu .* (residual_storage - c))#(sum(r .* pt, dims=1)' - c))
        μ⁺ = μ⁺a .^ (1 - args.eta_mu) .* exp.(eta_mu .* (residual_storage - c))
        normv = (μ⁻ + μ⁺)
        μ⁻ ./= normv
        μ⁺ ./= normv

        μ⁺p = μ⁺p + ηp * (μ⁺t - μ⁺p)
        μ⁻p = μ⁻p + ηp * (μ⁻t - μ⁻p)

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
    p = softmax(-(W * 0.5 * st .+ μ⁺p' .- μ⁻p') ./ ηp, norm_dims=2)
    return r .* p, μ⁺, μ⁻, st
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
            p = softmax(-(W * 0.5 .+ μ⁺' .- μ⁻') ./ ηp, norm_dims=2)
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



