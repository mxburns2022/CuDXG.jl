using CUDA



function residual_c!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img2[1, tid_x]
        pix1g = img2[2, tid_x]
        pix1b = img2[3, tid_x]
        diff = μ⁺[diff] - μ⁻[diff]

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
            value = -(l2dist + diff) / reg - logZi
            local_acc += exp(value)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        sync_warp()
        if local_id == 0
            output[tid_x] = -N - local_acc2
        end
        tid_x += nwarps
    end
    return
end

@inline function rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
    return (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
end

function naive_findmaxindex_ct!(output::CuDeviceVector{Int}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, φ::CuDeviceVector{T}, ψ::CuDeviceVector{T}, reg::T) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    if tid_x > N
        return
    end
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    φi = φ[tid_x]
    maxval = -Inf
    maxind = Int(-1)
    for i in 1:N
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        value = -(l2dist) / reg + φi + ψ[i]
        if value > maxval
            maxval = value
            maxind = i
        end
    end
    output[tid_x] = maxind


    return
end

function warp_logsumexp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, μ⁻::CuDeviceVector{T}, reg::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img1[1, tid_x]
        pix1g = img1[2, tid_x]
        pix1b = img1[3, tid_x]
        maxval = -Inf
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
            value = -(l2dist + μ⁺[i+local_id] - μ⁻[i+local_id]) / reg
            maxval = max(value, maxval)
        end
        maxval = CUDA.reduce_warp(max, maxval)
        if local_id == 0
            output[tid_x] = maxval
        end
        sync_warp()
        maxval = output[tid_x]

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
            value = -(l2dist + μ⁺[i+local_id] - μ⁻[i+local_id]) / reg
            local_acc += exp(value - maxval)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        sync_warp()
        if local_id == 0
            output[tid_x] = (log(local_acc2) + maxval)
        end
        tid_x += nwarps
    end
    return
end

function naive_logsumexp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, ψ::CuDeviceVector{T}, reg::T) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    if tid_x > N
        return
    end
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    # φi = φ[tid_x]
    maxval = -Inf
    for i in 1:N
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        value = -(l2dist) / reg + ψ[i]
        maxval = max(maxval, value)
    end
    acc = 0.0
    for i in 1:N
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        value = -(l2dist) / reg + ψ[i]
        acc += exp(value - maxval)
    end
    output[tid_x] = -log(N) - log(acc) - maxval
    # end
    return
end
@inline function reduce_block(op, val::T, neutral, shared) where T
    threads = blockDim().x
    thread = threadIdx().x

    # shared mem for a complete reduction
    @inbounds shared[thread] = val

    # perform a reduction
    d = 1
    while d < threads
        sync_threads()
        index = 2 * d * (thread - 1) + 1
        @inbounds if index <= threads
            other_val = if index + d <= threads
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if thread == 1
        val = @inbounds shared[thread]
    end

    return val
end
function block_logsumexp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, ψ::CuDeviceVector{T}, reg::T) where T
    step = blockDim().x
    # nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = blockIdx().x
    N = size(img1, 2)
    local_id = threadIdx().x - 1
    shared = CuStaticSharedArray(T, (1024,))

    if tid_x > N
        return
    end
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    maxval = -Inf
    for i in 1:step:N
        # if tid_x == 1 && local_id == 0
        #     @cuprintln("$(step)")
        # end
        if i + local_id > N
            break
        end
        pix2r = img2[1, i+local_id]
        pix2g = img2[2, i+local_id]
        pix2b = img2[3, i+local_id]
        l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
        value = -(l2dist) / reg + ψ[i+local_id]
        maxval = max(value, maxval)
    end
    maxval = reduce_block(max, maxval, -Inf, shared)
    if local_id == 0
        output[tid_x] = maxval
    end
    sync_threads()
    maxval = output[tid_x]

    local_acc = 0.0
    for i in 1:step:N
        if i + local_id > N
            break
        end
        pix2r = img2[1, i+local_id]
        pix2g = img2[2, i+local_id]
        pix2b = img2[3, i+local_id]
        l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
        value = -(l2dist) / reg + ψ[i+local_id]
        local_acc += exp(value - maxval)
    end
    local_acc2 = reduce_block(+, local_acc, 0.0, shared)#CUDA.reduce_warp(+, local_acc)
    sync_threads()
    if local_id == 0
        output[tid_x] = -log(N) - (log(local_acc2) + maxval)
    end
    # tid_x += nwarps
    return
end
function test_logsumexp_kernel()
    N = 100
    img1 = rand(3, N)
    img2 = rand(3, N)
    W = zeros(N, N)
    η = 1e-2
    for (i, j) in product(axes(W, 1), axes(W, 2))
        pix1r = img1[1, i]
        pix1g = img1[2, i]
        pix1b = img1[3, i]
        pix2r = img2[1, j]
        pix2g = img2[2, j]
        pix2b = img2[3, j]
        W[i, j] = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
    end
    ψ = zeros(N)
    φ = zeros(N)

    maxvalsr = maximum(-W ./ η .+ ψ', dims=2)
    φ = -log(N) .- (log.(sum(exp.(-W ./ η .+ ψ' .- maxvalsr), dims=2)) + maxvalsr)
    φ_cu = CuArray(reshape(φ, N))

    maxvalsc = maximum(-W ./ η .+ φ, dims=1)
    ψ = -log(N) .- (log.(sum(exp.(-W ./ η .+ φ .- maxvalsc), dims=1)) + maxvalsc)'
    ψ_cu = CuArray(reshape(ψ, N))
    maxvalsr = maximum(-W ./ η .+ ψ', dims=2)
    logsumexp_value_r = -log(N) .- (log.(sum(exp.(-W ./ η .+ ψ' .- maxvalsr), dims=2)) + maxvalsr)
    maxvalsc = maximum((-W ./ η .+ φ), dims=1)
    logsumexp_value_c = -log(N) .- (log.(sum(exp.(-W ./ η .+ φ .- maxvalsc), dims=1)) + maxvalsc)'
    output_r = CuArray(zeros(N))
    output_c = CuArray(zeros(N))
    img1_cu = CuArray(img1)
    img2_cu = CuArray(img2)
    @cuda threads = 256 blocks = 32 warp_logsumexp_ct!(output_r, img1_cu, img2_cu, ψ_cu, η)
    @cuda threads = 256 blocks = 32 warp_logsumexp_ct!(output_c, img2_cu, img1_cu, φ_cu, η)
    CUDA.synchronize()
    @test norm(Array(output_r) - logsumexp_value_r) ≈ 0 atol = 1e-10
    @test norm(Array(output_c) - logsumexp_value_c) ≈ 0 atol = 1e-10

    @cuda threads = 256 blocks = 2 warp_logsumexp_ct!(output_r, img1_cu, img2_cu, ψ_cu, η)
    @cuda threads = 256 blocks = 2 warp_logsumexp_ct!(output_c, img2_cu, img1_cu, φ_cu, η)
    CUDA.synchronize()
    @test norm(Array(output_r) - logsumexp_value_r) ≈ 0 atol = 1e-10
    @test norm(Array(output_c) - logsumexp_value_c) ≈ 0 atol = 1e-10

    @cuda threads = 256 blocks = 32 residual!(output_r, img1_cu, img2_cu, φ_cu, ψ_cu, η)
    @cuda threads = 256 blocks = 32 residual!(output_c, img2_cu, img1_cu, ψ_cu, φ_cu, η)
    CUDA.synchronize()
    marginal_r = sum(exp.(-W ./ η .+ φ .+ ψ'), dims=2)
    marginal_c = sum(exp.(-W ./ η .+ φ .+ ψ')', dims=2)
    residual_r = ones(N) / N - marginal_r
    residual_c = ones(N) / N - marginal_c
    @test norm(residual_r - Array(output_r)) ≈ 0 atol = 1e-10
    @test norm(residual_c - Array(output_c)) ≈ 0 atol = 1e-10
    # marginal_c
end