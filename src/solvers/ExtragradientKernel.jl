using CUDA
using IterTools
using LinearAlgebra



function residual_spp_c!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    α = 0.5st
    invreg = 1 / reg
    for i in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img2[1, tid_x]
        pix1g = img2[2, tid_x]
        pix1b = img2[3, tid_x]
        diff = 2μ⁺[tid_x] - 1

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img1[1, i+local_id]
            pix2g = img1[2, i+local_id]
            pix2b = img1[3, i+local_id]
            l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
            value = -(l2dist * α + diff) * invreg - logZi[i+local_id]
            local_acc += exp(value)
        end
        local_acc2 = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = local_acc2 / N - 1 / N
        end
        tid_x += nwarps
    end
    return
end


function naive_findmaxindex_spp_ct!(output_img::CuDeviceMatrix{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    if tid_x > N
        return
    end
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    avgr = 0.0
    avgg = 0.0
    avgb = 0.0
    probsum = 0.0
    norm = logZi[tid_x]
    for i in 1:N
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        diff = 2μ[i] - 1
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        prob = exp(-(l2dist * 0.5st + diff) / reg - norm)
        avgr += img2[1, i] * prob
        avgg += img2[2, i] * prob
        avgb += img2[3, i] * prob
    end
    # @cuprintln(probsum)
    output_img[1, tid_x] = avgr
    output_img[2, tid_x] = avgg
    output_img[3, tid_x] = avgb

    return
end
function update_μ_ct(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺_1::CuDeviceArray{R}, μ⁻_1::CuDeviceArray{R}, μ⁺_2::CuDeviceArray{R}, μ⁻_2::CuDeviceArray{R}, η) where R
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


function update_μ_residual_ct(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺a::CuDeviceArray{R}, μ⁻a::CuDeviceArray{R}, residual::CuDeviceArray{R}, ημᵢ::R, ημ::R, B::R, adjust::Bool) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    difference = residual[tid]
    maxval = ημᵢ * max(-difference, difference)
    new_μ⁺ = μ⁺a[tid]^(1 - ημ) * exp(ημᵢ * (difference) - maxval)
    new_μ⁻ = μ⁻a[tid]^(1 - ημ) * exp(-ημᵢ * (difference) - maxval)
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
function warp_logsumexp_spp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T) where T
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
            value = -(l2dist * 0.5 * st + 2μ⁺[i+local_id] - 1.) / reg
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
            value = -(l2dist * 0.5 * st + 2μ⁺[i+local_id] - 1) / reg
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

function naive_logsumexp_spp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
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

function extragradient_color_transfer(img1::CuArray{T}, img2::CuArray{T}, η::T, maxiter::Int=10000, frequency::Int=100) where T<:Real
    N = size(img1, 2)
    μ⁺ = 0.5 * CUDA.ones(T, N)
    μ⁻ = 0.5 * CUDA.ones(T, N)
    μt⁺ = 0.5 * CUDA.ones(T, N)
    μt⁻ = 0.5 * CUDA.ones(T, N)
    ν⁺ = 0.5 * CUDA.ones(T, N)
    ν⁻ = 0.5 * CUDA.ones(T, N)
    νt⁺ = 0.5 * CUDA.ones(T, N)
    νt⁻ = 0.5 * CUDA.ones(T, N)
    residual_cache = CUDA.zeros(T, N)
    sumvals = CUDA.zeros(T, N)
    eta_mu = 1. / (1.01 / N)
    threads = 256
    B = 1.0
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)
    st = 0.0
    @inline function infeas(μ⁺)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, μ⁺, η, st)
        CUDA.synchronize()
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st)
        CUDA.synchronize()
    end
    for i in 1:maxiter
        # perform the extragradient step
        infeas(ν⁺)
        # println("update1")
        @cuda threads = threads blocks = linear_blocks update_μ_residual_ct(μt⁺, μt⁻, μ⁺, μ⁻, residual_cache, eta_mu, 0.0, B, false)
        # println("update2")
        @cuda threads = threads blocks = linear_blocks update_μ_ct(νt⁺, νt⁻, ν⁺, ν⁻, μ⁺, μ⁻, η)
        st = (1 - η) * st + η
        infeas(νt⁺)
        @cuda threads = threads blocks = linear_blocks update_μ_residual_ct(μ⁺, μ⁻, μ⁺, μ⁻, residual_cache, eta_mu, 0.0, B, true)
        @cuda threads = threads blocks = linear_blocks update_μ_ct(ν⁺, ν⁻, ν⁺, ν⁻, μt⁺, μt⁻, η)

        CUDA.synchronize()
        if (i - 1) % frequency == 0
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, ν⁺, η, st)
            @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, ν⁺, sumvals, η, st)
            CUDA.synchronize()
            residual_c = sum(abs.(residual_cache))
            # println(residual_cache)
            objective = -sum(2μ⁺ .- 1) / N - η * sum(sumvals) / N
            println("$(i) $(residual_c) $(objective)")
        end
    end
    output_img1 = CUDA.zeros(Float64, 3, N)
    output_img2 = CUDA.zeros(Float64, 3, N)

    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, ν⁺, η, st)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img1, img1, img2, ν⁺, sumvals, η, st)
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img2, img1, ν⁺, η, st)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img2, img2, img1, ν⁺, sumvals, η, st)
    return ν⁺, Array(output_img1), Array(output_img2)

end

function extragradient_color_transfer(f1::String, f2::String, out_f1::String, out_f2::String, η::Float64, resolution::Int, maxiter::Int)
    img1, dims1 = load_rgb(f1; cuda=true, size=(resolution, resolution))
    img2, dims2 = load_rgb(f2; cuda=true, size=(resolution, resolution))
    _, img1_new, img2_new = extragradient_color_transfer(img1, img2, η, maxiter)
    save_image(out_f1, img1_new, dims1)
    save_image(out_f2, img2_new, dims2)
end