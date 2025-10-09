using CUDA
using IterTools
using LinearAlgebra



function residual_spp_c!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    # step = warpsize()
    # nwarps = (gridDim().x * blockDim().x) ÷ step
    # tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    # N = size(img1, 2)
    # N_outer = Int(ceil(N / nwarps))
    # local_id = (threadIdx().x - 1) % step
    # α = 0.5st / W∞
    # invreg = 1 / reg
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = T(0.5) * st / W∞
    invreg = one(T) / reg
    Ntiles = (N) ÷ step
    for _ in 1:N_outer
        local_acc = 0.0
        @inbounds begin
            pix1r = img2[1, tid_x]
            pix1g = img2[2, tid_x]
            pix1b = img2[3, tid_x]
            diff = (2μ⁺[tid_x] - 1.)
        end
        for tile in 0:Ntiles-1
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img1[1, j+local_id]
                pix2g = img1[2, j+local_id]
                pix2b = img1[3, j+local_id]

                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = muladd(muladd(l2dist, c1, diff), -invreg, -logZi[j+local_id])
            end

            local_acc += exp(value)
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img1[1, j+local_id]
                pix2g = img1[2, j+local_id]
                pix2b = img1[3, j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = muladd(muladd(l2dist, c1, diff), -invreg, -logZi[j+local_id])
            end

            local_acc += exp(value)
        end
        local_acc = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = local_acc / N
            end
        end
        tid_x += nwarps
    end
    return
end


function naive_findmaxindex_spp_ct!(output_img::CuDeviceMatrix{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
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
        diff = (2μ[i] - 1)
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        prob = exp(-(l2dist / W∞ * 0.5st + diff) / reg - norm)
        avgr += img2[1, i] * prob
        avgg += img2[2, i] * prob
        avgb += img2[3, i] * prob
        probsum += prob
    end

    output_img[1, tid_x] = avgr
    output_img[2, tid_x] = avgg
    output_img[3, tid_x] = avgb

    return
end

function naive_findmaxindex_spp_ct_t!(output_img::CuDeviceMatrix{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    if tid_x > N
        return
    end
    pix1r = img2[1, tid_x]
    pix1g = img2[2, tid_x]
    pix1b = img2[3, tid_x]
    diff = (2μ[tid_x] - 1)
    avgr = 0.0
    avgg = 0.0
    avgb = 0.0
    probsum = 0.0
    for i in 1:N
        pix2r = img1[1, i]
        pix2g = img1[2, i]
        pix2b = img1[3, i]
        norm = logZi[i]
        l2dist = rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
        prob = exp(-(l2dist / W∞ * 0.5st + diff) / reg - norm)
        avgr += img1[1, i] * prob
        avgg += img1[2, i] * prob
        avgb += img1[3, i] * prob
        probsum += prob
    end
    if abs(probsum - 1.0) > 1e-10
        @cuprintln(probsum)
    end
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


function update_μ_residual_ct(μ⁺::CuDeviceArray{R}, μ⁻::CuDeviceArray{R}, μ⁺a::CuDeviceArray{R}, μ⁻a::CuDeviceArray{R}, residual::CuDeviceArray{R}, ημᵢ::CuDeviceArray{R}, ημ::R, B::R, adjust::Bool) where R
    N = size(μ⁺, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    difference = residual[tid] - one(R) / N
    maxval = ημᵢ[tid] * max(-difference, difference)
    new_μ⁺ = μ⁺a[tid]^(1 - ημ) * exp(ημᵢ[tid] * (difference) - maxval)
    new_μ⁻ = μ⁻a[tid]^(1 - ημ) * exp(-ημᵢ[tid] * (difference) - maxval)
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
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
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
            value = -(l2dist * 0.5 * st / W∞ + (2μ⁺[i+local_id] - 1.)) / reg
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
            value = -(l2dist * 0.5 * st / W∞ + (2μ⁺[i+local_id] - 1.)) / reg
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
function max_logsumexp_spp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T}, img2::CuDeviceMatrix{T}) where T
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
            maxval = max(l2dist, maxval)
        end
        maxval = CUDA.reduce_warp(max, maxval)
        if local_id == 0
            output[tid_x] = maxval
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

const smemsize = 256
function warp_logsumexp_spp_ct_opt_smem!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = T(0.5) * st / W∞
    invreg = one(T) / reg

    smem = CuStaticSharedArray(T, 4 * smemsize)
    warpiters = smemsize ÷ warpsize()
    Ntiles = (N) ÷ smemsize
    warpiter_epi = (N - Ntiles * smemsize) ÷ warpsize()
    epi_size = N - (Ntiles * smemsize + warpiter_epi * warpsize())
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        @inbounds begin
            pix1r = img1[1, tid_x]
            pix1g = img1[2, tid_x]
            pix1b = img1[3, tid_x]
        end
        maxval = -Inf

        for blocktile in 0:Ntiles-1
            if threadIdx().x <= smemsize
                @inbounds begin
                    smem[threadIdx().x] = img2[1, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+smemsize] = img2[2, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+2smemsize] = img2[3, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+3smemsize] = 2μ⁺[blocktile*smemsize+threadIdx().x] - 1
                end
            end
            sync_threads()
            for tile in 0:warpiters-1
                @inbounds begin
                    j = tile * warpsize()
                    pix2r = smem[j+local_id+1]
                    pix2g = smem[j+local_id+smemsize+1]
                    pix2b = smem[j+local_id+2smemsize+1]
                    muval = smem[j+local_id+3smemsize+1]

                    dr = (pix1r - pix2r)
                    dg = (pix1g - pix2g)
                    db = (pix1b - pix2b)
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                    value = -(muladd(l2dist, c1, muval)) * invreg
                    # end
                    maxval = max(maxval, value)
                end
                # break
            end
            # break
        end
        if (Ntiles) * smemsize + threadIdx().x <= N
            if threadIdx().x < smemsize
                @inbounds begin

                    smem[threadIdx().x] = img2[1, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+smemsize] = img2[2, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+2smemsize] = img2[3, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+3smemsize] = μ⁺[Ntiles*smemsize+threadIdx().x]
                end
            end
        end
        sync_threads()
        for tile in 0:warpiter_epi-1
            j = tile * warpsize()
            @inbounds begin
                pix2r = smem[j+local_id+1]
                pix2g = smem[j+local_id+smemsize+1]
                pix2b = smem[j+local_id+2smemsize+1]
                muval = smem[j+local_id+3smemsize+1]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end
            maxval = max(value, maxval)
        end
        # sync_threads()
        if local_id <= epi_size
            @inbounds begin
                pix2r = smem[local_id+1]
                pix2g = smem[local_id+smemsize+1]
                pix2b = smem[local_id+2smemsize+1]
                muval = smem[local_id+3smemsize+1]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
                maxval = max(value, maxval)
            end
        end
        maxval = CUDA.reduce_warp(max, maxval)
        maxval = CUDA.shfl_sync(CUDA.FULL_MASK, maxval, 1)

        local_acc = T(0.0)
        for blocktile in 0:Ntiles-1
            if threadIdx().x <= smemsize
                @inbounds begin
                    smem[threadIdx().x] = img2[1, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+smemsize] = img2[2, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+2smemsize] = img2[3, blocktile*smemsize+threadIdx().x]
                    smem[threadIdx().x+3smemsize] = 2μ⁺[blocktile*smemsize+threadIdx().x] - 1
                end
            end
            sync_threads()
            for tile in 0:warpiters-1
                @inbounds begin
                    j = tile * warpsize()
                    pix2r = smem[j+local_id+1]
                    pix2g = smem[j+local_id+smemsize+1]
                    pix2b = smem[j+local_id+2smemsize+1]
                    muval = smem[j+local_id+3smemsize+1]

                    dr = (pix1r - pix2r)
                    dg = (pix1g - pix2g)
                    db = (pix1b - pix2b)
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                    value = -(muladd(l2dist, c1, muval)) * invreg
                    local_acc += exp(value - maxval)
                end
            end
            # break
        end
        if (Ntiles) * smemsize + threadIdx().x <= N
            if threadIdx().x < smemsize
                @inbounds begin
                    smem[threadIdx().x] = img2[1, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+smemsize] = img2[2, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+2smemsize] = img2[3, Ntiles*smemsize+threadIdx().x]
                    smem[threadIdx().x+3smemsize] = μ⁺[Ntiles*smemsize+threadIdx().x]
                end
            end
        end
        sync_threads()
        for tile in 0:warpiter_epi-1
            j = tile * warpsize()
            @inbounds begin
                pix2r = smem[j+local_id+1]
                pix2g = smem[j+local_id+smemsize+1]
                pix2b = smem[j+local_id+2smemsize+1]
                muval = smem[j+local_id+3smemsize+1]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end
            local_acc += exp(value - maxval)
        end
        # sync_threads()
        if local_id <= epi_size
            @inbounds begin
                pix2r = smem[local_id+1]
                pix2g = smem[local_id+smemsize+1]
                pix2b = smem[local_id+2smemsize+1]
                muval = smem[local_id+3smemsize+1]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end
            local_acc += exp(value - maxval)
        end
        local_acc = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = log(local_acc) + maxval
            end
        end
        tid_x += nwarps
    end
    return
end
function warp_logsumexp_spp_ct_opt!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = T(0.5) * st / W∞
    invreg = one(T) / reg
    Ntiles = (N) ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img1[1, tid_x]
        pix1g = img1[2, tid_x]
        pix1b = img1[3, tid_x]
        maxval = -Inf

        for tile in 0:Ntiles-1
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = 2μ⁺[j+local_id] - 1
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end
            maxval = max(value, maxval)
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = 2μ⁺[j+local_id] - 1
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end
            maxval = max(value, maxval)
        end
        maxval = CUDA.reduce_warp(max, maxval)
        maxval = CUDA.shfl_sync(CUDA.FULL_MASK, maxval, 1)

        local_acc = 0.0
        for tile in 0:Ntiles-1
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = 2μ⁺[j+local_id] - 1
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end

            local_acc += exp(value - maxval)
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = 2μ⁺[j+local_id] - 1
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                value = -(muladd(l2dist, c1, muval)) * invreg
            end

            local_acc += exp(value - maxval)
        end
        local_acc = CUDA.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = log(local_acc) + maxval
        end
        tid_x += nwarps
    end
    return
end


function extragradient_color_transfer(img1::CuArray{T}, img2::CuArray{T}, args::EOTArgs, frequency::Int=100) where T<:Real
    N = size(img1, 2)
    μ⁺ = T(0.5) * CUDA.ones(T, N)
    μ⁻ = copy(μ⁺)
    μt⁺ = copy(μ⁺)
    μt⁻ = copy(μ⁺)
    ν⁺ = copy(μ⁺)
    ν⁻ = copy(μ⁺)
    νt⁺ = copy(μ⁺)
    νt⁻ = copy(μ⁺)
    residual_cache = CUDA.zeros(T, N)
    sumvals = CUDA.zeros(T, N)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)
    @cuda threads = threads blocks = warp_blocks max_logsumexp_spp_ct!(sumvals, img1, img2)
    CUDA.synchronize()
    W∞ = maximum(sumvals)
    η = T(args.eta_p / 2 / W∞)

    eta_mu = CUDA.ones(T, N) / T(1.01 / N)
    B = T(1.0)
    time_start = time_ns()
    st = T(0.0)
    # sleep(5)
    @inline function infeas(μ⁺)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, img1, img2, μ⁺, η, st, W∞)
        CUDA.synchronize()
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st, W∞)
        CUDA.synchronize()
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (i - 1) % frequency == 0
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, img1, img2, ν⁺, η, st, W∞)
            @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, ν⁺, sumvals, η, st, W∞)
            CUDA.synchronize()
            residual_c = sum(abs.(residual_cache .- 1 / N))
            objective = -sum(2μ⁺ .- 1) / N - η * sum(sumvals) / N

            @printf "%.6e,%d,%.14e,%.14e,extragrad_dual_ctransfer\n" elapsed_time i residual_c objective
        end
        # perform the extragradient step
        infeas(ν⁺)
        # copy!(eta_mu, residual_cache)
        # println("update1")
        @cuda threads = threads blocks = linear_blocks update_μ_residual_ct(μt⁺, μt⁻, μ⁺, μ⁻, residual_cache, eta_mu, T(args.eta_mu), B, false)
        # println("update2")
        @cuda threads = threads blocks = linear_blocks update_μ_ct(νt⁺, νt⁻, ν⁺, ν⁻, μ⁺, μ⁻, η)
        st = (1 - η) * st + η
        infeas(νt⁺)
        # copy!(eta_mu, residual_cache)
        @cuda threads = threads blocks = linear_blocks update_μ_residual_ct(μ⁺, μ⁻, μ⁺, μ⁻, residual_cache, eta_mu, T(args.eta_mu), B, true)
        @cuda threads = threads blocks = linear_blocks update_μ_ct(ν⁺, ν⁻, ν⁺, ν⁻, μt⁺, μt⁻, η)

        CUDA.synchronize()

    end
    output_img1 = CUDA.zeros(T, 3, N)
    output_img2 = CUDA.zeros(T, 3, N)

    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, img1, img2, ν⁺, η, st, W∞)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img1, img1, img2, ν⁺, sumvals, η, st, W∞)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct_t!(output_img2, img1, img2, ν⁺, sumvals, η, st, W∞)
    return ν⁺, Array(output_img1), Array(output_img2)

end

function extragradient_color_transfer(f1::String, f2::String, out_f1::String, out_f2::String, resolution::Tuple{Int,Int}, args::EOTArgs, frequency::Int)
    img1, dims1 = load_rgb(f1; cuda=true, resolution=resolution)
    img2, dims2 = load_rgb(f2; cuda=true, resolution=resolution)
    _, img1_new, img2_new = extragradient_color_transfer(img1, img2, args, frequency)
    save_image(out_f1, img1_new, dims1)
    save_image(out_f2, img2_new, dims2)
end

function accelerated_bregman_descent_transfer(
    img1::CuArray{T}, img2::CuArray{T}, _η::T, maxiter::Int=10000, frequency::Int=100, γ::Float64=2.0
) where T

    N = size(img1, 2)
    μ⁺ = 0.5 * CUDA.ones(T, N)
    μ⁻ = 0.5 * CUDA.ones(T, N)
    γ⁺ = 0.5 * CUDA.ones(T, N)
    γ⁻ = 0.5 * CUDA.ones(T, N)
    ν⁺ = 0.5 * CUDA.ones(T, N)
    ν⁻ = 0.5 * CUDA.ones(T, N)
    # νt⁺ = 0.5 * CUDA.ones(T, N)
    # νt⁻ = 0.5 * CUDA.ones(T, N)
    residual_cache = CUDA.zeros(T, N)
    sumvals = CUDA.zeros(T, N)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)
    @cuda threads = threads blocks = warp_blocks max_logsumexp_spp_ct!(sumvals, img1, img2)
    CUDA.synchronize()
    W∞ = maximum(sumvals)
    η = _η / 2 / W∞
    ημ = CUDA.ones(T, N) / (1.01 / N)

    B = 1.0
    st = 0.0
    @inline function infeas(μ⁺)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, μ⁺, η, st, W∞)
        CUDA.synchronize()
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st, W∞)
        CUDA.synchronize()
    end
    function dual_value(μ⁺, μ⁻)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, μ⁺, η, st, W∞)
        CUDA.synchronize()
        return sum(μ⁺ - (1. .- μ⁺)) / N + η * sum(sumvals) / N
    end
    function bregman(μ⁺, μ⁻, ν⁺, ν⁻)
        return dot(μ⁺ ./ ημ, log.(μ⁺ ./ ν⁺)) + dot((1. .- μ⁺) ./ ημ, log.((1. .- μ⁺) ./ (1. .- ν⁺)))
    end
    function linearization(μ⁺, μ⁻, ν⁺, ν⁻, ∇dν)
        dual_value(μ⁺, μ⁻) - dual_value(ν⁺, ν⁻) - dot(∇dν, μ⁺ - ν⁺ - (μ⁻ - ν⁻))
    end
    ρ = 2.
    B = 1.0
    Gmin = exp(-B) / 10
    G = 1
    function ΔBProject!(μ⁺, μ⁻)
        normv = (μ⁺ + μ⁻)

        μ⁻ .= μ⁻ ./ normv
        μ⁺ .= μ⁺ ./ normv
        μ⁻adjust = max.(μ⁻, exp(-B) .* max.(μ⁺, μ⁻))
        μ⁺adjust = max.(μ⁺, exp(-B) .* max.(μ⁺, μ⁻))
        normv = (μ⁺adjust + μ⁻adjust)

        μ⁻ .= μ⁻adjust ./ normv
        μ⁺ .= μ⁺adjust ./ normv
    end
    θ = 1
    for k in 1:maxiter
        Mₖ = max(G / ρ, Gmin)
        Gₖ = Mₖ
        niter_inner = 0
        st = 1.0
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
            infeas(γ⁺)
            ν⁺_new = ν⁺ .* exp.(η ./ (θₖ^(γ - 1) * Gₖ) .* (residual_cache))
            ν⁻_new = ν⁻ .* exp.(-η ./ (θₖ^(γ - 1) * Gₖ) .* (residual_cache))
            ΔBProject!(ν⁺_new, ν⁻_new)
            μ⁺_new = (1 - θₖ) * μ⁺ + θₖ * ν⁺_new
            μ⁻_new = (1 - θₖ) * μ⁻ + θₖ * ν⁻_new
            if linearization(μ⁺_new, μ⁻_new, γ⁺, γ⁻, residual_cache) <= θₖ^γ * Gₖ ./ η * bregman(ν⁺_new, ν⁻_new, ν⁺, ν⁻)
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
        if (k - 1) % frequency == 0
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, μ⁺, η, st, W∞)
            CUDA.synchronize()
            @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st, W∞)
            CUDA.synchronize()
            @printf "%d,%.14e,%.14e,-1,dual_extrap\n" k sum(residual_cache) dual_value(μ⁺, μ⁻)
        end
        if θ <= 1e-20
            break
        end
        G = Gₖ
    end
    output_img1 = CUDA.zeros(T, 3, N)
    output_img2 = CUDA.zeros(T, 3, N)

    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img1, img2, μ⁺, η, st, W∞)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img1, img1, img2, μ⁺, sumvals, η, st, W∞)
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct!(sumvals, img2, img1, μ⁺, η, st, W∞)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img2, img2, img1, μ⁺, sumvals, η, st, W∞)
    return ν⁺, Array(output_img1), Array(output_img2)
end
function accelerated_bregman_descent_transfer(f1::String, f2::String, out_f1::String, out_f2::String, η::Float64, resolution::Tuple{Int,Int}, maxiter::Int, frequency::Int)
    img1, dims1 = load_rgb(f1; cuda=true, resolution=resolution)
    img2, dims2 = load_rgb(f2; cuda=true, resolution=resolution)
    _, img1_new, img2_new = accelerated_bregman_descent_transfer(img1, img2, η, maxiter, frequency)
    save_image(out_f1, img1_new, dims1)
    save_image(out_f2, img2_new, dims2)
end