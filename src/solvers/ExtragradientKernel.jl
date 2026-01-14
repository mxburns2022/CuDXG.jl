using CUDA
using IterTools
using LinearAlgebra



function residual_spp_c!(output::CuDeviceVector{T}, cost_output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, marginal1::CuDeviceVector{T}, θ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T, p::Float64) where T
    step = warpsize()

    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    M = size(img2, 2)
    N = size(img1, 2)
    N_outer = Int(ceil(M / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = st / 2W∞
    invreg = one(T) / (reg / 2W∞)
    Ntiles = (N) ÷ step
    for _ in 1:N_outer
        local_acc = 0.0
        cost_acc = 0.0
        if tid_x > M
            continue
        end
        @inbounds begin
            pix1r = img2[1, tid_x]
            pix1g = img2[2, tid_x]
            pix1b = img2[3, tid_x]
            diff = θ[tid_x]
        end
        for tile in 0:Ntiles-1
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img1[1, j+local_id]
                pix2g = img1[2, j+local_id]
                pix2b = img1[3, j+local_id]
                marginalv = marginal1[j+local_id]

                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(muladd(l2dist, c1, diff), -invreg, -logZi[j+local_id])
            end

            local_acc += exp(value) * marginalv
            cost_acc += exp(value) * l2dist * marginalv
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img1[1, j+local_id]
                pix2g = img1[2, j+local_id]
                pix2b = img1[3, j+local_id]
                marginalv = marginal1[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(muladd(l2dist, c1, diff), -invreg, -logZi[j+local_id])
            end

            local_acc += exp(value) * marginalv
            cost_acc += exp(value) * l2dist * marginalv
        end
        local_acc = CUDA.reduce_warp(+, local_acc)
        cost_acc = CUDA.reduce_warp(+, cost_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = local_acc
                cost_output[tid_x] = cost_acc
            end
        end
        tid_x += nwarps
    end
    return
end


function naive_findmaxindex_spp_ct!(output_img::CuDeviceMatrix{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ::CuDeviceVector{T}, marginal1::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T, p::Float64) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    M = size(img2, 2)
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
    ri = marginal1[tid_x]
    for i in 1:M
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        diff = μ[i]
        dr = (pix1r - pix2r)
        dg = (pix1g - pix2g)
        db = (pix1b - pix2b)
        if p == 1
            l2dist = abs(dr) + abs(dg) + abs(db)
        else
            l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
        end
        prob = exp(-(l2dist * st / 2W∞ + diff) / (reg / 2W∞) - norm)
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
    img2::CuDeviceMatrix{T}, marginal1::CuDeviceVector{T}, μ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, W∞::T, p::Float64) where T
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    M = size(img2, 2)
    if tid_x > M
        return
    end
    pix1r = img2[1, tid_x]
    pix1g = img2[2, tid_x]
    pix1b = img2[3, tid_x]
    diff = μ[tid_x]
    avgr = 0.0
    avgg = 0.0
    avgb = 0.0
    probsum = 0.0
    for i in 1:N
        pix2r = img1[1, i]
        pix2g = img1[2, i]
        pix2b = img1[3, i]
        norm = logZi[i]
        ri = marginal1[i]
        dr = (pix1r - pix2r)
        dg = (pix1g - pix2g)
        db = (pix1b - pix2b)
        if p == 1
            l2dist = abs(dr) + abs(dg) + abs(db)
        else
            l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
        end
        prob = exp(-(l2dist * st / 2W∞ + diff) / (reg  / 2W∞) - norm) * ri
        avgr += img1[1, i] * prob
        avgg += img1[2, i] * prob
        avgb += img1[3, i] * prob
        probsum += prob
    end
    output_img[1, tid_x] = avgr / probsum
    output_img[2, tid_x] = avgg / probsum
    output_img[3, tid_x] = avgb / probsum

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


function update_θ_residual_ct(theta::CuDeviceArray{R}, theta_0::CuDeviceArray{R}, marginal2::CuDeviceArray{R}, residual::CuDeviceArray{R}, eta_mu_i::CuDeviceArray{R}, eta_mu::R, adjust::Bool, minv, maxv) where R
    N = size(theta, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    @inbounds begin
        difference = 2(residual[tid] - marginal2[tid]) / eta_mu_i[tid]
        thetav = theta_0[tid]
    end
    expv = exp(difference) * ((thetav + 1) / (1-thetav))^(1 - eta_mu)
    theta_value_new = (expv - 1)/ (expv + 1)
    if adjust
        theta_value_new = clamp(theta_value_new, minv, maxv)#max(min(theta_value_new, maxv), minv)
    end
    @inbounds begin
        theta[tid] = theta_value_new
    end 

    
    return
end
function warp_logsumexp_spp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, μ⁺::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    M = size(img2, 2)
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
        for i in 1:step:M
            if i + local_id > M
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]

            l2dist = abs(pix1r - pix2r) + abs(pix1g - pix2g) + abs(pix1b - pix2b)
            # l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
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
        for i in 1:step:M
            if i + local_id > M
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]

            l2dist = abs(pix1r - pix2r) + abs(pix1g - pix2g) + abs(pix1b - pix2b)
            # l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
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
function max_logsumexp_spp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T}, img2::CuDeviceMatrix{T}, p::Float64) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    M = size(img2, 2)
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
        for i in 1:step:M
            if i + local_id > M
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            if p == 1
                l2dist = abs(pix1r - pix2r) + abs(pix1g - pix2g) + abs(pix1b - pix2b)
            else
                l2dist = (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
            end
            # l2dist = abs(pix1r - pix2r) + abs(pix1g - pix2g) + abs(pix1b - pix2b)

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
    M = size(img2, 2)

    smem = CuStaticSharedArray(T, 4 * smemsize)
    warpiters = smemsize ÷ warpsize()
    Ntiles = (M) ÷ smemsize
    warpiter_epi = (M - Ntiles * smemsize) ÷ warpsize()
    epi_size = M - (Ntiles * smemsize + warpiter_epi * warpsize())
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
                    l2dist = abs(dr) + abs(dg) + abs(db)
                    # l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                    value = -(muladd(l2dist, c1, muval)) * invreg
                    # end
                    maxval = max(maxval, value)
                end
                # break
            end
            # break
        end
        if (Ntiles) * smemsize + threadIdx().x <= M
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
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                    l2dist = abs(dr) + abs(dg) + abs(db)
                    # l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
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
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
    img2::CuDeviceMatrix{T}, θ::CuDeviceVector{T}, reg::T, st::T, W∞::T, p::R) where {T, R}
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = st / 2W∞
    invreg = one(T) / (reg / 2W∞)
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
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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
                muval = θ[j+local_id] 
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
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



function warp_logsumexp_spp_ct_fused!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, θ::CuDeviceVector{T}, reg::T, st::T, W∞::T, p::Float64) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    M = size(img2, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 =  st / 2W∞
    invreg = one(T) / (reg / 2W∞)
    Ntiles = (M) ÷ step

    for _ in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img1[1, tid_x]
        pix1g = img1[2, tid_x]
        pix1b = img1[3, tid_x]
        m_local = T(-Inf)
        s_local = T(0)
        
        for tile in 0:Ntiles-1
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
            end
            v = -(muladd(l2dist, c1, muval)) * invreg
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end
        if (Ntiles) * warpsize() + local_id < M
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
            end
            v = -(muladd(l2dist, c1, muval)) * invreg
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


function extragradient_color_transfer(img1::CuArray{T}, img2::CuArray{T}, marginal1::CuArray{T}, marginal2::CuArray{T}, args::EOTArgs, frequency::Int=100, normalize_cost::Bool=false, p::Float64=2.0) where T<:Real
    N = size(img1, 2)
    M = size(img2, 2)
    θ = CUDA.zeros(T, M)
    ν = copy(θ)
    θ̄ = copy(θ)
    ν̄ = copy(θ)
    residual_cache = CUDA.zeros(T, M)
    sumvals = CUDA.zeros(T, N)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)
    @cuda threads = threads blocks = warp_blocks max_logsumexp_spp_ct!(sumvals, img1, img2, p)
    CUDA.synchronize()
    W∞ = maximum(sumvals)
    η = T(args.eta_p / 2)

    eta_mu = (marginal2 .+ T(args.C3) / N) / args.C2
    time_start = time_ns()
    st = T(0.0)
    # sleep(5)
    cost_cache = CUDA.zeros(T, N)
    @inline function infeas(θt, ηt_value, s_value)
        # Sanity check
        # W = (img1[1, :] .- img2[1, :]').^2 + (img1[2, :] .- img2[2, :]').^2 + (img1[3, :] .- img2[3, :]').^2
        # maxvals = maximum(-(W .* s_value./2W∞ .+ θt') ./ (ηt_value / 2W∞), dims=2)
        # sumv = log.(sum(exp.(-(W .* s_value/2W∞ .+ θt') / (ηt_value / 2W∞) .- maxvals), dims=2)) .+ maxvals
        # marginal_v = sum((marginal1.*softmax(-(W .* s_value .+ 2W∞*θt') / (ηt_value) , norm_dims=2))', dims=2)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_fused!(sumvals, img1, img2, θt, ηt_value, s_value, W∞, p)
        CUDA.synchronize()
        # println()
        # println()
        # println(sumvals, " ", p)
        # println(sumv)
        # sleep(2)
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, cost_cache, img1, img2, marginal1, θt, sumvals, ηt_value, s_value, W∞, p)
        CUDA.synchronize()
        # println()
        # println()
        # println(sumv)
        # println(sumvals)
        # println()
        # println((residual_cache))
        # println((marginal_v))
        # sleep(2)
    end
    hr = sum(neg_entropy(marginal1))
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    ηt = if args.eta_p == 0
        1.0
    else
        η
    end
    W∞_scaling = (W∞)^p
    img1 ./= (W∞_scaling)^(1/p)
    img2 ./= (W∞_scaling)^(1/p)
    W∞ = W∞/W∞_scaling#1.0
    st = if args.eta_p == 0
        1.0
    else
        0.0
    end
    minv = tanh(-args.B/2)
    maxv = tanh(args.B/2)
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        # println(ν, " ", st, " ", ηt)
        # sleep(1)
        if args.verbose && (i - 1) % frequency == 0
            infeas(ν, ηt, st)
            CUDA.synchronize()
            primal_value = ηt * dot(marginal1, sumvals) + dot(marginal2, ν) + hr
            residual_value = sum(abs.(residual_cache - marginal2))
            objective = W∞_scaling*sum(cost_cache)
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_fused!(sumvals, img1, img2, θ, ηt, 1.0, W∞, p)

            dual_value = ηt * dot(marginal1, sumvals) + dot(marginal2, θ) + hr
            # @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, cost_cache, img1, img2, marginal1, μ⁺, sumvals, η, 1.0, W∞)
            CUDA.synchronize()
            # objective_dual = sum(cost_cache) / W∞

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,extragrad_dual_ctransfer\n" elapsed_time i residual_value objective primal_value dual_value
            if primal_value - dual_value < args.epsilon / 6 && residual_value < args.epsilon / 6
                break
            end
        end
        # perform the extragradient step
        infeas(ν, ηt, st)
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ̄, θ, residual_cache, marginal2, eta_mu, T(args.eta_mu), false, minv, maxv, 1/2W∞)
        # println(θ̄)
        # println(residual_cache-marginal2)
        # sleep(1)
        ηt = ηt / ( 1 + (ηt - η) )
        ν̄ .= (1-ηt) * ν + ηt * θ

        CUDA.synchronize()
        st = (1 - ηt) * st + ηt
        infeas(ν̄, ηt, st)
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ, θ, residual_cache, marginal2, eta_mu, T(args.eta_mu), true, minv, maxv, 1/2W∞)
        ν .= (1-ηt) * ν + ηt * θ̄

        CUDA.synchronize()


    end
    output_img1 = CUDA.zeros(T, 3, N)
    output_img2 = CUDA.zeros(T, 3, N)

    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, img1, img2, ν, ηt, st, W∞, p)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct!(output_img1, img1, img2, ν, marginal1, sumvals, ηt, st, W∞, p)
    @cuda threads = threads blocks = linear_blocks naive_findmaxindex_spp_ct_t!(output_img2, img1, img2, marginal1, ν, sumvals, ηt, st, W∞, p)
    # recover the dual potentials
    ψ = -2W∞ * ν ./ args.eta_p
    φ = log.(marginal1) - sumvals
    return Array(ν), Array(φ), Array(ψ), Array(output_img1), Array(output_img2)

end

function extragradient_color_transfer(f1::String, f2::String, out_f1::String, out_f2::String, resolution::Tuple{Int,Int}, args::EOTArgs, frequency::Int, p::Float64)
    img1, dims1, marginal1 = load_rgb(f1; cuda=true, resolution=resolution)
    img2, dims2, marginal2 = load_rgb(f2; cuda=true, resolution=resolution)
    mu1, phi, psi, img1_new, img2_new = extragradient_color_transfer(img1, img2, marginal1, marginal2, args, frequency, false, p)
    save_image(out_f1, img1_new, dims1)
    save_image(out_f2, img2_new, dims2)
end


function extragradient_euclidean(marginal1::CuArray{T}, marginal2::CuArray{T}, location1::CuArray{T}, location2::CuArray{T}, out1::String, out2::String, outmu::String, args::EOTArgs, frequency::Int, p::Float64) where T
    μ, φ, ψ, assignments1, assignments2 = extragradient_color_transfer(location1, location2, marginal1, marginal2, args, frequency, p)
    if outmu != ""
        open(outmu * ".spp_col", "w") do outfile
            for μi in μ
                println(outfile, μi)
            end
        end
        open(outmu * ".row", "w") do outfile
            # println(outfile, "")
            for φi in φ
                println(outfile, φi)
            end
        end
        open(outmu * ".col", "w") do outfile
            # println(outfile, "")
            for ψi in ψ
                println(outfile, ψi)
            end
        end
    end
    if out1 != ""
        open(out1, "w") do outfile
            for index in eachindex(assignments1)
                println(outfile, "$(assignments1),$(index)")
            end
        end
    end
    if out2 != ""
        open(out2, "w") do outfile
            for index in eachindex(assignments2)
                println(outfile, "$(assignments2),$(index)")
            end
        end
    end
end


function accelerated_bregman_descent_transfer(
    img1::CuArray{T}, img2::CuArray{T}, _η::T, maxiter::Int=10000, frequency::Int=100, γ::Float64=2.0, p::Float64=2
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
        @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st, W∞, p)
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
            @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(sumvals, img1, img2, μ⁺, η, st, W∞, p)
            CUDA.synchronize()
            @cuda threads = threads blocks = warp_blocks residual_spp_c!(residual_cache, img1, img2, μ⁺, sumvals, η, st, W∞, p)
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