function extragradient_cell_distance(
    feature_ids1::AbstractVector{<:Integer},
    feature_ids2::AbstractVector{<:Integer},
    marginal1::AbstractVector{T},
    marginal2::AbstractVector{T},
    data::CuArray{T,2},
    row_sums::CuArray{T,1},
    row_sqnorms::CuArray{T,1},
    row_means::CuArray{T,1},
    row_centered_sqnorms::CuArray{T,1},
    scale::T,
    metric::Symbol,
    args::EOTArgs,
    frequency::Int,
) where T<:Real
    length(feature_ids1) == size(data, 1) || throw(ArgumentError("feature_ids1 length must match the number of features"))
    length(feature_ids2) == size(data, 1) || throw(ArgumentError("feature_ids2 length must match the number of features"))
    marginal1_gpu = CuArray(marginal1)
    marginal2_gpu = CuArray(marginal2)
    _, _, _, objective = extragradient_cellsim(data, scale, metric, marginal1_gpu, marginal2_gpu, args, frequency)
    return T(objective)
end

function sinkhorn_cell_distance(
    feature_ids1::AbstractVector{<:Integer},
    feature_ids2::AbstractVector{<:Integer},
    marginal1::AbstractVector{T},
    marginal2::AbstractVector{T},
    data::CuArray{T,2},
    row_sums::CuArray{T,1},
    row_sqnorms::CuArray{T,1},
    row_means::CuArray{T,1},
    row_centered_sqnorms::CuArray{T,1},
    scale::T,
    metric::Symbol,
    args::EOTArgs,
    frequency::Int,
) where T<:Real
    length(feature_ids1) == size(data, 1) || throw(ArgumentError("feature_ids1 length must match the number of features"))
    length(feature_ids2) == size(data, 1) || throw(ArgumentError("feature_ids2 length must match the number of features"))
    marginal1_gpu = CuArray(marginal1)
    marginal2_gpu = CuArray(marginal2)
    _, _, objective = sinkhorn_cellsim(data, data, scale, metric, marginal1_gpu, marginal2_gpu, args, frequency)
    return T(objective)
end

function maxheap_sift_up!(heap::Vector{T}, idx::Int) where T
    i = idx
    while i > 1
        parent = i >>> 1
        heap[parent] >= heap[i] && break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    end
    return heap
end

function maxheap_sift_down!(heap::Vector{T}, idx::Int) where T
    i = idx
    n = length(heap)
    while true
        left = i << 1
        right = left + 1
        largest = i
        left <= n && heap[left] > heap[largest] && (largest = left)
        right <= n && heap[right] > heap[largest] && (largest = right)
        largest == i && break
        heap[i], heap[largest] = heap[largest], heap[i]
        i = largest
    end
    return heap
end

function minheap_sift_up!(heap::Vector{T}, idx::Int) where T
    i = idx
    while i > 1
        parent = i >>> 1
        heap[parent] <= heap[i] && break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    end
    return heap
end

function minheap_sift_down!(heap::Vector{T}, idx::Int) where T
    i = idx
    n = length(heap)
    while true
        left = i << 1
        right = left + 1
        smallest = i
        left <= n && heap[left] < heap[smallest] && (smallest = left)
        right <= n && heap[right] < heap[smallest] && (smallest = right)
        smallest == i && break
        heap[i], heap[smallest] = heap[smallest], heap[i]
        i = smallest
    end
    return heap
end

function push_smallest!(heap::Vector{T}, current_sum::T, value::T, keep::Int) where T
    keep == 0 && return current_sum
    if length(heap) < keep
        push!(heap, value)
        maxheap_sift_up!(heap, length(heap))
        return current_sum + value
    end
    if value < heap[1]
        new_sum = current_sum + value - heap[1]
        heap[1] = value
        maxheap_sift_down!(heap, 1)
        return new_sum
    end
    return current_sum
end

function push_largest!(heap::Vector{T}, current_sum::T, value::T, keep::Int) where T
    keep == 0 && return current_sum
    if length(heap) < keep
        push!(heap, value)
        minheap_sift_up!(heap, length(heap))
        return current_sum + value
    end
    if value > heap[1]
        new_sum = current_sum + value - heap[1]
        heap[1] = value
        minheap_sift_down!(heap, 1)
        return new_sum
    end
    return current_sum
end

function within_cluster_pairs(clusters::AbstractVector)
    counts = Dict{eltype(clusters), Int}()
    for cluster in clusters
        counts[cluster] = get(counts, cluster, 0) + 1
    end
    return sum(count * (count - 1) ÷ 2 for count in values(counts))
end

function streaming_c_index(
    cell_data::OTScOmicsCellData{T},
    kernel::CellCostKernel{T},
    args::EOTArgs;
    frequency::Int=100,
    solver::Function=extragradient_cell_distance,
) where T<:Real
    ncells = length(cell_data.cell_names)
    nfeatures = size(cell_data.data, 1)
    feature_ids = feature_support(kernel)
    column_sums = vec(sum(cell_data.data; dims=1))
    data_gpu = CuArray(kernel.data)
    row_sums_gpu = CuArray(kernel.row_sums)
    row_sqnorms_gpu = CuArray(kernel.row_sqnorms)
    row_means_gpu = CuArray(kernel.row_means)
    row_centered_sqnorms_gpu = CuArray(kernel.row_centered_sqnorms)
    marginal_i = zeros(T, nfeatures)
    marginal_j = zeros(T, nfeatures)

    nw = within_cluster_pairs(cell_data.clusters)
    smallest_heap = T[]
    largest_heap = T[]
    smin = zero(T)
    smax = zero(T)
    sw = zero(T)

    npairs = ncells * (ncells - 1) ÷ 2
    processed = 0
    println(size(feature_ids), " ", ncells)
    for i in 1:ncells
        fill_normalized_cell!(marginal_i, cell_data.data, i, column_sums)
        for j in 1:(i-1)
            fill_normalized_cell!(marginal_j, cell_data.data, j, column_sums)
            marginal_i .+= 1e-7
            marginal_j .+= 1e-7
            normalize!(marginal_i, 1)
            normalize!(marginal_j, 1)
            distance = solver(
                feature_ids,
                feature_ids,
                marginal_i,
                marginal_j,
                data_gpu,
                row_sums_gpu,
                row_sqnorms_gpu,
                row_means_gpu,
                row_centered_sqnorms_gpu,
                kernel.scale,
                kernel.metric,
                args,
                frequency,
            )
            cell_data.clusters[i] == cell_data.clusters[j] && (sw += distance)
            smin = push_smallest!(smallest_heap, smin, distance, nw)
            smax = push_largest!(largest_heap, smax, distance, nw)
            processed += 1
            if frequency > 0 && processed % frequency == 0
                println("processed_pairs=$(processed)/$(npairs)")
            end
        end
    end

    denominator = smax - smin
    denominator > zero(T) || throw(ArgumentError("Degenerate C-index denominator; pairwise distances do not separate within/between cluster pairs"))

    return (
        c_index = (sw - smin) / denominator,
        Sw = sw,
        Smin = smin,
        Smax = smax,
        Nw = nw,
        pairs = npairs,
    )
end

function compute_otscomics_c_index(
    fpath::String,
    args::EOTArgs;
    metric::AbstractString="correlation",
    normalize_features::Bool=true,
    frequency::Int=100,
    solver::Function=extragradient_cell_distance,
)
    cell_data = read_otscomics_cell_data(fpath)
    kernel = CellCostKernel(cell_data.data, metric; normalize_features=normalize_features)
    return streaming_c_index(cell_data, kernel, args; frequency=frequency, solver=solver)
end
