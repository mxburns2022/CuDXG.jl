using CUDA
using Statistics

const CELL_COST_METRICS = ("l1", "l2", "cosine", "pearson", "correlation")

struct OTScOmicsCellData{T}
    data::Matrix{T}
    feature_names::Vector{String}
    cell_names::Vector{String}
    clusters::Vector{String}
end

struct CellCostKernel{T}
    data::Matrix{T}
    metric::Symbol
    normalize_features::Bool
    row_sums::Vector{T}
    row_sqnorms::Vector{T}
    row_means::Vector{T}
    row_centered_sqnorms::Vector{T}
    scale::T
end

function normalize_cell_cost_metric(metric::AbstractString)
    metric_lc = lowercase(metric)
    metric_lc == "pearson" && return "correlation"
    metric_lc in CELL_COST_METRICS || throw(ArgumentError("Unsupported cell cost metric: $(metric)"))
    return metric_lc
end

function infer_cell_cluster(name::AbstractString)
    matched = match(r"^Cell_\d+_(.+)$", name)
    matched === nothing || return matched.captures[1]
    parts = split(name, "_")
    return isempty(parts) ? String(name) : parts[end]
end

infer_cell_clusters(cell_names::AbstractVector{<:AbstractString}) = [infer_cell_cluster(name) for name in cell_names]

function _read_otscomics_table(fpath::String)
    if endswith(lowercase(fpath), ".gz")
        return open(`gzip -cd $fpath`) do io
            DataFrame(CSV.File(io))
        end
    end
    return CSV.read(fpath, DataFrame)
end

function read_otscomics_cell_data(fpath::String)
    table = _read_otscomics_table(fpath)
    ncol(table) >= 2 || throw(ArgumentError("Expected at least one feature column and one cell column in $(fpath)"))
    feature_names = String.(table[:, 1])
    cell_names = String.(names(table)[2:end])
    data = Float64.(Matrix(table[:, 2:end]))
    clusters = [String(i) for i in infer_cell_clusters(cell_names)]
    return OTScOmicsCellData(data, feature_names, cell_names, clusters)
end

function row_scales(data::AbstractMatrix{T}, normalize_features::Bool) where T<:Real
    scales = vec(sum(data; dims=2))
    if normalize_features
        for i in eachindex(scales)
            scales[i] = scales[i] > zero(T) ? scales[i] : one(T)
        end
    else
        fill!(scales, one(T))
    end
    return scales
end

function _cost_scale(metric::Symbol)
    metric === :l1 && return 2.0
    metric === :l2 && return sqrt(2.0)
    metric === :cosine && return 2.0
    metric === :correlation && return 2.0
    return 1.0
end

function CellCostKernel(data::Matrix{T}, metric::AbstractString="correlation"; normalize_features::Bool=true) where T<:Real
    metric_sym = Symbol(normalize_cell_cost_metric(metric))
    scales = row_scales(data, normalize_features)
    row_sqnorms = vec(sum(abs2, data; dims=2))
    row_means = vec(Statistics.mean(data; dims=2))
    row_centered_sqnorms = Vector{T}(undef, size(data, 1))
    @inbounds for i in axes(data, 1)
        μ = row_means[i]
        centered_sum = zero(T)
        for value in @view data[i, :]
            centered_sum += abs2(value - μ)
        end
        row_centered_sqnorms[i] = centered_sum
    end
    return CellCostKernel(
        data,
        metric_sym,
        normalize_features,
        scales,
        row_sqnorms,
        row_means,
        row_centered_sqnorms,
        T(_cost_scale(metric_sym)),
    )
end

feature_count(kernel::CellCostKernel) = size(kernel.data, 1)
feature_support(kernel::CellCostKernel) = collect(1:feature_count(kernel))

@inline scaled_row_mean(kernel::CellCostKernel{T}, i::Int) where T = kernel.row_means[i] / kernel.row_sums[i]
@inline scaled_row_sqnorm(kernel::CellCostKernel{T}, i::Int) where T = kernel.row_sqnorms[i] / (kernel.row_sums[i]^2)
@inline scaled_row_centered_sqnorm(kernel::CellCostKernel{T}, i::Int) where T = kernel.row_centered_sqnorms[i] / (kernel.row_sums[i]^2)

function raw_row_dot(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    dot_value = zero(T)
    @inbounds @views for k in axes(kernel.data, 2)
        dot_value += kernel.data[i, k] * kernel.data[j, k]
    end
    return dot_value
end

function l1_feature_cost(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    scale_i = kernel.row_sums[i]
    scale_j = kernel.row_sums[j]
    distance = zero(T)
    @inbounds @views for k in axes(kernel.data, 2)
        distance += abs(kernel.data[i, k] / scale_i - kernel.data[j, k] / scale_j)
    end
    return distance / kernel.scale
end

function l2_feature_cost(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    dot_ij = raw_row_dot(kernel, i, j) / (kernel.row_sums[i] * kernel.row_sums[j])
    norm_i = scaled_row_sqnorm(kernel, i)
    norm_j = scaled_row_sqnorm(kernel, j)
    return sqrt(max(norm_i + norm_j - 2dot_ij, zero(T))) / kernel.scale
end

function cosine_feature_cost(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    dot_ij = raw_row_dot(kernel, i, j) / (kernel.row_sums[i] * kernel.row_sums[j])
    denom = sqrt(scaled_row_sqnorm(kernel, i) * scaled_row_sqnorm(kernel, j))
    denom > zero(T) || return zero(T)
    similarity = clamp(dot_ij / denom, -one(T), one(T))
    return (one(T) - similarity) / kernel.scale
end

function correlation_feature_cost(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    n = size(kernel.data, 2)
    dot_ij = raw_row_dot(kernel, i, j) / (kernel.row_sums[i] * kernel.row_sums[j])
    μi = scaled_row_mean(kernel, i)
    μj = scaled_row_mean(kernel, j)
    centered_i = scaled_row_centered_sqnorm(kernel, i)
    centered_j = scaled_row_centered_sqnorm(kernel, j)
    denom = sqrt(centered_i * centered_j)
    if denom <= zero(T)
        return isapprox(μi, μj; atol=sqrt(eps(T))) ? zero(T) : inv(kernel.scale)
    end
    correlation = clamp((dot_ij - n * μi * μj) / denom, -one(T), one(T))
    return (one(T) - correlation) / kernel.scale
end

function feature_cost(kernel::CellCostKernel{T}, i::Int, j::Int) where T
    kernel.metric === :l1 && return l1_feature_cost(kernel, i, j)
    kernel.metric === :l2 && return l2_feature_cost(kernel, i, j)
    kernel.metric === :cosine && return cosine_feature_cost(kernel, i, j)
    kernel.metric === :correlation && return correlation_feature_cost(kernel, i, j)
    throw(ArgumentError("Unsupported kernel metric $(kernel.metric)"))
end

function fill_normalized_cell!(buffer::AbstractVector{T}, data::AbstractMatrix{T}, column::Int, column_sums::AbstractVector{T}) where T<:Real
    scale = column_sums[column]
    if scale <= zero(T)
        fill!(buffer, inv(T(length(buffer))))
        return buffer
    end
    @inbounds @views for i in eachindex(buffer)
        buffer[i] = data[i, column] / scale
    end
    return buffer
end

@inline function scaled_row_mean_cuda(row_means::CuDeviceVector{T}, row_sums::CuDeviceVector{T}, i::Int) where T
    return row_means[i] / row_sums[i]
end

@inline function scaled_row_sqnorm_cuda(row_sqnorms::CuDeviceVector{T}, row_sums::CuDeviceVector{T}, i::Int) where T
    return row_sqnorms[i] / (row_sums[i]^2)
end

@inline function scaled_row_centered_sqnorm_cuda(
    row_centered_sqnorms::CuDeviceVector{T},
    row_sums::CuDeviceVector{T},
    i::Int,
) where T
    return row_centered_sqnorms[i] / (row_sums[i]^2)
end

@inline function raw_row_dot_cuda(data::CuDeviceMatrix{T}, i::Int, j::Int, ncols::Int) where T
    dot_value = zero(T)
    @inbounds for k in 1:ncols
        dot_value += data[i, k] * data[j, k]
    end
    return dot_value
end

@inline function l1_feature_cost_cuda(
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    scale::T,
    i::Int,
    j::Int,
    ncols::Int,
) where T
    scale_i = row_sums[i]
    scale_j = row_sums[j]
    distance = zero(T)
    @inbounds for k in 1:ncols
        distance += abs(data[i, k] / scale_i - data[j, k] / scale_j)
    end
    return distance / scale
end



@inline function l2_feature_cost_cuda(
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    scale::T,
    i::Int,
    j::Int,
    ncols::Int,
) where T
    dot_ij = raw_row_dot_cuda(data, i, j, ncols) / (row_sums[i] * row_sums[j])
    norm_i = scaled_row_sqnorm_cuda(row_sqnorms, row_sums, i)
    norm_j = scaled_row_sqnorm_cuda(row_sqnorms, row_sums, j)
    return sqrt(max(norm_i + norm_j - 2dot_ij, zero(T))) / scale
end

@inline function cosine_feature_cost_cuda(
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    scale::T,
    i::Int,
    j::Int,
    ncols::Int,
) where T
    dot_ij = raw_row_dot_cuda(data, i, j, ncols) / (row_sums[i] * row_sums[j])
    denom = sqrt(scaled_row_sqnorm_cuda(row_sqnorms, row_sums, i) * scaled_row_sqnorm_cuda(row_sqnorms, row_sums, j))
    denom > zero(T) || return zero(T)
    similarity = clamp(dot_ij / denom, -one(T), one(T))
    return (one(T) - similarity) / scale
end

@inline function correlation_feature_cost_cuda(
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    i::Int,
    j::Int,
    ncols::Int,
) where T
    dot_ij = raw_row_dot_cuda(data, i, j, ncols) / (row_sums[i] * row_sums[j])
    μi = scaled_row_mean_cuda(row_means, row_sums, i)
    μj = scaled_row_mean_cuda(row_means, row_sums, j)
    centered_i = scaled_row_centered_sqnorm_cuda(row_centered_sqnorms, row_sums, i)
    centered_j = scaled_row_centered_sqnorm_cuda(row_centered_sqnorms, row_sums, j)
    denom = sqrt(centered_i * centered_j)
    if denom <= zero(T)
        return abs(μi - μj) <= sqrt(eps(T)) ? zero(T) : inv(scale)
    end
    correlation = clamp((dot_ij - ncols * μi * μj) / denom, -one(T), one(T))
    return (one(T) - correlation) / scale
end

@inline function feature_cost_cuda(
    metric::Symbol,
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    i::Int,
    j::Int,
    ncols::Int,
) where T
    metric === :l1 && return l1_feature_cost_cuda(data, row_sums, scale, i, j, ncols)
    metric === :l2 && return l2_feature_cost_cuda(data, row_sums, row_sqnorms, scale, i, j, ncols)
    metric === :cosine && return cosine_feature_cost_cuda(data, row_sums, row_sqnorms, scale, i, j, ncols)
    metric === :correlation && return correlation_feature_cost_cuda(data, row_sums, row_means, row_centered_sqnorms, scale, i, j, ncols)
    return T(Inf)
end


function get_flat_metrics(M::AbstractMatrix{R}) where {R<:Real}
  N = size(M, 1)
  row_sums = reshape(sum(M, dims=2), N)
  row_sqnorms = reshape(sum(M.^2, dims=2), N)
  row_means = reshape(mean(M, dims=2), N)
  row_centered_sqnorms = reshape(sum((M .- row_means).^2, dims=2), N)
  return row_sums, row_sqnorms, row_means, row_centered_sqnorms
end
