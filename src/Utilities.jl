import CSV
using DataFrames
using JSON3
using IterTools
using StructTypes
using ArgParse

@kwdef struct EOTProblem{TA,TM,R}
    η::R
    r::TA
    c::TA
    W::TM
    b::TA = vcat(r, c)
    N = size(r, 1)
end
@kwdef mutable struct EOTArgs{R}
    eta_p::R = 0.0
    eta_mu::R = 0.0
    C1::R = 1.0
    C2::R = 1.0
    C3::R = 1e-3
    B::R = 1.0
    epsilon::R = 1e-4
    itermax::Int = 10_000
    tmax::Float64 = Inf
    verbose::Bool = true
end
StructTypes.StructType(::Type{EOTArgs}) = StructTypes.Mutable()

function logsumexp(x::AbstractArray{T}, dims=Nothing) where T<:Number
    if dims == Nothing
        dims = 1:length(size(x))
    end
    maxx = maximum(x, dims=dims)
    v1 = log.(sum(exp.(x .- maxx), dims=dims)) .+ maxx
    return v1
end
function logsumexp!(out::AbstractArray{T}, maxcache::AbstractArray{T}, x::AbstractArray{T}, dims=Nothing) where T<:Number
    maximum!(maxcache, x)
    sum!(out, exp.(x .- maxcache))
    out .= log.(out) .+ maxcache
end
function Zvals(x::AbstractArray{T}; dims=[]) where T
    maxx = maximum(x, dims=dims)
    return sum(exp.(x .- maxx), dims=dims)
end

@inline
function dual_gradient!(output::TA, x::TA, prob::EOTProblem) where TA
    grad_cache1 = sofitermax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]', dims=2)
    grad_cache2 = sofitermax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]', dims=1)'
    output .= vcat(grad_cache1, grad_cache2)
    output .-= prob.b
end
# function f(p, prob::EOTProblem)
#     return dot(p, prob.W)
# end
function φ(x, prob::EOTProblem)
    return sum(logsumexp(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]')) - dot(prob.b, x)
end


function get_p(x, prob::EOTProblem)
    return sofitermax(-prob.W / prob.η .+ x[1:prob.N] .+ x[prob.N+1:end]')
end

function read_args_json(fpath::String)
    json_string = read(fpath, String)
    settings = JSON3.read(json_string, EOTArgs)
    return settings
end
function sofitermax(x::AbstractArray{T}; normalize_values=true, dims=[], norm_dims=Nothing) where T<:Real
    if norm_dims == Nothing
        norm_dims = [1:ndims(x)...]
    end

    if !normalize_values
        return sum(exp.(x), dims=dims)
    else
        maxx = maximum(x, dims=norm_dims)
        v1 = sum(exp.(x .- maxx), dims=dims)
        return v1 ./ sum(v1, dims=norm_dims)
    end
end

function neg_entropy(x::TA; dims=[]) where TA
    return sum(map(y -> if y > 1e-30
                y * log(y)
            else
                0.0
            end, x), dims=dims)
end

function get_euclidean_distance(height::Int, width::Int)
    N = height * width
    W = zeros(N, N)
    for (i, j) in product(0:(N-1), 0:(N-1))
        W[i+1, j+1] = (i ÷ height - j ÷ height)^2 + (i % height - j % height)^2
    end
    return W
end
function round(γ::AbstractMatrix{T}, μ::AbstractArray{T}, ν::AbstractArray{T}) where T<:Real
    γ⁺ = γ .* min.(μ ./ (sum(γ, dims=2)), 1.0)
    γ⁺⁺ = γ⁺ .* min.(ν ./ (sum(γ⁺, dims=1))', 1.0)'
    rμ = μ - sum(γ⁺⁺, dims=2)
    rν = ν - sum(γ⁺⁺, dims=1)'
    γ̂ = γ⁺⁺ + rμ * rν' / norm(rμ, 1)
    return γ̂
end
function read_dotmark_data(fpath::String)
    input_data = Matrix(CSV.read(fpath, header=false, DataFrame))
    h, w = size(input_data)
    N = h * w
    marginal = reshape(input_data, N) / sum(input_data)
    return marginal, h, w, N
end



