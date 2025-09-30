include("src/Utilities.jl")
include("src/solvers/Extragradient.jl")
include("src/solvers/Sinkhorn.jl")
include("src/solvers/APDAGD.jl")
include("src/solvers/APDAMD.jl")
include("src/solvers/AccSinkhorn.jl")
include("src/solvers/Greenkhorn.jl")
using CUDA
using ReverseDiff
using Test
using Random
marginal1, h, w = read_dotmark_data("/home/matt/Documents/bench/optimization/DOTmark_1.0/Data/ClassicImages/data32_1001.csv")
marginal2, h, w = read_dotmark_data("/home/matt/Documents/bench/optimization/DOTmark_1.0/Data/ClassicImages/data32_1004.csv")
r = normalize(CuArray(marginal1) + 1e-5 * CUDA.ones(h * w), 1)
c = normalize(CuArray(marginal2) + 1e-5 * CUDA.ones(h * w), 1)
W = normalize(CuArray(get_euclidean_distance(h, w)), Inf)
rng = Xoshiro(0)
# N = 8
# r = normalize(rand(rng, N), 1)
# c = normalize(rand(rng, N), 1)
# W = rand(rng, N, N)
# W /= norm(W, Inf)
# r, c, W = map(CuArray, [r, c, W])
args = EOTArgs(B=1.0, eta_p=1e-2, eta_mu=0.0, tmax=1000, verbose=true, epsilon=1e-10)

function test_gradient()
    N = 5
    r = normalize(rand(rng, N), 1)
    c = normalize(rand(rng, N), 1)
    ϕ = rand(N)
    ψ = rand(N)
    W = rand(N, N)
    args = EOTArgs(B=1.0, eta_p=1e-5, eta_mu=0.0, tmax=50000, verbose=true)
    prob = EOTProblem(η=args.eta_p, r=r, c=c, W=W)
    p = softmax(-W ./ prob.η)
    state = APDAMDState(p=p, λ=vcat(ϕ, ψ), L=1 / prob.η)
    grad_ad = ReverseDiff.gradient(x -> φ(x, prob), state.λ)
    dual_gradient!(state.grad_cache, state.λ, prob)
    # println()
    @test norm(state.grad_cache - grad_ad) ≈ 0.0 atol = 1e-9
end
function test_formulation()
    rng = Xoshiro(1)
    a = rand(rng, 5)
    b = rand(rng, 5)
    W = rand(rng, 5, 5)
    r = softmax(W .+ a .+ b', dims=2; normalize_values=false)
    c = softmax((W .+ a .+ b')', dims=2; normalize_values=false)

end
