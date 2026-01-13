using PythonOT
include("src/Utilities.jl")
include("src/solvers/Extragradient.jl")
using LinearAlgebra
using Random
using CUDA

# function r(x,y,A,maxA)
#   term1 = x'*A*(y.^2)
#   term2 = 10 * maxA * x'*log.(x + 1e-20)
#   return term1 + term2
# end


# function g(x,y,A,b,c)
#   return [A*y + c; -A'*x + b]
# end


function ACMirrorProxDual(C, r, c, args::EOTArgs)
  n = size(r, 1)
  Kiters = 60 * log(2.0 / args.epsilon)
  if isa(C, CuArray)
    μ  = CUDA.zeros(n)
    ν = CUDA.zeros(2n)
    ν̃ = CUDA.zeros(2n)
    d = CUDA.zeros(2n)
    yt = CUDA.zeros(2n)
    ut = CUDA.zeros(2n)
    vt = CUDA.zeros(2n)
    dt = CUDA.zeros(2n)
    p̄ = CUDA.zeros(n, n)
  else
    y = zeros(2n)
    u = zeros(2n)
    v = zeros(2n)
    d = zeros(2n)
    yt = zeros(2n)
    ut = zeros(2n)
    vt = zeros(2n)
    dt = zeros(2n)
    p̄ = zeros(n, n)
  end
  b = [r; c]
  λ = 0
  λt = 0
  function getp(u, v, λt)
    return softmax(λt .* C .+ (v[n+1:end] + u[n+1:end])' .+ v[1:n] .+ u[1:n])
  end
  
end


n = 10
N = n * n
m = 10
M = m * m
# N = 5
# M = 5
M = 500
N = 500
rng = Xoshiro(0)
r, c, W, optimum = generate_random_ot(N, M, rng)
c .+= 1e-6
r .+= 1e-6
normalize!(r, 1)
normalize!(c, 1)
fill!(c, 0.0)
c[1] = 1.0
c .+= 1e-5
normalize!(c, 1)
# fill!(r, 0.0)
# r[2] = 1.0
# r .+= 1e-6
# normalize!(, 1)
# W = get_euclidean_distance(n, n, p=1.)
# W /= norm(W, Inf)
r, c, W = map(CuArray, [r, c, W])
args = EOTArgs(itermax=100000, epsilon=1e-6, eta_p=1.)
# ACMirrorProxDual(W, r, c, args)
args = EOTArgs(itermax=2, epsilon=1e-8, eta_p=1e-5, eta_mu=1e-10, B=1.1)
# prototype_extragradient_ot2(r, c, W, args)
W /= norm(W, Inf)
# extragradient_ot_dual(r, c, W, args)
args = EOTArgs(itermax=10000, epsilon=1e-8, eta_p=0.0, eta_mu=0.0, B=2.0)
# prototype_extragradient_ot(r, c, W, args)
# prototype_extragradient_ot(r, c, W, args)
# papc_ot_dual(r, c, W, args)
# extragradient_ot_dual(r, c, W, args)