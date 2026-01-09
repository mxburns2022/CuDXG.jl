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
    y = CUDA.zeros(2n)
    u = CUDA.zeros(2n)
    v = CUDA.zeros(2n)
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
  function getb(u, v, λt)
    p = getp(u, v, λt)
    return [sum(p, dims=2); sum(p, dims=1)']
  end
  η = 5.
  for i in 1:args.itermax
    yt .= y
    ut .= u
    bt = getb(u, v, λ)
    gamma_y = 1 / 3 * (b - bt .* (1 .+ 2y))
    λt = -1 / (2 * args.eta_p) * (1 / 3 - 2 * args.eta_p * λ)
    vt = -1 / (2 * args.eta_p) * (1 / 3 * y - 2 * args.eta_p * v)
    for k in 1:Kiters
      ut = -1 / (2 * args.eta_p) * (-y .^ 2 + yt .^ 2 - 2 * args.eta_p * u)
      dt .= 4 * getb(ut, vt, λt)
      yt = clamp.(-gamma_y ./ dt, -1.0, 1.0)
    end

    p = getp(ut, vt, λt)
    axpby!(1 / i, p, (i - 1) / i, p̄)
    gamma_y = 1 / 3 * (b - getb(ut, vt, λt) .- 2y .* bt)
    ut .= u
    λ = -1 / (2 * args.eta_p) * (1 / 3 - 2 * args.eta_p * λ)
    v = -1 / (2 * args.eta_p) * (1 / 3 * yt - 2 * args.eta_p * v)
    yt .= y
    for k in 1:Kiters
      u = -1 / (2 * args.eta_p) * (-yt .^ 2 + y .^ 2 - 2 * args.eta_p * ut)
      d .= 4 * getb(u, v, λ)
      y = clamp.(-gamma_y ./ d, -1.0, 1.0)
    end
    if (i - 1) % 10 == 0
      pv = round(p̄, r, c)
      println(i, " ", dot(pv, C), " ", norm(sum(pv, dims=1)' - c, 1))
    end
  end
end


n = 50
N = n * n
m = 20
M = m * m

rng = Xoshiro(0)
r, c, W, optimum = generate_random_ot(N, N, rng)
fill!(c, 0.0)
c[1] = 1.0
c .+= 1e-1
normalize!(c, 1)
# fill!(r, 0.0)
# r[2] = 1.0
# r .+= 1e-6
# normalize!(, 1)
W = get_euclidean_distance(n, n, p=1.)
# W /= norm(W, Inf)
r, c, W = map(CuArray, [r, c, W])
args = EOTArgs(itermax=100000, epsilon=1e-6, eta_p=1.)
# ACMirrorProxDual(W, r, c, args)
args = EOTArgs(itermax=20000, epsilon=1e-8, eta_p=1e-5, eta_mu=1e-10, B=1.1)
# prototype_extragradient_ot2(r, c, W, args)
W /= 2norm(W, Inf)
# extragradient_ot_dual(r, c, W, args)
args = EOTArgs(itermax=20000, epsilon=1e-8, eta_p=1e-5, eta_mu=1e-5, B=2.)
# prototype_extragradient_ot(r, c, W, args)
# prototype_extragradient_ot(r, c, W, args)
papc_ot_dual(r, c, W, args)
extragradient_ot_dual(r, c, W, args)