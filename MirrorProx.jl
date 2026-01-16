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


function DualExtrapolation(W::TW, r::TM, c::TM, args::EOTArgs, frequency::Int=100) where {TW<:AbstractMatrix, TM<:AbstractVector}
  n = size(r, 1)
  x = TW(ones(n, n)) / n^2
  y = TM(zeros(n))
  z = TM(zeros(n))
  x̂ = copy(x)
  ŷ = copy(y)
  ẑ = copy(z)
  sx = TW(zeros(n, n))
  sy = TM(zeros(n))
  sz = TM(zeros(n))
  W∞ = norm(W, Inf)
  function col(x)
    return sum(x', dims=2)
  end
  function row(x)
    return sum(x, dims=2)
  end

  println("time(s),iter,inner_iter,infeas,ot_objective,primal,dual,solver")
  time_start = time_ns()
  for i in 1:args.itermax

    for k in 1:args.inner_iter
      x = exp.(args.tau_p/(2W∞).*sx .- args.tau_p * (y.^2 .+ (z.^2)'))
      x ./= sum(x)
      y = min.(1, max.(-1, args.tau_mu * sy ./ (W∞.*row(x))))
      z = min.(1, max.(-1, args.tau_mu * sz ./ (W∞.*col(x))))
    end
    axpby!(1/(i+1), x, (i)/(i+1), x̂)
    axpby!(1/(i+1), y, (i)/(i+1), ŷ)
    axpby!(1/(i+1), z, (i)/(i+1), ẑ)

    sxt = sx + 1/3 * (-W - 2W∞.*(y .+ z'))

    syt = sy + 1/3 * (2W∞.*(row(x)-r))
    szt = sz + 1/3 * (2W∞.*(col(x)-c))
    for k in 1:args.inner_iter
      x = exp.(args.tau_p/(2W∞) .* sxt .- args.tau_p * (y.^2 .+ (z.^2)'))
      x ./= sum(x)
      y = min.(1, max.(-1, args.tau_mu * syt ./ (W∞.*row(x))))
      z = min.(1, max.(-1, args.tau_mu * szt ./ (W∞.*col(x))))
    end
    sx += 1/6 * (-W - 2W∞.*(y .+ z'))
    sy += 1/6 * (2W∞.*(row(x)-r))
    sz += 1/6 * (2W∞.*(col(x)-c))
    time_elapsed = time_ns()
    objective = dot(W, x̂)
    infeas = norm(col(x̂) - c, 1) + norm(row(x̂) - r, 1)
    primal = objective + 2W∞*infeas
    dual = -2W∞*(dot(ŷ, r) + dot(ẑ, c)) + minimum(W .+ 2W∞*(ŷ .+ ẑ'))
    if (i-1) % frequency == 0
      println("$((time_elapsed-time_start)/1e9),$(i),$(i * 2args.inner_iter),$(infeas),$(objective),$(primal),$(dual),dual_extrap")
    end
    if primal-dual <= args.epsilon
      break
    end
  end
  return x̂, ŷ, ẑ
end


n = 10
N = n * n
m = 10
M = m * m
# N = 5
# M = 5
M = 10
N = 10
rng = Xoshiro(0)
r, c, W, optimum = generate_random_ot(N, M, rng)
c .+= 1e-6
r .+= 1e-6
normalize!(r, 1)
normalize!(c, 1)
# fill!(c, 0.0)
# c[1] = 1.0
# c .+= 1e-5
# normalize!(c, 1)
# fill!(r, 0.0)
# r[2] = 1.0
# r .+= 1e-6
# normalize!(, 1)
# W = get_euclidean_distance(n, n, p=1.)
# W /= norm(W, Inf)
# r, c, W = map(CuArray, [r, c, W])
args = EOTArgs(itermax=100000, epsilon=1e-6, eta_p=1.)
# ACMirrorProxDual(W, r, c, args)
args = EOTArgs(itermax=2, epsilon=1e-8, eta_p=1e-5, eta_mu=1e-10, B=1.1)
# prototype_extragradient_ot2(r, c, W, args)
W /= norm(W, Inf)
# extragradient_ot_dual(r, c, W, args)
args = EOTArgs(itermax=200000, epsilon=1e-4, eta_p=0.0, eta_mu=0.0, B=2.0, tau_p = 0.5, tau_mu = 0.3, inner_iter=20)
DualExtrapolation(W, r, c, args)
r, c, W = map(Array, [r, c, W])

println(emd2(r, c, W))
# prototype_extragradient_ot(r, c, W, args)
# prototype_extragradient_ot(r, c, W, args)
# papc_ot_dual(r, c, W, args)
# extragradient_ot_dual(r, c, W, args)