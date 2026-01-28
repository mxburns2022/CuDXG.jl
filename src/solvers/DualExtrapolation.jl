
function dual_extrapolation(r::TM, c::TM, W::TW, args::EOTArgs, frequency::Int=100) where {TW<:AbstractMatrix, TM<:AbstractVector}
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
