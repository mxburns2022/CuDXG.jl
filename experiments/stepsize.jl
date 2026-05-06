using Pkg
Pkg.activate(".")
using CuLAMP
using Images
using ImageFiltering
using Plots
using Suppressor
using Random
using LinearAlgebra
using CUDA
using IterTools
using ProgressBars
using DataFrames
using CSV



function downsample(marginal, original_w, original_h, new_w, new_h)
  img = reshape(marginal, (original_w, original_h))
  if new_w <= original_w && new_h <= original_h
    blurred = imfilter(img, ImageFiltering.Kernel.gaussian(0.5))
    return reshape(imresize(blurred, new_w, new_h), (new_w * new_h,))
  else
    return reshape(imresize(img, new_w, new_h), (new_w * new_h,))
  end
end


function run_experiment()
  gen = Xoshiro(0)
  nprobs = 50
  classes = readdir(ENV["BENCH"] * "/DOTmark_1.0/Data")
  class_labels = abs.(rand(gen, Int, nprobs)) .% size(classes) .+ 1
  prob_labels_a = abs.(rand(gen, Int, nprobs)) .% 10 .+ 1
  prob_labels_b = abs.(rand(gen, Int, nprobs)) .% 10 .+ 1
  # Ensure there are no duplicates
  prob_labels_a[prob_labels_a .== prob_labels_b] = ((prob_labels_a[prob_labels_a .== prob_labels_b] .+ 1) .% 10) .+ 1
  problems_a = ["$(classes[i])/data32_10$(lpad(j, 2,"0")).csv" for (i, j) in zip(class_labels, prob_labels_a)]
  problems_b = ["$(classes[i])/data32_10$(lpad(j, 2,"0")).csv" for (i, j) in zip(class_labels, prob_labels_b)]
  widthvals = [20, 28, 32, 36, 40]
  tau_p = 1.0
  tau_mu = 1.0
  eta = 0.
  p = 2.
  df_base = DataFrame()
  for width in widthvals
  for tau in [0.25, 0.5, 0.75, 1.0, 1.01, 1.05, 1.1]
    if tau != 1.1 && width == 32
      continue
    end
    for (prob1, prob2) in ProgressBar(zip(problems_a, problems_b))
      tau_p = tau
      tau_mu = tau

      # function run_problem(path1, path2)
      r, w, h, N = read_dotmark_data(ENV["BENCH"] * "/DOTmark_1.0/Data/$(prob1)")

      c, w, h, N = read_dotmark_data(ENV["BENCH"] * "/DOTmark_1.0/Data/$(prob2)")
      r = downsample(r, w, h, width, width)
      c = downsample(c, w, h, width, width)
      r .+= 1e-6
      c .+= 1e-6
      r ./= sum(r)
      c ./= sum(c)
      W = get_euclidean_distance(width, width; p = p)
      W /= norm(W, Inf)

      args = EOTArgs(tau_p = tau_p, tau_mu = tau_mu, eta_p = eta, eta_mu = eta, epsilon=1e-10)
      r, c, W = map(CuArray, [r, c, W])

      a = @capture_out begin primal, theta = LAMP(r, c, W, args, 500) end
      # CSV.read(a)
      df = CSV.read(IOBuffer(a), DataFrame)
      df[!, "N"] .= width * width
      df[!, "w"] .= width
      df[!, "tau"] .= tau
      df[!, "eta"] .= eta
      df[!, "p"] .= p
      df[!, "prob1"] .= prob1
      df[!, "prob2"] .= prob2
      df_base = vcat(df_base, df)
    end
    CSV.write( "../data/stepsize_experiment.csv", df_base)

  end
  end
end
