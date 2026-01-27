using BenchmarkTools
using DataFrames
using CuLAMP
using CUDA
using CSV

function test_extragradient_kernels(;run_bench=false)
  data = []
  for W in [32, 64, 128, 256]
    H = W
    N = W * H
    img1 = CUDA.rand(Float64, 3, N)
    img2 = CUDA.rand(Float64, 3, N)
    theta = 2*CUDA.rand(Float64, N) .- 1
    output = CUDA.zeros(Float64, N)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)

    global output
    global img1
    global img2
    global theta
    global threads
    global warp_blocks
    global linear_blocks
    println(CUDA.memory_status())
    @cuda threads = threads blocks = linear_blocks naive_logsumexp_spp_ct!(output, img1,
      img2, theta, 1e-4, 0.5, 1.0, 2.0) 
    if run_bench
       @cuda threads = threads blocks = linear_blocks naive_logsumexp_spp_ct!(output, img1,
      img2, theta, 1e-4, 0.5, 1.0, 2.0)
      b1 = @benchmark CUDA.@sync @cuda threads = threads blocks = linear_blocks naive_logsumexp_spp_ct!(output, img1, img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      push!(data,
        (N=N, W=W, kernel="naive", med_time=median(b1.times), avg_time=mean(b1.times), std_time=std(b1.times))
        )
    end

    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_fused!(output, img1,
      img2, theta, 1e-4, 0.5, 1.0, 2.0) 
    if run_bench
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_fused!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      b2 = @benchmark CUDA.@sync @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_fused!(output, img1, img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      push!(data,
        (N=N, W=W, kernel="fused_warp", med_time=median(b2.times), avg_time=mean(b2.times), std_time=std(b2.times))
        )
    end

    
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(output, img1,
      img2, theta, 1e-4, 0.5, 1.0, 2.0) 
    if run_bench
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      b3 = @benchmark CUDA.@sync @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt!(output, img1, img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      push!(data,
        (N=N, W=W, kernel="warp", med_time=median(b3.times), avg_time=mean(b3.times), std_time=std(b3.times))
        )
    end

    
    
    if run_bench
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      b4 = @benchmark CUDA.@sync @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem!(output, img1, img2, theta, 1e-4, 0.5, 1.0, 2.0) 

      push!(data,
        (N=N, W=W, kernel="warp_smem", med_time=median(b4.times), avg_time=mean(b4.times), std_time=std(b4.times))
        )
    end

    if run_bench
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem_fused!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem_fused!(output, img1,
        img2, theta, 1e-4, 0.5, 1.0, 2.0)
      b5 = @benchmark CUDA.@sync @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_ct_opt_smem_fused!(output, img1, img2, theta, 1e-4, 0.5, 1.0, 2.0) 
      push!(data,
        (N=N, W=W, kernel="warp_smem_fused", med_time=median(b5.times), avg_time=mean(b5.times), std_time=std(b5.times))
        )
      df = DataFrame(data)
      CSV.write("kernel_timings_output.csv", df)
    end

  end
  return DataFrame(data)
end

test_extragradient_kernels(;run_bench=true)
