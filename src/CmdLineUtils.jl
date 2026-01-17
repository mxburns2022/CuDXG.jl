using ArgParse
using PythonOT
solvers = Dict(
    "dual_extragradient" => extragradient_ot_dual,
    "primal_extragradient" => extragradient_ot,
    "sinkhorn" => sinkhorn_log,
    "greenkhorn" => greenkhorn_log,
    "apdamd" => APDAMD,
    "apdagd" => APDAGD,
    "hpd" => HPD,
    "accelerated_sinkhorn" => accelerated_sinkhorn,
    "dual_extrapolation" => dual_extrapolation
    # "abdg" => accelerated_bregman_descent,
)
ctransfer_solvers = Dict(
    "dual_extragradient" => extragradient_ot_dual,
    "sinkhorn" => sinkhorn_log
)
barycenter_solvers = Set([
    "dual_extragradient",
    "ipb"]
)
settings = ArgParseSettings(prog="cudxg")
@add_arg_table! settings begin
    "run"
    help = "Run discrete OT problem"
    action = :command

    "ctransfer"
    help = "Perform color transfer using a metric kernel. CUDA is used by default."
    action = :command

    "barycenter"
    help = "Compute a Wasserstein barycenter for a collection of marginals. CUDA is used by default."
    action = :command

end
@add_arg_table! settings["run"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the DOT instance. Options are: $(join(keys(solvers), ", "))"
    range_tester = x -> x in keys(solvers)
    default = "sinkhorn"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--cuda"
    help = "Use CUDA"
    action = :store_true
    "--p"
    help = "p for distance computation (>= 10 for infinity norm, 0 for uniform cost)"
    arg_type = Float64
    default = 2.0
    "--weights"
    help = "Path to CSV-formatted weight matrix"
    default = ""
    "--kernel"
    help = "Use kernels to compute OT matrices on the fly (only dual_extragradient and sinkhorn are supported)"
    action = :store_true
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
    "file1"
    help = "Path to target DOTMark-formatted file (row marginal) (TODO: Add support for more input types)"
    required = true
    "file2"
    help = "Path to target DOTMark-formatted file (col marginal) (TODO: Add support for more input types)"
    required = true
    "--output1"
    help = "Output path for assignment 1"
    default = ""
    "--output2"
    help = "Output path for assignment 2"
    default = ""
    "--potential-out"
    help = "Output path for dual potentials. Order is (1) Simplex dual (if using extragradient), (2) Potential for Row Marginal, (3) Potential for Column Marginal>"
    default = ""
end
@add_arg_table! settings["ctransfer"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the color transfer instance. Options are: $(join(keys(ctransfer_solvers), ", "))"
    range_tester = x -> x in keys(ctransfer_solvers)
    default = "sinkhorn"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
    "--p"
    help = "p for distance computation (>= 10 for infinity norm, 0 for uniform cost)"
    arg_type = Float64
    default = 2.0
    range_tester = x -> x >= 0
    "--height"
    help = "Image height"
    default = 128
    arg_type = Int
    "--width"
    help = "Image width"
    default = 128
    arg_type = Int
    "file1"
    help = "Path to target input image file (row marginal)"
    required = true
    "file2"
    help = "Path to target input image file (column marginal)"
    required = true
    "--output1"
    help = "Output path for color mapped image 1"
    required = true
    "--output2"
    help = "Output path for color mapped image 2"
    required = true
end
@add_arg_table! settings["barycenter"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the barycenter problem. Options are: $(join(barycenter_solvers, ", "))"
    range_tester = x -> x in barycenter_solvers
    default = "ipb"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
    "--p"
    help = "p for distance computation (>= 10 for infinity norm, 0 for uniform cost)"
    arg_type = Float64
    default = 2.0
    range_tester = x -> x >= 0
    "--weights"
    help = "Weights for Barycenter objective (default is uniform). If provided, number of weights must match the number of distributions"
    arg_type = Float64
    action = :store_arg
    nargs = '*'
    "--cost"
    help = "Path to cost matrix. If not provided, then \"--supports\" must be provided and Euclidean kernel will be used"
    "--supports"
    help = "Path to distribution supports for kernel computation. If not provided, then \"--cost\" must be provided. 
    Either one support must be provided (common support) or the number of supports must match the number of input distributions"
    "marginals"
    help = "Paths to target input image file (row marginal)"
    required = true
    action = :store_arg
    nargs = '*'
    "--output"
    help = "Output path for Wasserstein barycenter"
    required = true
    "--kernel"
    action = :store_true
    help = "Use "
end
function run_dot(parsed_args)
    args = read_args_json(parsed_args["settings"])
    marginal1, h, w, N = read_dotmark_data(parsed_args["file1"])
    marginal2, h2, w2, N2 = read_dotmark_data(parsed_args["file2"])
    @assert h == h2 && w == w2 && N == N2
    # mix it with a little bit of the uniform distribution for stability
    r = normalize(marginal1 .+ 1e-6, 1)
    c = normalize(marginal2 .+ 1e-6, 1)
    if !parsed_args["kernel"]
        if parsed_args["weights"] != ""
            W = read_weights(parsed_args["weights"])
        else
            W = get_euclidean_distance(h, w; p=parsed_args["p"])
        end
        W∞ = norm(W, Inf)
        # args.eta_p /= W∞
        # args.epsilon /= W∞
        W ./= W∞
        # W = (W .+ 1) / 2
        # W .= rand(rng, N, N2)
        if parsed_args["cuda"]
            r, c, W = map(CuArray, [r, c, W])
        end
        solvers[parsed_args["algorithm"]](r, c, W, args, parsed_args["frequency"])
    else
        r = CuArray(r)
        c = CuArray(c)
        locations = zeros(Float64, 3, h * w)
        for i in 1:h*w
            locations[1, i] = (i - 1) ÷ w
            locations[2, i] = (i - 1) % w
        end
        locations = CuArray(locations)
        if parsed_args["algorithm"] == "sinkhorn"
            sinkhorn_euclidean(r, c, locations, locations, parsed_args["output1"], parsed_args["output2"], parsed_args["potential-out"], args, parsed_args["frequency"], parsed_args["p"])
        elseif parsed_args["algorithm"] == "dual_extragradient"
            extragradient_euclidean(r, c, locations, locations, parsed_args["output1"], parsed_args["output2"], parsed_args["potential-out"], args, parsed_args["frequency"], parsed_args["p"])
        end
    end
end

function run_ctransfer(parsed_args)
    args = read_args_json(parsed_args["settings"])
    size = (parsed_args["height"], parsed_args["width"])
    if parsed_args["algorithm"] == "dual_extragradient"
        extragradient_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], size, args, parsed_args["frequency"], parsed_args["p"])
    elseif parsed_args["algorithm"] == "sinkhorn"
        sinkhorn_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], size, args, parsed_args["frequency"], parsed_args["p"])
    end
end

function run_barycenter(parsed_args)
    args = read_args_json(parsed_args["settings"])
    weights = parsed_args["weights"]
    if size(weights, 1) != 0
        @assert size(weights, 1) == size(parsed_args["marginals"], 1)
    else
        weights = ones(size(parsed_args["marginals"], 1))
    end
    h = 0
    w = 0
    marginals = Vector{CuArray}()
    for fname in (parsed_args["marginals"])
        marginal, h, w, N = read_dotmark_data(fname)
        marginal .+= 1e-6
        normalize!(marginal, 1)
        push!(marginals, CuArray(marginal))
    end
    locations = zeros(Float64, 3, h * w)
    for i in 1:h*w
        locations[1, i] = (i - 1) ÷ w
        locations[2, i] = (i - 1) % w
    end
    locations = CuArray(locations)
    args = read_args_json(parsed_args["settings"])
    W∞ = (h - 1.0)^2 + (w - 1)^2
    if parsed_args["algorithm"] == "ipb"
        r, mup, mun = ipb_kernel(marginals, locations, locations, W∞, args, weights, parsed_args["frequency"], parsed_args["p"])
        outname = ".duals"
    elseif parsed_args["algorithm"] == "dual_extragradient"
        r, mup, mun = extragradient_barycenter_kernel(marginals, locations, locations, W∞, args, weights, parsed_args["frequency"], Int(parsed_args["p"]))
        outname = ".mu"
    end
    m = size(marginals, 1)
    r = Array(r)
    mup = map(Array, mup)
    mun = map(Array, mun)
    outpath = parsed_args["output"]
    open(outpath * ".barycenter", "w") do outfile
        for ri in r
            println(outfile, ri)
        end
    end
    open(outpath * outname, "w") do outfile
        # println(outfile, "")
        for i in 1:h*w
            for j in 1:m
                print(outfile, "$(mup[j][i]) $(mun[j][i]) ")
            end
            println(outfile)
        end
    end

end

function run_from_arguments(arguments::Vector{String})
    parsed_args = parse_args(arguments, settings)
    if parsed_args["%COMMAND%"] == "run"
        run_dot(parsed_args["run"])
    elseif parsed_args["%COMMAND%"] == "ctransfer"
        run_ctransfer(parsed_args["ctransfer"])
    elseif parsed_args["%COMMAND%"] == "barycenter"
        run_barycenter(parsed_args["barycenter"])
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_from_arguments(ARGS)
end