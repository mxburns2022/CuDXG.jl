using ArgParse
solvers = Dict(
    "dual_extragradient" => extragradient_ot_dual,
    "primal_extragradient" => extragradient_ot,
    "sinkhorn" => sinkhorn_log,
    "greenkhorn" => greenkhorn_log,
    "apdamd" => APDAMD,
    "apdagd" => APDAGD,
    "accelerated_sinkhorn" => accelerated_sinkhorn,
    "abdg" => accelerated_bregman_descent,
)
ctransfer_solvers = Dict(
    "dual_extragradient" => extragradient_ot_dual,
    "sinkhorn" => sinkhorn_log
)
settings = ArgParseSettings(prog="eotfom")
@add_arg_table! settings begin
    "run"
    help = "Run discrete OT problem with an explicit matrix"
    action = :command

    "ctransfer"
    help = "Perform color transfer using a Euclidean kernel. CUDA is used by default."
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
    help = "Algorithm to solve the barycenter problem. Options are: $(join(keys(ctransfer_solvers), ", "))"
    range_tester = x -> x in keys(ctransfer_solvers)
    default = "sinkhorn"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
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
    "files"
    help = "Path to target input image file (row marginal)"
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
    W = get_euclidean_distance(h, w)
    W∞ = norm(W, Inf)
    # args.eta_p /= W∞
    # args.epsilon /= W∞
    W ./= W∞
    if parsed_args["cuda"]
        r, c, W = map(CuArray, [r, c, W])
    end
    solvers[solver](r, c, W, args, parsed_args["frequency"])
end

function run_ctransfer(parsed_args)
    args = read_args_json(parsed_args["settings"])
    size = (parsed_args["height"], parsed_args["width"])
    if parsed_args["algorithm"] == "dual_extragradient"
        extragradient_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], size, args, parsed_args["frequency"])
    elseif parsed_args["algorithm"] == "sinkhorn"
        sinkhorn_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], args.eta_p, size, args.itermax, parsed_args["frequency"])
    end
end

function run_barycenter(parsed_args)
    args = read_args_json(parsed_args["settings"])
    weights = if isin("weights",)[parsed_args["weights"]...]
    else
        0
    end

    marginal2, h2, w2, N2 = read_dotmark_data(parsed_args["file2"])
    @assert h == h2 && w == w2 && N == N2
    # mix it with a little bit of the uniform distribution for stability
    r = normalize(marginal1 .+ 1e-6, 1)
    c = normalize(marginal2 .+ 1e-6, 1)
    W = get_euclidean_distance(h, w)
    W∞ = norm(W, Inf)
    # args.eta_p /= W∞
    # args.epsilon /= W∞
    W ./= W∞
    if parsed_args["cuda"]
        r, c, W = map(CuArray, [r, c, W])
    end
    solvers[solver](r, c, W, args, parsed_args["frequency"])
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