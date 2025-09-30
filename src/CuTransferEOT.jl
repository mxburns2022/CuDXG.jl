#file CuTransferEOT.jl
module CuTransferEOT
# no dependencies declared here
include("Utilities.jl")
include("solvers/Extragradient.jl")
include("solvers/Sinkhorn.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
include("solvers/Greenkhorn.jl")
include("solvers/AccSinkhorn.jl")
export read_dotmark_data, get_euclidean_distance
export extragradient_ot, extragradient_ot_dual
export sinkhorn_log
export EOTArgs
end
using ArgParse

include("Utilities.jl")
include("solvers/Extragradient.jl")
include("solvers/Sinkhorn.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
include("solvers/Greenkhorn.jl")
include("solvers/AccSinkhorn.jl")
solvers = Dict(
    "dual_extragradient" => extragradient_ot_dual,
    "primal_extragradient" => extragradient_ot,
    "sinkhorn" => sinkhorn_log,
    "greenkhorn" => greenkhorn_log,
    "apdamd" => APDAMD,
    "apdagd" => APDAGD,
    "accelerated_sinkhorn" => accelerated_sinkhorn
)


settings = ArgParseSettings(prog="eotfom")
@add_arg_table! settings begin
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
    "file1"
    help = "Path to target DOTMark-formatted file (row marginal) (TODO: Add support for more input types)"
    required = true
    "file2"
    help = "Path to target DOTMark-formatted file (col marginal) (TODO: Add support for more input types)"
    required = true
end


parsed_args = parse_args(ARGS, settings)
println(parsed_args)
args = read_args_json(parsed_args["settings"])
solver = parsed_args["algorithm"]
marginal1, h, w, N = read_dotmark_data(parsed_args["file1"])
marginal2, h2, w2, N2 = read_dotmark_data(parsed_args["file2"])
@assert h == h2 && w == w2 && N == N2
# mix it with a little bit of the uniform distribution for stability
r = normalize(marginal1 .+ 1e-6, 1)
c = normalize(marginal2 .+ 1e-6, 1)
W = get_euclidean_distance(h, w)
W∞ = norm(W, Inf)
# args.ηp /= W∞
# args.ε /= W∞
W ./= W∞
if parsed_args["cuda"]
    r, c, W = map(CuArray, [r, c, W])
end
solvers[solver](r, c, W, args, parsed_args["frequency"])