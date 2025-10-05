#file CuTransferEOT.jl

module CuTransferEOT
# no dependencies declared here
include("Utilities.jl")
include("solvers/Extragradient.jl")
include("solvers/ExtragradientKernel.jl")
include("solvers/Sinkhorn.jl")
include("solvers/SinkhornKernel.jl")
include("solvers/AccBregmanDescent.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
include("solvers/Greenkhorn.jl")
include("solvers/AccSinkhorn.jl")
include("CmdLine.jl")
export read_dotmark_data, get_euclidean_distance
export extragradient_ot, extragradient_ot_dual, extragradient_ot_full_dual
export sinkhorn_log
export EOTArgs
export run_from_arguments, solvers, sinkhorn_color_transfer, extragradient_color_transfer

end