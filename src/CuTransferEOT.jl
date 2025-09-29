#file CuTransferEOT.jl
module CuTransferEOT
# no dependencies declared here
include("Utilities.jl")
include("solvers/Extragradient.jl")
include("solvers/Sinkhorn.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
export read_dotmark_data, get_euclidean_distance
export extragradient_ot, extragradient_ot_dual
export sinkhorn_log
export EOTArgs

end

