#file CuTransferEOT.jl

module CuTransferEOT
# no dependencies declared here
include("Utilities.jl")
include("ImgUtilities.jl")
include("solvers/Extragradient.jl")
include("solvers/ExtragradientKernel.jl")
include("solvers/Sinkhorn.jl")
include("solvers/DualExtrapolation.jl")
include("solvers/SinkhornKernel.jl")
include("solvers/AccBregmanDescent.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
include("solvers/Greenkhorn.jl")
include("solvers/AccSinkhorn.jl")
include("solvers/HPD.jl")
include("CmdLineUtils.jl")
include("solvers/ExtragradientBarycenter.jl")
export read_dotmark_data, get_euclidean_distance
export extragradient_ot, extragradient_ot_dual, extragradient_ot_full_dual, extragradient_barycenter_dual
export sinkhorn_log
export EOTArgs, load_rgb, save_image
export run_from_arguments, solvers, sinkhorn_color_transfer, extragradient_color_transfer, accelerated_bregman_descent_transfer, accelerated_bregman_descent, dual_extrapolation
export warp_logsumexp_spp_ct_opt!, warp_logsumexp_spp_ct_fused!, warp_logsumexp_spp_opt!, warp_logsumexp_fused!, warp_logsumexp!,warp_logsumexp_spp_ct_opt_smem!,warp_logsumexp_spp_ct_opt_smem_fused!,naive_logsumexp_spp_ct!

end