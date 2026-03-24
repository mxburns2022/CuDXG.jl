#file CuLAMP.jl

module CuLAMP
# no dependencies declared here
include("Utilities.jl")
include("ImgUtilities.jl")
include("CellUtilities.jl")
include("CellSimilarity.jl")
include("solvers/MirrorProx.jl")
include("solvers/MirrorProxKernel.jl")
include("solvers/MirrorProxCellSim.jl")
include("solvers/Sinkhorn.jl")
include("solvers/DualExtrapolation.jl")
include("solvers/SinkhornKernel.jl")
include("solvers/SinkhornCellSim.jl")
include("solvers/AccBregmanDescent.jl")
include("solvers/APDAMD.jl")
include("solvers/APDAGD.jl")
include("solvers/Greenkhorn.jl")
include("solvers/AccSinkhorn.jl")
include("solvers/HPD.jl")
include("CmdLineUtils.jl")
export read_dotmark_data, get_euclidean_distance
export OTScOmicsCellData, CellCostKernel, read_otscomics_cell_data, infer_cell_clusters, feature_cost
export extragradient_ot, LAMP, extragradient_ot_full_dual, extragradient_barycenter_dual
export extragradient_cellsim
export sinkhorn_cellsim
export sinkhorn_log
export EOTArgs, load_rgb, save_image
export run_from_arguments, solvers, sinkhorn_color_transfer, extragradient_color_transfer, accelerated_bregman_descent_transfer, accelerated_bregman_descent, dual_extrapolation
export extragradient_cell_distance, streaming_c_index, compute_otscomics_c_index
export sinkhorn_cell_distance
export warp_logsumexp_spp_ct_opt!, warp_logsumexp_spp_ct_fused!, warp_logsumexp_spp_opt!, warp_logsumexp_fused!, warp_logsumexp!,warp_logsumexp_spp_ct_opt_smem!,warp_logsumexp_spp_ct_opt_smem_fused!,naive_logsumexp_spp_ct!

end
