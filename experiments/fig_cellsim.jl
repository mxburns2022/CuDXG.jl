import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CuLAMP
using IterTools
using Suppressor
using Formatting

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATASET_FILE = joinpath(REPO_ROOT, "OT-scOmics", "data", "leukemia_preprocessed.csv.gz")
const OUTPUT_DIRECTORY = joinpath(REPO_ROOT, "data", "experiment_cellsim_broad")

"""
Run cell-similarity solvers on a fixed OT-scOmics dataset while varying
the number of features and the feature-cost metric.
"""
function main(argval)
    mkpath(OUTPUT_DIRECTORY)

    dataset_name = splitext(splitext(basename(DATASET_FILE))[1])[1]
    num_cells = 20
    num_subproblems = 1
    if parse(Int, argval) == 1
        costs = ["cosine", "correlation"]
    else
        costs = ["l1", "l2"]
    end
        
    feature_counts = [500]
    
    algorithms = ["lamp", "sinkhorn", "annealed_sinkhorn"]
    niters = size(algorithms, 1) * size(feature_counts, 1) * size(costs, 1)
    i = 0
    for (solver, cost, num_features) in product(algorithms, costs, feature_counts)
        println((round(i / niters * 100), dataset_name, num_cells, num_features, cost, solver))
        flush(stdout)

        input_file = joinpath(REPO_ROOT, "configurations", "cellsim", "$(solver).json")
        output_file = join(
            [dataset_name, "cells$(num_cells)", "features$(num_features)", cost, solver],
            "_",
        ) * "_log.csv"

        arglist = [
            "cellsim",
            "--algorithm", solver,
            "--settings", input_file,
            "--frequency", "250",
            "--cost", cost,
            "--num-features", string(num_features),
            "--num-cells", string(num_cells),
            "--num-subproblems", string(num_subproblems),
            DATASET_FILE,
        ]
        
        output_log = @capture_out begin
            run_from_arguments(arglist)
        end

        open(joinpath(OUTPUT_DIRECTORY, output_file), "w") do fout
            write(fout, output_log)
        end
        i += 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
