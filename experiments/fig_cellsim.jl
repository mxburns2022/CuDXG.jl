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
function main()
    mkpath(OUTPUT_DIRECTORY)

    dataset_name = splitext(splitext(basename(DATASET_FILE))[1])[1]
    num_cells = 50
    feature_counts = [1000, 2000, 3000, 4000, 5000, 6000]
    costs = ["l1", "l2", "cosine", "correlation"]
    algorithms = ["lamp", "sinkhorn", "annealed_sinkhorn"]

    for (solver, num_features, cost) in product(algorithms, feature_counts, costs)
        println((dataset_name, num_cells, num_features, cost, solver))
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
            "--frequency", "25",
            "--cost", cost,
            "--num-features", string(num_features),
            "--num-cells", string(num_cells),
            DATASET_FILE,
        ]

        output_log = @capture_out begin
            run_from_arguments(arglist)
        end

        open(joinpath(OUTPUT_DIRECTORY, output_file), "w") do fout
            write(fout, output_log)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
