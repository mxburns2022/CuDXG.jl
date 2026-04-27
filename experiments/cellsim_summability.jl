import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CuLAMP
using IterTools
using Formatting
using Random
using DataFrames
using CSV

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATASET_FILE = joinpath(REPO_ROOT, "OT-scOmics", "data", "liu_scatac_preprocessed.csv.gz")
const OUTPUT_DIRECTORY = joinpath(REPO_ROOT, "data", "experiment_summability")

const seed = 1
"""
Run cell-similarity solvers on a fixed OT-scOmics dataset while varying
the number of features and the feature-cost metric.
"""
function main(argval)
    mkpath(OUTPUT_DIRECTORY)
    rng = Xoshiro(seed)
    dataset_name = splitext(splitext(basename(DATASET_FILE))[1])[1]
    num_cells = 50
    num_subproblems = 10
    if parse(Int, argval) == 1
        costs = ["cosine", "correlation"]
    else
        costs = ["l1", "l2"]
    end

    feature_counts = [256, 576, 1024, 1600]
    probnumbers = [1:num_subproblems...]
    niters = size(feature_counts, 1) * size(costs, 1)
    i = 0
    args = EOTArgs(itermax=10_000, verbose=false)
    for (cost, _, num_features) in product(costs, probnumbers, feature_counts)
        println((round(i / niters * 100), dataset_name, num_cells, num_features, cost))
        flush(stdout)

        cell_index_1 = Int(rand(rng, UInt32) % num_cells + 1)
        cell_index_2 = Int(rand(rng, UInt32) % num_cells + 1)
        while cell_index_2 == cell_index_1
            cell_index_2 = Int(rand(rng, UInt32) % num_cells + 1)
        end
        output_file = join(
            ["cell_summability", dataset_name, "cells$(num_cells)",
                "features$(num_features)", cost, "cella$(cell_index_1)", "cellb$(cell_index_2)"],
            "_",
        ) * "_log.csv"
        r, c, W, _, _ = get_cell_similarity_problem(DATASET_FILE, num_features, num_cells, cell_index_1, cell_index_2, cost)
        r = (r .+ 1e-7) / (1 + num_features * 1e-7)
        c = (c .+ 1e-7) / (1 + num_features * 1e-7)
        _, _, log_output = LAMP(r, c, W, args, 10000; log_output=true,)
        df = DataFrame(log_output)
        df[!, "diff"] = df[!, "cross_term_2"] - df[!, "cross_term_1"]
        df[!, "cumsum"] = cumsum(df[!, "diff"])
        df = df[1:20:nrow(df), :]



        open(joinpath(OUTPUT_DIRECTORY, output_file), "w") do fout
            CSV.write(fout, df)
        end
        i += 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
