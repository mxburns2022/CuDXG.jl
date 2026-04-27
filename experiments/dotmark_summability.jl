import Pkg
Pkg.activate(".")
using CuLAMP
using IterTools
using CSV
using DataFrames
using Suppressor
using CUDA
using Formatting
benchmark_directory = ENV["BENCH"] * "/DOTmark_1.0/Data"
output_directory = "data/experiment_dotmark_summability"
if !isdir(output_directory)
    mkdir(output_directory)
end

"""
Run DOT solvers on selected problems from the DOTmark_1.0 dataset
"""
benchmark_classes = ["ClassicImages", "GRFsmooth", "GRFrough"]
sizes = [16, 24, 32, 40]
instance_pairs = [(2, 6), (3, 5), (4, 6), (8, 10), (1, 7)]

args = EOTArgs()
args.itermax = 10_000
args.epsilon = 1e-10
pvals = [1.0, 2.0, Inf]

for (p, sizeval, (ind1, ind2), probclass) in product(pvals, sizes, instance_pairs, benchmark_classes)
    println((sizeval, p, (ind1, ind2), probclass))
    flush(stdout)
    file1 = joinpath(benchmark_directory, probclass, "data64_10$(lpad(ind1, 2, '0')).csv")
    file2 = joinpath(benchmark_directory, probclass, "data64_10$(lpad(ind2, 2, '0')).csv")
    size_tuple = (sizeval, sizeval)
    r, h, w, N = read_dotmark_data(file1, size_tuple)
    c, h, w, N = read_dotmark_data(file2, size_tuple)
    W = get_euclidean_distance(sizeval, sizeval; p=p)
    # Format ouutput file as <problem_class>_<resolution>_<index of img1>_<index of img2>_<η value>_<solver_name>_log.csv
    output_file = join(["dotmark", "p$p", lowercase(probclass), "size$sizeval", ind1, ind2], "_") * "_log.csv"

    r = (r .+ 1e-7) / (1 + N * 1e-7)
    c = (c .+ 1e-7) / (1 + N * 1e-7)
    r, c, W = map(CuArray, [r, c, W])
    _, _, log_output = LAMP(r, c, W, args, 10000; log_output=true,)
    df = DataFrame(log_output)
    df[!, "diff"] = df[!, "cross_term_2"] - df[!, "cross_term_1"]
    df[!, "cumsum"] = cumsum(df[!, "diff"])
    df = df[1:20:nrow(df), :]



    open(joinpath(output_directory, output_file), "w") do fout
        CSV.write(fout, df)
    end
end

