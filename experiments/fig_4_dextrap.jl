import Pkg
Pkg.activate(".")
using CuLAMP
using IterTools
using Suppressor
using Formatting
benchmark_directory = ENV["BENCH"] * "/DOTmark_1.0/Data"
output_directory = "../data"
if !isdir(output_directory)
    mkdir(output_directory)
end

"""
Run DOT solvers on selected problems from the DOTmark_1.0 dataset
"""
benchmark_classes = ["ClassicImages", "GRFsmooth", "MicroscopyImages"]
sizes = [32]
instance_pairs = [(2, 6), (3, 5), (5, 7), (1, 2), (2, 10)]
algorithms = [keys(solvers)...]
pvals = ["2"]
algorithms = [("dual_extrapolation", 0.0), ("dual_extragradient", 0)]
# algorithms = [ "sinkhorn", ]
# configuration = "default_1e-05.json"
for (p, (solver,eps), (ind1, ind2),  probclass, size) in product(pvals, algorithms, instance_pairs, benchmark_classes, sizes)
    println((size, (ind1, ind2), solver, probclass, eps))
    flush(stdout)
    file1 = joinpath(benchmark_directory, probclass, "data$(size)_10$(lpad(ind1, 2, '0')).csv")
    file2 = joinpath(benchmark_directory, probclass, "data$(size)_10$(lpad(ind2, 2, '0')).csv")
    epsstring = sprintf1("%1.0e", eps)
    # Format ouutput file as <problem_class>_<resolution>_<index of img1>_<index of img2>_<η value>_<solver_name>_log.csv
    output_file = "extrap_comp_" * join([lowercase(probclass), size, ind1, ind2, epsstring, solver, p], "_") * "_log.csv"
    input_file = if solver == "sinkhorn" "../configurations/default_$(epsstring).json" elseif solver ==  "dual_extrapolation" "../configurations/dual_extragradient_tuned.json" else "../configurations/default_0.json" end
    arglist = [
        "run",
        "--algorithm", solver,
        "--p", p,
        "--settings", input_file,
        "--frequency", if solver == "greenkhorn"
            "$(25 * size * size)"
        else
            "200"
        end,
        file1, file2]
    if solver != "greenkhorn"
        push!(arglist, "--cuda")
    end
    output_log = @capture_out begin
        run_from_arguments(arglist)
    end
    open(joinpath(output_directory, output_file), "w") do fout
        write(fout, output_log)
    end
end

