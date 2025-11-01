using CuTransferEOT
using IterTools
using Suppressor
using Formatting
benchmark_directory = ENV["BENCH"] * "/DOTmark_1.0/Data"
output_directory = "data"
if !isdir(output_directory)
    mkdir(output_directory)
end

"""
Run DOT solvers on selected problems from the DOTmark_1.0 dataset
"""
benchmark_classes = ["ClassicImages"]
sizes = [32, 64, 128, 256]
instance_pairs = [(2, 6)]
algorithms = ["dual_extragradient"]
epsilon = [1e-6]
deleteat!(algorithms, findall(algorithms .== "apdagd"))

for (size, (ind1, ind2), solver, probclass, eps) in product(sizes, instance_pairs, algorithms, benchmark_classes, epsilon)
    println((size, (ind1, ind2), solver, probclass, eps))
    file1 = joinpath(benchmark_directory, probclass, "data$(size)_10$(lpad(ind1, 2, '0')).csv")
    file2 = joinpath(benchmark_directory, probclass, "data$(size)_10$(lpad(ind2, 2, '0')).csv")
    epsstring = sprintf1("%1.0e", eps)
    # Format ouutput file as <problem_class>_<resolution>_<index of img1>_<index of img2>_<η value>_<solver_name>_log.csv
    output_file = join([lowercase(probclass), size, ind1, ind2, epsstring, solver], "_") * "_log.csv"
    input_file = "configurations/default_$(epsstring).json"
    arglist = [
        "--algorithm", solver,
        "--settings", input_file,
        "--frequency", if solver == "greenkhorn"
            "$(25 * size * size)"
        else
            "25"
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

