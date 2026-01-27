using CuLAMP
using IterTools
using Suppressor
using Formatting
# file1 = "eot_images/gemini_flowers_newer.png"
# file2 = "eot_images/gemini_cityscape_newer.png"
# # bmarkdir = ENV["BENCH"] * "/DOTmark_1.0/Data/"
# file1 = bmarkdir * "ClassicImages/data32_1004.csv"
# file2 = bmarkdir * "ClassicImages/data32_1008.csv"
# file1 = "test1.csv"
# file2 = "test2.csv"


# arglist = [
#     "run",
#     "--algorithm", "sinkhorn",
#     "--settings", "test.json",
#     "--frequency", "1000",
#     # "--height", "512",
#     # "--width", "512",
#     "--cuda",
#     file1, file2,
#     "--output1", "eg_assignment1.txt",
#     "--output2", "eg_assignment2.txt",
#     "--potential-out", "potentials2.txt",
# ]
run_from_arguments(ARGS)