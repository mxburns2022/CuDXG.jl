# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.33 0.33 0.33 --output outtest_33_33_33 > data/barycenter64_33_33_33.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.25 0.25 0.5 --output outtest_25_25_5 > data/barycenter64_25_25_5.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.25 0.25 --output outtest_5_25_25 > data/barycenter64_5_25_25.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.25 0.5 0.25 --output outtest_25_5_25 > data/barycenter64_25_5_25.txt

# #______________

# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv  --weights 0.5 0.5 --output outtest_5_5_0 > data/barycenter64_5_5_0.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv  $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.5 --output outtest_5_0_5 > data/barycenter64_5_0_5.txt

# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv  $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.5 --output outtest_0_5_5 > data/barycenter64_0_5_5.txt


julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_9_1 --weights 0.9 0.1 > data/barycenter/extragrad_test_64_9_1.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_75_25 --weights 0.75 0.25 > data/barycenter/extragrad_test_64_75_25.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_5_5 --weights 0.5 0.5 > data/barycenter/extragrad_test_64_5_5.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_25_75 --weights 0.25 0.75 > data/barycenter/extragrad_test_64_25_75.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_1_9 --weights 0.1 0.9 > data/barycenter/extragrad_test_64_1_9.csv



julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_9_1 --weights 0.9 0.1 > data/barycenter/sinkhorn_test_64_9_1.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_75_25 --weights 0.75 0.25 > data/barycenter/sinkhorn_test_64_75_25.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_5_5 --weights 0.5 0.5 > data/barycenter/sinkhorn_test_64_5_5.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_25_75 --weights 0.25 0.75 > data/barycenter/sinkhorn_test_64_25_75.csv
julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_1_9 --weights 0.1 0.9 > data/barycenter/sinkhorn_test_64_1_9.csv