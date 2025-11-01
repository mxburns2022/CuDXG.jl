# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.33 0.33 0.33 --output outtest_33_33_33 > data/barycenter64_33_33_33.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.25 0.25 0.5 --output outtest_25_25_5 > data/barycenter64_25_25_5.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.25 0.25 --output outtest_5_25_25 > data/barycenter64_5_25_25.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.25 0.5 0.25 --output outtest_25_5_25 > data/barycenter64_25_5_25.txt

# #______________

# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv  --weights 0.5 0.5 --output outtest_5_5_0 > data/barycenter64_5_5_0.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1007.csv  $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.5 --output outtest_5_0_5 > data/barycenter64_5_0_5.txt

# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/Shapes/data64_1008.csv  $BENCH/DOTmark_1.0/Data/Shapes/data64_1009.csv --weights 0.5 0.5 --output outtest_0_5_5 > data/barycenter64_0_5_5.txt


# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_9_1 --weights 0.9 0.1 > data/barycenter/extragrad_test_64_9_1.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_75_25 --weights 0.75 0.25 > data/barycenter/extragrad_test_64_75_25.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_5_5 --weights 0.5 0.5 > data/barycenter/extragrad_test_64_5_5.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_25_75 --weights 0.25 0.75 > data/barycenter/extragrad_test_64_25_75.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm dual_extragradient --output data/barycenter/extragrad_test_64_1_9 --weights 0.1 0.9 > data/barycenter/extragrad_test_64_1_9.csv



# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_9_1 --weights 0.9 0.1 > data/barycenter/sinkhorn_test_64_9_1.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_75_25 --weights 0.75 0.25 > data/barycenter/sinkhorn_test_64_75_25.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_5_5 --weights 0.5 0.5 > data/barycenter/sinkhorn_test_64_5_5.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_25_75 --weights 0.25 0.75 > data/barycenter/sinkhorn_test_64_25_75.csv
# julia --project SingleRunTest.jl barycenter $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1001.csv $BENCH/DOTmark_1.0/Data/ClassicImages/data64_1010.csv --algorithm sinkhorn --output data/barycenter/sinkhorn_test_64_1_9 --weights 0.1 0.9 > data/barycenter/sinkhorn_test_64_1_9.csv




# julia --project SingleRunTest.jl run data10_1001.txt data10_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_10.csv
# julia --project SingleRunTest.jl run data20_1001.txt data20_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_20.csv
# julia --project SingleRunTest.jl run data30_1001.txt data30_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_30.csv
# julia --project SingleRunTest.jl run data40_1001.txt data40_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_40.csv
# julia --project SingleRunTest.jl run data50_1001.txt data50_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_50.csv
# julia --project SingleRunTest.jl run data60_1001.txt data60_1010.txt --algorithm dual_extragradient --cuda > data/scaling/run_60.csv

julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm dual_extragradient --cuda --p 1 > data/scaling/ms_run_10p1_n7.csv
julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm dual_extragradient --cuda --p 1 > data/scaling/ms_run_20p1_n7.csv
julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm dual_extragradient --cuda --p 1 > data/scaling/ms_run_30p1_n7.csv
julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm dual_extragradient --cuda --p 1 > data/scaling/ms_run_40p1_n7.csv


julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm dual_extragradient --cuda --p 2 > data/scaling/ms_run_10p2_n7.csv
julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm dual_extragradient --cuda --p 2 > data/scaling/ms_run_20p2_n7.csv
julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm dual_extragradient --cuda --p 2 > data/scaling/ms_run_30p2_n7.csv
julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm dual_extragradient --cuda --p 2 > data/scaling/ms_run_40p2_n7.csv

julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm dual_extragradient --cuda --p 3 > data/scaling/ms_run_10p3_n7.csv
julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm dual_extragradient --cuda --p 3 > data/scaling/ms_run_20p3_n7.csv
julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm dual_extragradient --cuda --p 3 > data/scaling/ms_run_30p3_n7.csv
julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm dual_extragradient --cuda --p 3 > data/scaling/ms_run_40p3_n7.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm apdamd --cuda --p 1 > data/scaling/ms_run_10p1_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm apdamd --cuda --p 1 > data/scaling/ms_run_20p1_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm apdamd --cuda --p 1 > data/scaling/ms_run_30p1_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm apdamd --cuda --p 1 > data/scaling/ms_run_40p1_2_apdamd.csv


# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm apdamd --cuda --p 2 > data/scaling/ms_run_10p2_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm apdamd --cuda --p 2 > data/scaling/ms_run_20p2_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm apdamd --cuda --p 2 > data/scaling/ms_run_30p2_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm apdamd --cuda --p 2 > data/scaling/ms_run_40p2_2_apdamd.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm apdamd --cuda --p 3 > data/scaling/ms_run_10p3_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm apdamd --cuda --p 3 > data/scaling/ms_run_20p3_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm apdamd --cuda --p 3 > data/scaling/ms_run_30p3_2_apdamd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm apdamd --cuda --p 3 > data/scaling/ms_run_40p3_2_apdamd.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm sinkhorn --cuda --p 1 > data/scaling/ms_run_10p1_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm sinkhorn --cuda --p 1 > data/scaling/ms_run_20p1_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm sinkhorn --cuda --p 1 > data/scaling/ms_run_30p1_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm sinkhorn --cuda --p 1 > data/scaling/ms_run_40p1_sinkhorn.csv


# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm sinkhorn --cuda --p 2 > data/scaling/ms_run_10p2_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm sinkhorn --cuda --p 2 > data/scaling/ms_run_20p2_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm sinkhorn --cuda --p 2 > data/scaling/ms_run_30p2_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm sinkhorn --cuda --p 2 > data/scaling/ms_run_40p2_2_sinkhorn.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm sinkhorn --cuda --p 3 > data/scaling/ms_run_10p3_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm sinkhorn --cuda --p 3 > data/scaling/ms_run_20p3_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm sinkhorn --cuda --p 3 > data/scaling/ms_run_30p3_2_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm sinkhorn --cuda --p 3 > data/scaling/ms_run_40p3_2_sinkhorn.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm accelerated_sinkhorn --cuda --p 1 > data/scaling/ms_run_10p1_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm accelerated_sinkhorn --cuda --p 1 > data/scaling/ms_run_20p1_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm accelerated_sinkhorn --cuda --p 1 > data/scaling/ms_run_30p1_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm accelerated_sinkhorn --cuda --p 1 > data/scaling/ms_run_40p1_accelerated_sinkhorn.csv


# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm accelerated_sinkhorn --cuda --p 2 > data/scaling/ms_run_10p2_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm accelerated_sinkhorn --cuda --p 2 > data/scaling/ms_run_20p2_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm accelerated_sinkhorn --cuda --p 2 > data/scaling/ms_run_30p2_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm accelerated_sinkhorn --cuda --p 2 > data/scaling/ms_run_40p2_2_accelerated_sinkhorn.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm accelerated_sinkhorn --cuda --p 3 > data/scaling/ms_run_10p3_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm accelerated_sinkhorn --cuda --p 3 > data/scaling/ms_run_20p3_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm accelerated_sinkhorn --cuda --p 3 > data/scaling/ms_run_30p3_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm accelerated_sinkhorn --cuda --p 3 > data/scaling/ms_run_40p3_2_accelerated_sinkhorn.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm dual_extragradient --cuda > data/scaling/ms_run_40p1.csv
# julia --project SingleRunTest.jl run ms_data50_1001.txt ms_data50_1010.txt --algorithm dual_extragradient --cuda > data/scaling/ms_run_50.csv
# julia --project SingleRunTest.jl run ms_data60_1001.txt ms_data60_1010.txt --algorithm dual_extragradient --cuda > data/scaling/ms_run_60.csv
# julia --project SingleRunTest.jl run data10_1001.txt data10_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_10_sk.csv
# julia --project SingleRunTest.jl run data20_1001.txt data20_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_20_sk.csv
# julia --project SingleRunTest.jl run data30_1001.txt data30_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_30_sk.csv
# julia --project SingleRunTest.jl run data40_1001.txt data40_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_40_sk.csv
# julia --project SingleRunTest.jl run data50_1001.txt data50_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_50_sk.csv
# julia --project SingleRunTest.jl run data60_1001.txt data60_1010.txt --algorithm sinkhorn --cuda > data/scaling/run_60_sk.csv


# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm hpd --cuda --p 1 > data/scaling/ms_run_10p1_hpd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm hpd --cuda --p 1 > data/scaling/ms_run_20p1_hpd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm hpd --cuda --p 1 > data/scaling/ms_run_30p1_hpd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm hpd --cuda --p 1 > data/scaling/ms_run_40p1_hpd.csv


# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm hpd --cuda --p 2 > data/scaling/ms_run_10p2_hpd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm hpd --cuda --p 2 > data/scaling/ms_run_20p2_hpd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm hpd --cuda --p 2 > data/scaling/ms_run_30p2_hpd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm hpd --cuda --p 2 > data/scaling/ms_run_40p2_hpd.csv

# julia --project SingleRunTest.jl run ms_data10_1001.txt ms_data10_1010.txt --algorithm hpd --cuda --p 3 > data/scaling/ms_run_10p3_hpd.csv
# julia --project SingleRunTest.jl run ms_data20_1001.txt ms_data20_1010.txt --algorithm hpd --cuda --p 3 > data/scaling/ms_run_20p3_hpd.csv
# julia --project SingleRunTest.jl run ms_data30_1001.txt ms_data30_1010.txt --algorithm hpd --cuda --p 3 > data/scaling/ms_run_30p3_hpd.csv
# julia --project SingleRunTest.jl run ms_data40_1001.txt ms_data40_1010.txt --algorithm hpd --cuda --p 3 > data/scaling/ms_run_40p3_hpd.csv