# CuLAMP.jl

Implementation of the Log-Averaged Mirror Prox (LAMP) algorithm for discrete (Entropic) Optimal Transport. The algorithm is a dual-only, linear-space implementation of the Primal-Dual Mirror Prox (PDMP) method originally proposed by [1].

## Setup
Assuming Julia is installed, run `julia --project -e 'import Pkg;Pkg.instantiate()'`. Note that color transfer and kernelized cost computations rely on CUDA acceleration.

## Usage
`CmdLine.jl` provides a command line interface with three subcommands

```shell
usage: julia --project CmdLine.jl [-h] {run|ctransfer|}

commands:
  run         Run discrete OT problem
  ctransfer   Perform color transfer using a metric kernel. CUDA is
              used by default.
  cellsim     Compute a streaming C-index over OT-scOmics cells using
              an on-the-fly feature cost kernel.

optional arguments:
  -h, --help  show this help message and exit
```
Note that each command allows for user-specified parameters via a `.json` configuration file. Example configurations can be found in `./configurations`.

### `run`
Subcommand for basic DOT problems (e.g. from DOTMark). Can either use user-provided weights or will construct using a $p$-norm metric. Defaults to Euclidean distances after interpreting the marginals on a 2D grid.
```shell
usage: culamp run [-a ALGORITHM] [--settings SETTINGS] [--cuda]
                  [--p P] [--weights WEIGHTS] [--kernel]
                  [--frequency FREQUENCY] [--output1 OUTPUT1]
                  [--output2 OUTPUT2] [--height HEIGHT]
                  [--width WIDTH] [--potential-out POTENTIAL-OUT] [-h]
                  file1 file2

positional arguments:
  file1                 Path to target DOTMark-formatted file (row
                        marginal) (TODO: Add support for more input
                        types)
  file2                 Path to target DOTMark-formatted file (col
                        marginal) (TODO: Add support for more input
                        types)

optional arguments:
  -a, --algorithm ALGORITHM
                        Algorithm to solve the DOT instance. Options
                        are: lamp, apdagd, acc_sinkhorn, greenkhorn,
                        apdamd, dextrap, pdmp, sinkhorn,
                        annealed_sinkhorn, hpd (default: "sinkhorn")
  --settings SETTINGS   Solver configuration settings (default:
                        "./test.json")
  --cuda                Use CUDA
  --p P                 p for distance computation (>= 10 for infinity
                        norm, 0 for uniform cost) (type: Float64,
                        default: 2.0)
  --weights WEIGHTS     Path to CSV-formatted weight matrix (default:
                        "")
  --kernel              Use kernels to compute OT matrices on the fly
                        (supported: lamp, sinkhorn, annealed_sinkhorn)
  --frequency FREQUENCY
                        Printing frequency (type: Int64, default: 100)
  --output1 OUTPUT1     Output path for assignment 1 (default: "")
  --output2 OUTPUT2     Output path for assignment 2 (default: "")
  --height HEIGHT       Image height (type: Int64, default:
                        9223372036854775807)
  --width WIDTH         Image width (type: Int64, default:
                        9223372036854775807)
  --potential-out POTENTIAL-OUT
                        Output path for dual potentials. Order is (1)
                        Simplex dual (if using extragradient), (2)
                        Potential for Row Marginal, (3) Potential for
                        Column Marginal> (default: "")
  -h, --help            show this help message and exit
```


### `ctransfer`
Subcommand for color transfer. Expects inputs to be `.png` files and will output `.png` files. Note that only `lamp` and `sinkhorn` are supported arguments for `--algorithm`.
```shell
usage: culamp ctransfer [-a ALGORITHM] [--settings SETTINGS]
                       [--frequency FREQUENCY] [--p P]
                       [--height HEIGHT] [--width WIDTH]
                       --output1 OUTPUT1 --output2 OUTPUT2 [-h] file1
                       file2

positional arguments:
  file1                 Path to target input image file (row marginal)
  file2                 Path to target input image file (column
                        marginal)

optional arguments:
  -a, --algorithm ALGORITHM
                        Algorithm to solve the color transfer
                        instance. Options are: dual_extragradient,
                        sinkhorn (default: "sinkhorn")
  --settings SETTINGS   Solver configuration settings (default:
                        "./test.json")
  --frequency FREQUENCY
                        Printing frequency (type: Int64, default: 100)
  --p P                 p for distance computation (>= 10 for infinity
                        norm, 0 for uniform cost) (type: Float64,
                        default: 2)
  --height HEIGHT       Image height (type: Int64, default: 128)
  --width WIDTH         Image width (type: Int64, default: 128)
  --output1 OUTPUT1     Output path for color mapped image 1
  --output2 OUTPUT2     Output path for color mapped image 2
  -h, --help            show this help message and exit
```

### `cellsim`
Subcommand for cell similarity computation. Expects inputs to be a `.csv` file and will output `.png` files. Note that only `lamp` and `sinkhorn`.
```shell
usage: culamp cellsim [-a ALGORITHM] [--settings SETTINGS]
                      [--frequency FREQUENCY] [--cost COST]
                      [--normalize-features NORMALIZE-FEATURES]
                      [--num-features NUM-FEATURES]
                      [--num-subproblems NUM-SUBPROBLEMS]
                      [--seed SEED] [--num-cells NUM-CELLS]
                      [--output OUTPUT] [-h] file

positional arguments:
  file                  Path to an OT-scOmics preprocessed CSV or
                        CSV.GZ file

optional arguments:
  -a, --algorithm ALGORITHM
                        Algorithm to solve the cell OT problem.
                        Options are: lamp, annealed_sinkhorn, sinkhorn
                        (default: "lamp")
  --settings SETTINGS   Solver configuration settings (default:
                        "./test.json")
  --frequency FREQUENCY
                        Progress printing frequency in number of cell
                        pairs (type: Int64, default: 100)
  --cost COST           Ground cost metric over features. Options are:
                        l1, l2, cosine, pearson, correlation (default:
                        "correlation")
  --normalize-features NORMALIZE-FEATURES
                        L1-normalize each feature across cells before
                        kernel distances are evaluated (type: Bool,
                        default: true)
  --num-features NUM-FEATURES
                        Number of features to include (type: Int64,
                        default: 9223372036854775807)
  --num-subproblems NUM-SUBPROBLEMS
                        Number of subproblems to run (limits pairwise
                        comparisons, just for benchmarking) (type:
                        Int64, default: 9223372036854775807)
  --seed SEED           Random seed for cell selection (type: Int64,
                        default: 0)
  --num-cells NUM-CELLS
                        Number of cells to include (type: Int64,
                        default: 9223372036854775807)
  --output OUTPUT       Optional output path for a one-line summary
                        (default: "")
  -h, --help            show this help message and exit
```

## Experiments and Data
The data used for plotting each figure in the main paper is provided in `data_archive`. Experiment code can be found in `experiments`, and the code to plot all figures can be found in `experiments/make_figures.ipynb`. Note that a compatible CUDA GPU is needed to run some experiemnts. A small example use case can be found in `example_problem.ipynb`.

The `experiments` folder assumes an environment variable `BENCH` exists which points to the parent of the DOTmark directory, e.g., `$BENCH/DOTmark_v1.0` exists. 