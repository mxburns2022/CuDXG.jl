# CuLAMP.jl

Implementation of the Log-Averaged Mirror Prox (LAMP) algorithm for discrete (Entropic) Optimal Transport. The algorithm is a dual-only, linear-space implementation of the Primal-Dual Mirror Prox (PDMP) method originally proposed by [1].

## Setup
Assuming Julia is installed, run `julia --project -e 'import Pkg;Pkg.instantiate()'`. Note that color transfer and kernelized cost computations rely on CUDA acceleration.

## Usage
`CmdLine.jl` provides a command line interface with three subcommands

```shell
usage: cudxg [-h] {run|ctransfer}

commands:
  run         Run discrete OT problem
  ctransfer   Perform color transfer using a metric kernel. CUDA is
              used by default.

optional arguments:
  -h, --help  show this help message and exit
```
Note that each command allows for user-specified parameters via a `.json` configuration file. Example configurations can be found in `./configurations`.

### `run`
Subcommand for basic DOT problems (e.g. from DOTMark). Can either use user-provided weights or will construct using a $p$-norm metric. Defaults to Euclidean distances after interpreting the marginals on a 2D grid.
```shell
usage: cudxg run [-a ALGORITHM] [--settings SETTINGS] [--cuda] [--p P]
                 [--weights WEIGHTS] [--kernel]
                 [--frequency FREQUENCY] [--output1 OUTPUT1]
                 [--output2 OUTPUT2] [--potential-out POTENTIAL-OUT]
                 [-h] file1 file2

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
                        are: apdagd, greenkhorn, lamp, pdmp,
                        acc_sinkhorn, apdamd, hpd, sinkhorn (default: "sinkhorn")
  --settings SETTINGS   Solver configuration settings (default:
                        "./test.json")
  --cuda                Use CUDA
  --p P                 p for distance computation (>= 10 for infinity
                        norm, 0 for uniform cost) (type: Float64)
  --weights WEIGHTS     Path to CSV-formatted weight matrix (default:
                        "")
  --kernel              Use kernels to compute OT matrices on the fly
                        (only dual_extragradient and sinkhorn are
                        supported)
  --frequency FREQUENCY
                        Printing frequency (type: Int64, default: 100)
  --output1 OUTPUT1     Output path for assignment 1 (default: "")
  --output2 OUTPUT2     Output path for assignment 2 (default: "")
  --potential-out POTENTIAL-OUT
                        Output path for dual potentials. Order is (1)
                        Simplex dual (if using extragradient), (2)
                        Potential for Row Marginal, (3) Potential for
                        Column Marginal> (default: "")
  -h, --help            show this help message and exit
```


### `ctransfer`
Subcommand for color transfer. Expects inputs to be `.png` files and will output `.png` files. Note that only `dual_extragradient` and `sinkhorn` are supported arguments for `--algorithm`.
```shell
usage: cudxg ctransfer [-a ALGORITHM] [--settings SETTINGS]
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


## Experiments and Data
The data used for plotting each figure in the main paper is provided in `data_archive`. Experiment code can be found in `experiments`, and the code to plot all figures can be found in `experiments/make_figures.ipynb`.

The `experiments` folder assumes an environment variable `BENCH` exists which points to the parent of the DOTmark directory, e.g., `$BENCH/DOTmark_v1.0` exists. 