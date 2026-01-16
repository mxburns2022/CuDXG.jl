# CuDXG.jl

Implementation of the Dual Extragradient (DXG) algorithm for discrete Entropic Optimal Transport (EOT) as described in the an upcoming paper. The algorithm is a dual-only, linear-space implementation of the Primal-Dual extragradient method originally proposed by [1].

## Setup
Assuming Julia is installed, run `julia --project -e 'import Pkg;Pkg.instantiate()'`. Note that color transfer and barycenter utilities rely on CUDA acceleration.

## Usage
`CmdLine.jl` provides a command line interface with three subcommands

```shell
usage: cudxg [-h] {run|ctransfer|barycenter}

commands:
  run         Run discrete OT problem
  ctransfer   Perform color transfer using a metric kernel. CUDA is
              used by default.
  barycenter  Compute a Wasserstein barycenter for a collection of
              marginals. CUDA is used by default.

optional arguments:
  -h, --help  show this help message and exit
```
Note that each command allows for user-specified parameters via a `.json` configuration file. Example configurations can be found in `./configurations`.
- `eta_p`: Primal EOT regularization ($\eta$ used by all EOT algorithms)
- `eta_μ`: Dual EOT regularization (used by extragradient methods)
- `C1`, `C2`, `C3`: Constants used for setting parameters in extragradient methods
- `B`: Balancing parameter used in extragradient methods
- `itermax`: Maximum iteration count
- `epsilon`: Target accuracy
- `verbose`: Enable updates logging
- `tmax`: Timeout (in seconds)

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
                        are: apdagd, greenkhorn, dual_extragradient,
                        accelerated_sinkhorn, apdamd, hpd, sinkhorn,
                        primal_extragradient (default: "sinkhorn")
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

### `barycenter`
Compute the Wasserstein barycenter for a series of input marginals using a p-norm ground distance. Only $p=1$ and $p=2$ are supported.
```shell
usage: cudxg barycenter [-a ALGORITHM] [--settings SETTINGS]
                        [--frequency FREQUENCY] [--p P]
                        [--weights [WEIGHTS...]] [--cost COST]
                        [--supports SUPPORTS] --output OUTPUT
                        [--kernel] [-h] marginals...

positional arguments:
  marginals             Paths to target input image file (row
                        marginal)

optional arguments:
  -a, --algorithm ALGORITHM
                        Algorithm to solve the barycenter problem.
                        Options are: dual_extragradient, sinkhorn
                        (default: "sinkhorn")
  --settings SETTINGS   Solver configuration settings (default:
                        "./test.json")
  --frequency FREQUENCY
                        Printing frequency (type: Int64, default: 100)
  --p P                 p for distance computation (>= 10 for infinity
                        norm, 0 for uniform cost) (type: Float64,
                        default: 2)
  --weights [WEIGHTS...]
                        Weights for Barycenter objective (default is
                        uniform). If provided, number of weights must
                        match the number of distributions (type:
                        Float64)
  --cost COST           Path to cost matrix. If not provided, then
                        "--supports" must be provided and Euclidean
                        kernel will be used
  --supports SUPPORTS   Path to distribution supports for kernel
                        computation. If not provided, then "--cost"
                        must be provided.     Either one support must
                        be provided (common support) or the number of
                        supports must match the number of input
                        distributions
  --output OUTPUT       Output path for Wasserstein barycenter
  --kernel              Use
  -h, --help            show this help message and exit
```