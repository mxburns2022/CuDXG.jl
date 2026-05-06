import torch
import pykeops
pykeops.test_torch_bindings()
from timeit import repeat
import pandas as pd
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
def test_speed(n:int):
    setup = f"""
import torch
from pykeops.torch import Genred
n={n}
device="{device}"
x = torch.rand(n*n,3, device=device, dtype=torch.double)
z = torch.rand(n*n, device=device, dtype=torch.double)
y = torch.rand_like(x)
formula = "-(Sum(Square(x-y)) + z) / p"
variables = [
    "x = Vi(3)",
    "y = Vj(3)",
    "z = Vj(1)",
    "p = Pm(1)"
]
p = torch.tensor([1e-4]).double().cuda()
my_routine = Genred(formula, variables, reduction_op="LogSumExp", axis=1)
"""
    tval = np.array(repeat(stmt="c = my_routine(x, y, z, p, backend=\"GPU\")", setup=setup, repeat=40, number=500)) / 500
    times = tval
    return np.mean(times), np.median(times), np.std(times)
# n = 128
data = []
for n in [32, 64, 128, 256]:
    values = test_speed(n)
    print(values)
    data.append((n, *values))
df = pd.DataFrame(data, columns=["n", "mean_time", "med_time", "std_time"])
df.to_csv("data/keops_time.csv", index=False)