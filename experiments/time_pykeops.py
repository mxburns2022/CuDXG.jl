import torch
import pykeops
pykeops.test_torch_bindings()
from timeit import timeit
import pandas as pd


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
    tval = timeit(stmt="c = my_routine(x, y, z, p, backend=\"GPU\")", setup=setup, number=500)
    return tval/500
# n = 128
data = []
for n in [32, 64, 128, 256]:
    print(n, test_speed(n))
    data.append((n, test_speed(n)))
df = pd.DataFrame
