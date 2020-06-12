import numpy as np

from Dataset_generation import get_dataset
from Method_sgd_frugal_adaptive import get_sgd_procs
from Quantile_procs import get_procs
dt = get_dataset('gau_1', 100)

procs = get_procs(dt, [0.1, 0.5, 0.9], "SGD", 'const')

print (procs[:, -1])
