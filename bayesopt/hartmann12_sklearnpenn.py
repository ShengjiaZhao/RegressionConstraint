from argparse import Namespace
import numpy as np

from probo import NelderMeadAcqOptimizer, SimpleBo
from hartmann import hartmann_nd, get_hartmann_domain_nd
from penn_sklearn import SklearnPenn


# define function
n_dim = 12
f = hartmann_nd

# define model
model = SklearnPenn()

# define acqfunction
acqfunction = {'acq_str': 'ts', 'n_gen': 500}

# define acqoptimizer
domain = get_hartmann_domain_nd(n_dim)
acqoptimizer = NelderMeadAcqOptimizer(
    {'rand_every': 10, 'max_iter': 200, 'jitter': True}, domain
)

# define  initial dataset
n_init = 100
data = Namespace()
data.x = domain.unif_rand_sample(n_init)
data.y = [f(x) for x in data.x]

# define and run BO
bo = SimpleBo(
    f, model, acqfunction, acqoptimizer, data=data, params={'n_iter': 200}, seed=11
)
results = bo.run()
