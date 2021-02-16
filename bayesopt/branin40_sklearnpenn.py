from argparse import Namespace
import numpy as np

from probo import NelderMeadAcqOptimizer, SimpleBo
from examples.branin.branin import branin, get_branin_domain_nd
from penn_sklearn import SklearnPenn


# define function
ndimx = 40
f = lambda x: np.sum([branin(x[2 * i : 2 * i + 2]) for i in range(ndimx // 2)])

# define model
model = SklearnPenn()

# define acqfunction
acqfunction = {'acq_str': 'ts', 'n_gen': 500}

# define acqoptimizer
domain = get_branin_domain_nd(ndimx)
acqoptimizer = NelderMeadAcqOptimizer(
    {'rand_every': 10, 'max_iter': 200, 'jitter': True}, domain
)

# define  initial dataset
n_init = 80
data = Namespace()
data.x = domain.unif_rand_sample(n_init)
data.y = [f(x) for x in data.x]

# define and run BO
bo = SimpleBo(
    f, model, acqfunction, acqoptimizer, data=data, params={'n_iter': 120}, seed=11
)
results = bo.run()
